"""任务执行引擎"""

from __future__ import annotations

from collections.abc import Sequence
import copy
from dataclasses import dataclass
import gc
from pathlib import Path
import re
import tomllib
from typing import Any, cast

from deepmerge import Merger
import torch

from .base import BaseTaskConfig
from .imports import import_files
from .runtime import format_output_path
from .tasks import get_task

_merger = Merger([(dict, ['merge'])], ['override'], ['override'])
_RUN_ID_PATTERN = re.compile(r'[A-Za-z0-9._-]+')
# --runs 中表示“所有默认运行的 run”的保留字，不能用作 run id
DEFAULT_RUNS = 'default'


def _parse_imports(raw: Any, path: Path) -> list[Path]:
	"""解析 imports 字段，相对当前 TOML 文件所在目录解析为绝对路径"""
	if raw is None:
		return []
	if not isinstance(raw, list) or not all(isinstance(p, str) and p for p in raw):
		raise ValueError(f"Invalid 'imports' in task file: {path}. Expected list of non-empty path strings.")
	return [(path.parent / p).resolve() for p in raw]


def _parse_remove(raw: Any, where: str) -> list[tuple[str, ...]]:
	"""解析 _remove 字段：点分隔的键路径列表（如 'model.checkpoint_path'）"""
	if raw is None:
		return []
	if not isinstance(raw, list) or not all(isinstance(k, str) and k and all(k.split('.')) for k in raw):
		raise ValueError(f"Invalid '_remove' in {where}. Expected list of dotted key paths such as 'model.checkpoint_path'.")
	return [tuple(k.split('.')) for k in raw]


def _remove_keys(data: dict[str, Any], keys: list[tuple[str, ...]], where: str) -> None:
	"""从原始配置字典中删除键，使其回到上一层的值或配置模型默认值；键不存在时报错"""
	for key in keys:
		node: Any = data
		for part in key[:-1]:
			node = node.get(part) if isinstance(node, dict) else None
		if not isinstance(node, dict) or key[-1] not in node:
			raise ValueError(f"Cannot remove '{'.'.join(key)}' in {where}: key is not set by the inherited configuration")
		del node[key[-1]]


@dataclass(slots=True, frozen=True)
class _RunEntry:
	"""run 数组的单个条目"""

	run_id: str
	default: bool  # 未指定 --runs（或 --runs 含 default）时是否运行
	remove: list[tuple[str, ...]]  # 合并前从基础配置中删除的键
	overrides: dict[str, Any]  # 深度合并到基础配置上的重载内容
	source: dict[str, Any]  # 条目去掉 id 后的原文，记录到快照


def _parse_runs(raw: Any, path: Path) -> list[_RunEntry]:
	"""解析 run 字段

	Args:
		raw: 合并继承链后的 run 字段值，未声明时为 None
		path: 当前 TOML 文件路径，用于报错信息
	"""
	if raw is None:
		return []
	if not isinstance(raw, list) or not raw or not all(isinstance(entry, dict) for entry in raw):
		raise ValueError(f"Invalid 'run' in task file: {path}. Expected non-empty array of tables ([[run]]).")

	runs: list[_RunEntry] = []
	for i, entry in enumerate(raw):
		source = dict(entry)
		run_id = source.pop('id', None)
		if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
			raise ValueError(
				f"Invalid run id {run_id!r} at run[{i}] in task file: {path}. "
				"Expected non-empty string of letters, digits, '.', '_' or '-'."
			)
		if run_id == DEFAULT_RUNS:
			raise ValueError(f"Run id '{DEFAULT_RUNS}' at run[{i}] in task file: {path} is reserved for selecting all default runs")
		if any(run_id == existing.run_id for existing in runs):
			raise ValueError(f"Duplicate run id '{run_id}' in task file: {path}")
		overrides = dict(source)
		default = overrides.pop('default', True)
		if not isinstance(default, bool):
			raise ValueError(f"Invalid 'default' in run '{run_id}' of task file: {path}. Expected boolean.")
		remove = _parse_remove(overrides.pop('_remove', None), f"run '{run_id}' of task file: {path}")
		runs.append(_RunEntry(run_id=run_id, default=default, remove=remove, overrides=overrides, source=source))
	return runs


def _load_raw_config(path: Path, visited: tuple[Path, ...] = ()) -> tuple[dict[str, Any], list[Path]]:
	"""递归加载 TOML 文件及其继承链，返回合并后的原始字典与 imports 路径列表。

	Args:
		path: 当前 TOML 文件路径
		visited: 继承链中已访问祖先路径元组，用于检测继承循环

	Returns:
		合并后的原始配置字典，以及父级在前、去重后的 imports 绝对路径列表
	"""
	# 第一阶段：循环继承检测
	resolved_path = path.resolve()
	if resolved_path in visited:
		chain = ' -> '.join(str(p) for p in (*visited, resolved_path))
		raise ValueError(f'Cyclic inheritance detected: {chain}')

	# 第二阶段：读取当前文件数据
	with open(path, 'rb') as f:
		data = tomllib.load(f)
	imports = _parse_imports(data.pop('imports', None), path)
	remove = _parse_remove(data.pop('_remove', None), f'task file: {path}')

	# 第三阶段：解析 extends 字段并递归加载父级配置；_remove 先从父级配置中删除键，再合并当前文件
	extends = data.pop('extends', None)
	if extends is not None:
		if not isinstance(extends, str) or not extends:
			raise ValueError(f"Invalid 'extends' in task file: {path}. Expected non-empty string.")

		parent_path = (path.parent / extends).resolve()
		parent_data, parent_imports = _load_raw_config(parent_path, (*visited, resolved_path))
		_remove_keys(parent_data, remove, f'task file: {path}')
		merged_imports = list(dict.fromkeys([*parent_imports, *imports]))
		return _merger.merge(parent_data, data), merged_imports

	if remove:
		raise ValueError(f"'_remove' in task file: {path} requires 'extends' (it removes inherited keys)")
	return data, list(dict.fromkeys(imports))


def _build_config(data: dict[str, Any], path: Path, imports: list[Path]) -> BaseTaskConfig:
	"""按 task_type 选择配置类，校验原始字典并设置运行时元信息"""
	task_type = data.get('task_type')
	if task_type is None:
		raise ValueError(f"Missing 'task_type' in task file: {path}. Please specify a task type.")
	if not isinstance(task_type, str) or not task_type:
		raise ValueError(f"Invalid 'task_type' in task file: {path}. Expected non-empty string.")

	config_cls, _ = get_task(task_type)
	config = config_cls.model_validate(data)
	config.task_file = path
	config.task_id = path.stem
	config.imports = tuple(imports)
	return config


def load_config(path: str | Path) -> BaseTaskConfig:
	"""从 TOML 文件加载任务配置

	根据 TOML 中的 task_type 字段自动选择正确的配置类。
	支持通过 top-level extends 字段继承父级 TOML 配置，
	通过 top-level imports 字段导入注册外部 task / feeder 的 Python 文件。
	run 数组（若有）被忽略，返回不含任何 run 重载的基础配置；需要逐个 run 的配置时使用 load_run_configs。

	Args:
		path: TOML 文件路径

	Returns:
		对应任务类型的配置对象
	"""
	path = Path(path)
	data, imports = _load_raw_config(path)
	import_files(imports)
	data.pop('run', None)
	return _build_config(data, path, imports)


def load_run_configs(path: str | Path) -> list[BaseTaskConfig]:
	"""从 TOML 文件加载全部 run 的任务配置

	run 数组的每个条目是一次参数重载：先按条目的 _remove 从基础配置中删除键（回到配置模型默认值），
	再将其余内容深度合并到基础配置上（语义同 extends）。条目的 default（默认 true）表示未指定 --runs 时是否运行。
	run 随 extends 继承，子文件声明 run 时整体替换父级的 run。基础配置（不含任何条目）须能单独通过校验。

	Args:
		path: TOML 文件路径

	Returns:
		按 run 数组顺序排列的配置列表（run_id、run_overrides、run_default 已设置）；无 run 时为只含基础配置的列表
	"""
	path = Path(path)
	data, imports = _load_raw_config(path)
	import_files(imports)
	runs = _parse_runs(data.pop('run', None), path)
	base = _build_config(data, path, imports)
	if not runs:
		return [base]

	configs: list[BaseTaskConfig] = []
	for run in runs:
		merged = copy.deepcopy(data)
		_remove_keys(merged, run.remove, f"run '{run.run_id}' of task file: {path}")
		merged = _merger.merge(merged, copy.deepcopy(run.overrides))
		config = _build_config(merged, path, imports)
		config.run_id = run.run_id
		config.run_overrides = run.source
		config.run_default = run.default
		configs.append(config)
	return configs


def select_runs(configs: Sequence[BaseTaskConfig], run_ids: Sequence[str]) -> list[BaseTaskConfig]:
	"""按 run id 选择 run（保持 TOML 中的顺序）

	run_ids 为空等价于 ['default']；其中的 'default' 表示所有默认运行（default = true）的 run，可与其他 id 组合。
	"""
	selected = set(run_ids or [DEFAULT_RUNS])
	explicit = selected - {DEFAULT_RUNS}
	available = [config.run_id for config in configs]
	if explicit and not any(available):
		raise ValueError("Run ids were given but the task file declares no 'run' array")
	unknown = sorted(explicit - set(available))
	if unknown:
		raise ValueError(f'Unknown run id(s): {", ".join(unknown)}. Available: {", ".join(available)}')
	chosen = [c for c in configs if c.run_id in explicit or (DEFAULT_RUNS in selected and c.run_default)]
	if not chosen:
		raise ValueError(f"No run selected: the task file has no default run. Available: {', '.join(available)}")
	return chosen


def run_task(
	config: BaseTaskConfig | str | Path,
	device: torch.device | str | None = None,
) -> Path:
	"""运行任务

	Args:
		config: BaseTaskConfig 对象或 TOML 文件路径
		device: 设备，默认自动选择 CUDA/CPU

	Returns:
		输出路径
	"""
	if isinstance(config, (str, Path)):
		config = load_config(config)

	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	elif isinstance(device, str):
		device = torch.device(device)

	_, run_fn = get_task(config.task_type)
	return run_fn(config, device)


def run_tasks(
	configs: Sequence[BaseTaskConfig],
	device: torch.device | str | None = None,
) -> list[Path]:
	"""在同一进程中顺序运行多个 run，任一 run 报错立即终止

	运行前检查各 run 渲染后的输出路径互不相同，避免结果互相覆盖。

	Args:
		configs: 各 run 的最终配置
		device: 设备，默认自动选择 CUDA/CPU

	Returns:
		各 run 的输出路径
	"""
	if not configs:
		raise ValueError('run_tasks requires at least one config')
	paths = [format_output_path(cast(Any, config).output.save_path, config).resolve() for config in configs]
	duplicates = sorted({str(p) for p in paths if paths.count(p) > 1})
	if duplicates:
		raise ValueError(
			f'Multiple runs resolve to the same output path: {", ".join(duplicates)}. '
			'Distinguish runs in output.save_path, e.g. with {run_id}.'
		)

	outputs: list[Path] = []
	for i, config in enumerate(configs):
		if len(configs) > 1:
			print(f'\n##### Run {i + 1}/{len(configs)}: {config.run_id or config.task_id} #####')
		outputs.append(run_task(config, device))
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	return outputs
