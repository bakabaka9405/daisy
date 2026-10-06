"""任务执行引擎"""

from __future__ import annotations

from collections.abc import Sequence
import copy
from dataclasses import dataclass
import gc
from itertools import product
from pathlib import Path
import re
import tomllib
from typing import Any, cast

from deepmerge import Merger
import icontract
import torch

from .base import BaseTaskConfig
from .imports import import_files
from .runtime import format_output_path
from .tasks import get_task

_merger = Merger([(dict, ['merge'])], ['override'], ['override'])
_RUN_ID_PATTERN = re.compile(r'[A-Za-z0-9._-]+')
# --runs 中表示“所有默认运行的 run”的保留字，不能用作 run id
DEFAULT_RUNS = 'default'


def _is_dotted_key_list(raw: Any) -> bool:
	"""raw 是否为点分隔键路径（各段非空）的列表"""
	return isinstance(raw, list) and all(isinstance(k, str) and k and all(k.split('.')) for k in raw)


@icontract.require(
	lambda raw: raw is None or (isinstance(raw, list) and all(isinstance(p, str) and p for p in raw)),
	"Invalid 'imports' in task file. Expected list of non-empty path strings.",
)
def _parse_imports(raw: Any, path: Path) -> list[Path]:
	"""解析 imports 字段，相对当前 TOML 文件所在目录解析为绝对路径"""
	if raw is None:
		return []
	return [(path.parent / p).resolve() for p in raw]


@icontract.require(
	lambda raw: raw is None or _is_dotted_key_list(raw),
	"Invalid '_remove'. Expected list of dotted key paths such as 'model.checkpoint_path'.",
)
def _parse_remove(raw: Any, where: str) -> list[tuple[str, ...]]:
	"""解析 _remove 字段：点分隔的键路径列表（如 'model.checkpoint_path'）

	where 仅用于违约信息中标明出处。
	"""
	if raw is None:
		return []
	return [tuple(k.split('.')) for k in raw]


def _remove_keys(data: dict[str, Any], keys: list[tuple[str, ...]], where: str) -> None:
	"""按顺序遍历并删除键，缺失时立即报告键路径及出处。"""
	for key in keys:
		node: Any = data
		for part in key[:-1]:
			if not isinstance(node, dict) or part not in node:
				raise ValueError(f"Cannot remove key '{'.'.join(key)}' in {where}: key is not set by the inherited configuration")
			node = node[part]
		if not isinstance(node, dict) or key[-1] not in node:
			raise ValueError(f"Cannot remove key '{'.'.join(key)}' in {where}: key is not set by the inherited configuration")
		del node[key[-1]]


def _is_table_array(raw: Any) -> bool:
	"""raw 是否为非空的表数组（[[run]]）"""
	return isinstance(raw, list) and len(raw) > 0 and all(isinstance(entry, dict) for entry in raw)


def _has_valid_run_ids(raw: Any) -> bool:
	"""raw 是否为表数组，且各条目的 id 均为由字母、数字、'.'、'_'、'-' 组成的非空字符串"""
	return _is_table_array(raw) and all(
		isinstance(run_id := entry.get('id'), str) and _RUN_ID_PATTERN.fullmatch(run_id) is not None for entry in raw
	)


@icontract.require(lambda raw: isinstance(raw, dict), "Invalid 'grid' in run. Expected a table.")
def _parse_grid(raw: Any, where: str) -> list[tuple[tuple[str, ...], list[Any]]]:
	"""按 TOML 插入顺序递归收集路径及候选值，候选内部结构保持原样。"""
	axes: list[tuple[tuple[str, ...], list[Any]]] = []

	def collect(node: dict[str, Any], prefix: tuple[str, ...]) -> None:
		for name, value in node.items():
			key = prefix + (name,)
			if isinstance(value, dict) and value:
				collect(value, key)
			else:
				if not isinstance(value, list) or not value:
					raise ValueError(f"Grid axis '{'.'.join(key)}' in {where} must be a non-empty list of candidate values")
				axes.append((key, value))

	collect(raw, ())
	return axes


@dataclass(slots=True, frozen=True)
class _RunEntry:
	"""run 数组的单个条目"""

	run_id: str
	default: bool  # 未指定 --runs（或 --runs 含 default）时是否运行
	remove: list[tuple[str, ...]]  # 合并前从基础配置中删除的键
	overrides: dict[str, Any]  # 深度合并到基础配置上的重载内容
	grid: list[tuple[tuple[str, ...], list[Any]]]  # 按声明顺序展开的参数轴
	source: dict[str, Any]  # 条目去掉 id 后的原文，记录到快照


@icontract.require(
	lambda raw: not _is_table_array(raw) or all(isinstance(entry.get('default', True), bool) for entry in raw),
	"Invalid 'default' in run. Expected boolean.",
)
@icontract.require(
	lambda raw: not _has_valid_run_ids(raw) or len({entry['id'] for entry in raw}) == len(raw), 'Duplicate run id in task file'
)
@icontract.require(
	lambda raw: not _is_table_array(raw) or all(entry.get('id') != DEFAULT_RUNS for entry in raw),
	f"Run id '{DEFAULT_RUNS}' is reserved for selecting all default runs",
)
@icontract.require(
	lambda raw: not _is_table_array(raw) or _has_valid_run_ids(raw),
	"Invalid run id. Expected non-empty string of letters, digits, '.', '_' or '-'.",
)
@icontract.require(lambda raw: raw is None or _is_table_array(raw), "Invalid 'run' in task file. Expected non-empty array of tables ([[run]]).")
@icontract.ensure(lambda raw, result: len(result) == (0 if raw is None else len(raw)))
def _parse_runs(raw: Any, path: Path) -> list[_RunEntry]:
	"""解析 run 字段

	Args:
		raw: 合并继承链后的 run 字段值，未声明时为 None
		path: 当前 TOML 文件路径，用于报错信息
	"""
	if raw is None:
		return []

	runs: list[_RunEntry] = []
	for entry in raw:
		source = dict(entry)
		run_id = source.pop('id')
		overrides = dict(source)
		default = overrides.pop('default', True)
		remove = _parse_remove(overrides.pop('_remove', None), f"run '{run_id}' of task file: {path}")
		grid = _parse_grid(overrides.pop('grid', {}), f"run '{run_id}' of task file: {path}")
		runs.append(_RunEntry(run_id=run_id, default=default, remove=remove, overrides=overrides, grid=grid, source=source))
	return runs


def _load_raw_config(path: Path, visited: tuple[Path, ...] = ()) -> tuple[dict[str, Any], list[Path]]:
	"""递归加载 TOML 文件及其继承链，返回合并后的原始字典与 imports 路径列表。

	Args:
		path: 当前 TOML 文件路径
		visited: 继承链中已访问祖先路径元组，用于检测继承循环

	Returns:
		合并后的原始配置字典，以及父级在前、去重后的 imports 绝对路径列表
	"""
	# 第一阶段：循环继承检测。icontract 在递归调用中不检查契约，因此不能写成前置条件
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


@icontract.require(
	lambda data: isinstance(data['task_type'], str) and len(data['task_type']) > 0, "Invalid 'task_type' in task file. Expected non-empty string."
)
@icontract.require(lambda data: data.get('task_type') is not None, "Missing 'task_type' in task file. Please specify a task type.")
def _build_config(data: dict[str, Any], path: Path, imports: list[Path]) -> BaseTaskConfig:
	"""按 task_type 选择配置类，校验原始字典并设置运行时元信息"""
	config_cls, _ = get_task(data['task_type'])
	config = config_cls.model_validate(data)
	config.task_file = path
	config.task_id = path.stem
	config.imports = tuple(imports)
	return config


def load_config(path: Path) -> BaseTaskConfig:
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
	data, imports = _load_raw_config(path)
	import_files(imports)
	data.pop('run', None)
	return _build_config(data, path, imports)


@icontract.ensure(lambda result: len(result) > 0)
def load_run_configs(path: Path) -> list[BaseTaskConfig]:
	"""从 TOML 文件加载全部 run 的任务配置

	run 数组的每个条目是一次参数重载：先按条目的 _remove 从基础配置中删除键（回到配置模型默认值），
	再深度合并静态重载（语义同 extends），最后用 grid 候选值整体替换对应字段。
	grid 按 TOML 插入顺序取笛卡尔积，最右侧参数轴变化最快；所有组合保留条目的 id 与 default。
	条目的 default（默认 true）表示未指定 --runs 时是否运行。
	run 随 extends 继承，子文件声明 run 时整体替换父级的 run。基础配置（不含任何条目）须能单独通过校验。

	Args:
		path: TOML 文件路径

	Returns:
		按 run 数组顺序排列的配置列表（run_id、run_overrides、run_default 已设置）；无 run 时为只含基础配置的列表
	"""
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
		for candidates in product(*(values for _, values in run.grid)):
			combination = copy.deepcopy(merged)
			for (key, _), value in zip(run.grid, candidates, strict=True):
				node = combination
				for part in key[:-1]:
					if not isinstance(node.get(part), dict):
						node[part] = {}
					node = node[part]
				node[key[-1]] = copy.deepcopy(value)
			config = _build_config(combination, path, imports)
			config.run_id = run.run_id
			config.run_overrides = copy.deepcopy(run.source)
			config.run_default = run.default
			configs.append(config)
	return configs


@icontract.require(lambda configs, run_ids: set(run_ids) - {DEFAULT_RUNS} <= {c.run_id for c in configs}, 'Unknown run id(s)')
@icontract.require(
	lambda configs, run_ids: not set(run_ids) - {DEFAULT_RUNS} or any(c.run_id for c in configs),
	"Run ids were given but the task file declares no 'run' array",
)
def select_runs(configs: Sequence[BaseTaskConfig], run_ids: Sequence[str]) -> list[BaseTaskConfig]:
	"""按 run id 选择 run（保持 TOML 中的顺序）

	run_ids 为空等价于 ['default']；其中的 'default' 表示所有默认运行（default = true）的 run，可与其他 id 组合。
	"""
	selected = set(run_ids or [DEFAULT_RUNS])
	explicit = selected - {DEFAULT_RUNS}
	chosen = [c for c in configs if c.run_id in explicit or (DEFAULT_RUNS in selected and c.run_default)]
	if not chosen:
		raise ValueError('No run selected: the task file has no default run')
	return chosen


def run_task(
	config: BaseTaskConfig | Path,
	device: torch.device | str | None = None,
) -> Path:
	"""运行任务

	Args:
		config: BaseTaskConfig 对象或 TOML 文件路径
		device: 设备，默认自动选择 CUDA/CPU

	Returns:
		输出路径
	"""
	if isinstance(config, Path):
		config = load_config(config)

	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	elif isinstance(device, str):
		device = torch.device(device)

	_, run_fn = get_task(config.task_type)
	return run_fn(config, device)


@icontract.require(lambda configs: len(configs) > 0, 'run_tasks requires at least one config')
@icontract.ensure(lambda configs, result: len(result) == len(configs))
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
	# 第一阶段：每个输出路径只渲染一次，派发任务前检查冲突。
	seen: dict[Path, int] = {}
	for i, config in enumerate(configs):
		path = format_output_path(cast(Any, config).output.save_path, config).resolve()
		if path in seen:
			previous = seen[path]
			raise ValueError(
				f'Multiple runs resolve to the same output path: {path} '
				f'(config {previous + 1}, run {configs[previous].run_id!r}; config {i + 1}, run {configs[i].run_id!r})'
			)
		seen[path] = i

	# 第二阶段：按配置顺序运行任务。
	outputs: list[Path] = []
	for i, config in enumerate(configs):
		if len(configs) > 1:
			print(f'\n##### Run {i + 1}/{len(configs)}: {config.run_id or config.task_id} #####')
		outputs.append(run_task(config, device))
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	return outputs
