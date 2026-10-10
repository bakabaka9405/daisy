"""任务执行引擎"""

from __future__ import annotations

from collections.abc import Sequence
import copy
import gc
from itertools import product
from pathlib import Path
import sys
import tomllib
from typing import Annotated, Any, cast

from deepmerge import Merger
import icontract
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter
import torch

from .base import BaseTaskConfig
from .imports import import_files
from .provenance import SourceSnapshot, changed_repositories, modules_outside_snapshot, snapshot_sources
from .runtime import format_output_path, save_json
from .tasks import get_task

_merger = Merger([(dict, ['merge'])], ['override'], ['override'])
# --runs 中表示“所有默认运行的 run”的保留字，不能用作 run id
DEFAULT_RUNS = 'default'

_NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
# 点分隔的键路径，各段非空（如 'model.checkpoint_path'）
_DottedKey = Annotated[str, StringConstraints(pattern=r'^[^.]+(\.[^.]+)*$')]


class _TaskFileHeader(BaseModel):
	"""task 文件顶层的保留键（不进入配置）"""

	model_config = ConfigDict(extra='forbid', strict=True, frozen=True)

	imports: list[_NonEmptyStr] = []  # 外部 Python 文件，相对当前 TOML 文件所在目录
	extends: _NonEmptyStr | None = None  # 父级 TOML 文件，相对当前 TOML 文件所在目录
	remove: list[_DottedKey] = Field(default=[], alias='_remove')  # 合并当前文件前从父级配置中删除的键


_HEADER_KEYS = ('imports', 'extends', '_remove')


class _RunEntry(BaseModel):
	"""run 数组的单个条目；id、default、_remove 以外的键为深度合并到基础配置上的重载内容（含条目的 grid）"""

	model_config = ConfigDict(extra='allow', strict=True, frozen=True)

	run_id: Annotated[str, StringConstraints(pattern=r'^[A-Za-z0-9._-]+$')] = Field(alias='id')
	default: bool = True  # 未指定 --runs（或 --runs 含 default）时是否运行
	remove: list[_DottedKey] = Field(default=[], alias='_remove')  # 合并前从基础配置中删除的键

	@property
	def overrides(self) -> dict[str, Any]:
		return cast(dict[str, Any], self.model_extra)

	@property
	def source(self) -> dict[str, Any]:
		"""条目去掉 id 后的原文，记录到快照"""
		return self.model_dump(by_alias=True, exclude={'run_id'}, exclude_unset=True)


_RUNS = TypeAdapter(Annotated[list[_RunEntry], Field(min_length=1)])


def _remove_keys(data: dict[str, Any], keys: Sequence[str], where: str) -> None:
	"""按顺序删除点分隔键路径，缺失时立即报告键路径及出处"""
	for key in keys:
		*parents, leaf = key.split('.')
		node: Any = data
		for part in parents:
			node = node.get(part) if isinstance(node, dict) else None
		if not isinstance(node, dict) or leaf not in node:
			raise ValueError(f"Cannot remove key '{key}' in {where}: key is not set by the inherited configuration")
		del node[leaf]


def _parse_grid(raw: dict[str, Any], where: str) -> list[list[dict[tuple[str, ...], Any]]]:
	"""按 TOML 插入顺序收集参数轴，每个候选包含同时写入的字段及其值。"""

	def fields(node: dict[str, Any], prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
		result: dict[tuple[str, ...], Any] = {}
		for name, value in node.items():
			key = prefix + (name,)
			if isinstance(value, dict) and value:
				result.update(fields(value, key))
			else:
				result[key] = value
		return result

	def independent_axes(node: dict[str, Any]) -> list[list[dict[tuple[str, ...], Any]]]:
		result: list[list[dict[tuple[str, ...], Any]]] = []
		for key, values in fields(node).items():
			if not isinstance(values, list) or not values:
				raise ValueError(f"Grid axis '{'.'.join(key)}' in {where} must be a non-empty list of candidate values")
			result.append([{key: value} for value in values])
		return result

	# 第一阶段：普通字段各自构成一个轴，_zip 的多个字段共同构成一个轴。
	axes: list[list[dict[tuple[str, ...], Any]]] = []
	for name, value in raw.items():
		if name != '_zip':
			axes.extend(independent_axes({name: value}))
		elif isinstance(value, dict) and value:
			columns = independent_axes(value)
			axes.append([{key: item for column in row for key, item in column.items()} for row in zip(*columns, strict=True)])
		elif isinstance(value, list) and value:
			rows: list[dict[tuple[str, ...], Any]] = []
			for row in value:
				if not isinstance(row, dict) or not row:
					raise ValueError(f"Grid '_zip' candidates in {where} must be non-empty tables")
				rows.append(fields(row))
			axes.append(rows)
		else:
			raise ValueError(f"Grid '_zip' in {where} must be a non-empty table or list of tables")

	# 第二阶段：同一字段只能属于一个轴，避免不同轴互相覆盖。
	seen: set[tuple[str, ...]] = set()
	for axis in axes:
		keys = {key for candidate in axis for key in candidate}
		if duplicate := seen & keys:
			raise ValueError(f'Duplicate grid fields in {where}: {sorted(".".join(key) for key in duplicate)}')
		seen.update(keys)
	return axes


def _parse_runs(raw: Any, path: Path) -> list[_RunEntry]:
	"""解析 run 字段：schema 校验条目结构后，检查 id 不为保留字且互不重复

	Args:
		raw: 合并继承链后的 run 字段值，未声明时为 None
		path: 当前 TOML 文件路径，用于报错信息
	"""
	if raw is None:
		return []

	runs = _RUNS.validate_python(raw)
	ids = [run.run_id for run in runs]
	if DEFAULT_RUNS in ids:
		raise ValueError(f"Run id '{DEFAULT_RUNS}' in task file: {path} is reserved for selecting all default runs")
	if len(set(ids)) != len(ids):
		raise ValueError(f'Duplicate run id in task file: {path}: {ids}')
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

	# 第二阶段：读取当前文件数据，schema 校验并取出顶层保留键
	with open(path, 'rb') as f:
		data = tomllib.load(f)
	header = _TaskFileHeader.model_validate({key: data.pop(key) for key in _HEADER_KEYS if key in data})
	imports = [(path.parent / p).resolve() for p in header.imports]

	# 第三阶段：递归加载父级配置；_remove 先从父级配置中删除键，再合并当前文件
	if header.extends is not None:
		parent_path = (path.parent / header.extends).resolve()
		parent_data, parent_imports = _load_raw_config(parent_path, (*visited, resolved_path))
		_remove_keys(parent_data, header.remove, f'task file: {path}')
		merged_imports = list(dict.fromkeys([*parent_imports, *imports]))
		return _merger.merge(parent_data, data), merged_imports

	if header.remove:
		raise ValueError(f"'_remove' in task file: {path} requires 'extends' (it removes inherited keys)")
	return data, list(dict.fromkeys(imports))


def _build_config(data: dict[str, Any], path: Path, imports: list[Path]) -> BaseTaskConfig:
	"""按 task_type 选择配置类，校验原始字典并设置运行时元信息"""
	task_type = data.get('task_type')
	if not isinstance(task_type, str) or not task_type:
		raise ValueError(f"Missing or invalid 'task_type' in task file: {path}. Expected non-empty string.")
	config = get_task(task_type).config_type.model_validate(data)
	config.task_file = path
	config.task_id = path.stem
	config.imports = tuple(imports)
	return config


def _expand_grid(data: dict[str, Any], path: Path, imports: list[Path], where: str) -> list[BaseTaskConfig]:
	"""弹出 data 中的 grid，按笛卡尔积逐个组合整体替换对应字段并构建配置；无 grid 时只含一个配置

	where 仅用于报错信息中标明出处。
	"""
	axes = _parse_grid(data.pop('grid', {}), where)
	configs: list[BaseTaskConfig] = []
	for candidates in product(*axes):
		combination = copy.deepcopy(data)
		for candidate in candidates:
			for key, value in candidate.items():
				node = combination
				for part in key[:-1]:
					if not isinstance(node.get(part), dict):
						node[part] = {}
					node = node[part]
				node[key[-1]] = copy.deepcopy(value)
		configs.append(_build_config(combination, path, imports))
	return configs


def load_config(path: Path) -> BaseTaskConfig:
	"""从 TOML 文件加载任务配置

	根据 TOML 中的 task_type 字段自动选择正确的配置类。
	支持通过 top-level extends 字段继承父级 TOML 配置，
	通过 top-level imports 字段导入注册外部 task / feeder 的 Python 文件。
	run 数组与 grid（若有）被忽略，返回不含任何 run 重载与网格展开的基础配置；需要逐个 run 的配置时使用 load_run_configs。

	Args:
		path: TOML 文件路径

	Returns:
		对应任务类型的配置对象
	"""
	data, imports = _load_raw_config(path)
	import_files(imports)
	data.pop('run', None)
	data.pop('grid', None)
	return _build_config(data, path, imports)


@icontract.ensure(lambda result: len(result) > 0)
def load_run_configs(path: Path) -> list[BaseTaskConfig]:
	"""从 TOML 文件加载全部 run 的任务配置

	run 数组的每个条目是一次参数重载：先按条目的 _remove 从基础配置中删除键（回到配置模型默认值），
	再深度合并其余内容（语义同 extends），最后用 grid 候选值整体替换对应字段。
	grid 可写在顶层（随 extends 继承）与条目中，条目的 grid 与顶层 grid 深度合并，同名参数轴由条目覆盖；
	grid 按 TOML 插入顺序取笛卡尔积，最右侧参数轴变化最快；所有组合保留条目的 id 与 default。
	grid._zip 将候选表列表或表中的等长候选数组按位置配对，作为一个轴参与笛卡尔积。
	条目的 default（默认 true）表示未指定 --runs 时是否运行。
	run 随 extends 继承，子文件声明 run 时整体替换父级的 run。基础配置（不含任何条目与 grid）须能单独通过校验。

	Args:
		path: TOML 文件路径

	Returns:
		按 run 数组顺序、各 run 内按网格顺序排列的配置列表（run_id、run_overrides、run_default 已设置）；
		无 run 时为基础配置按顶层 grid 展开的列表
	"""
	data, imports = _load_raw_config(path)
	import_files(imports)
	runs = _parse_runs(data.pop('run', None), path)
	_build_config({k: v for k, v in data.items() if k != 'grid'}, path, imports)  # 校验基础配置
	if not runs:
		return _expand_grid(data, path, imports, f'task file: {path}')

	configs: list[BaseTaskConfig] = []
	for run in runs:
		where = f"run '{run.run_id}' of task file: {path}"
		merged = copy.deepcopy(data)
		_remove_keys(merged, run.remove, where)
		merged = _merger.merge(merged, copy.deepcopy(run.overrides))
		for config in _expand_grid(merged, path, imports, where):
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
	"""运行单个任务，流程同 run_tasks

	Args:
		config: BaseTaskConfig 对象或 TOML 文件路径
		device: 设备，默认自动选择 CUDA/CPU

	Returns:
		输出路径
	"""
	return run_tasks([load_config(config) if isinstance(config, Path) else config], device)[0]


def _source_paths(configs: Sequence[BaseTaskConfig]) -> list[Path]:
	"""决定运行代码的源文件位置：daisy 包、各 task 文件及其 imports"""
	paths = [Path(__file__).resolve().parents[1]]
	for config in configs:
		if config.task_file is not None:
			paths.append(config.task_file.resolve())
		paths.extend(config.imports)
	return paths


@icontract.require(lambda configs: len(configs) > 0, 'run_tasks requires at least one config')
@icontract.ensure(lambda configs, result: len(result) == len(configs))
def run_tasks(
	configs: Sequence[BaseTaskConfig],
	device: torch.device | str | None = None,
) -> list[Path]:
	"""在同一进程中顺序运行多个 run，任一 run 报错立即终止

	运行前检查各 run 渲染后的输出路径互不相同，避免结果互相覆盖；
	再把相关仓库的工作区记录为快照 commit，已加载模块的源文件须全部在快照内（或属于 Python 安装）。
	每个 run 结束后再次核对，在输出目录写入 provenance.json。

	Args:
		configs: 各 run 的最终配置
		device: 设备，默认自动选择 CUDA/CPU

	Returns:
		各 run 的输出路径
	"""
	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	elif isinstance(device, str):
		device = torch.device(device)

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

	# 第二阶段：记录源码快照；task 与 feeder 已在加载配置时导入，此时须全部来自快照
	task_ids = ', '.join(dict.fromkeys(c.task_id for c in configs))
	sources = snapshot_sources(_source_paths(configs), f'daisy run: {task_ids}')
	_check_modules(sources)
	print(f'Source snapshot: {sources.ref}')
	for repo in sources.repositories:
		print(f'  {repo.root}: {repo.commit[:8]} (HEAD {repo.head[:8]}{", dirty" if repo.dirty else ""})')

	# 第三阶段：按配置顺序运行任务，每个 run 结束后核对源码并写入溯源信息。
	outputs: list[Path] = []
	for i, config in enumerate(configs):
		if len(configs) > 1:
			print(f'\n##### Run {i + 1}/{len(configs)}: {config.run_id or config.task_id} #####')
		output_path = get_task(config.task_type).run(config, device)
		_check_modules(sources)
		changed = changed_repositories(sources)
		if changed:
			print(f'Warning: source changed during run in {", ".join(map(str, changed))}')
		save_json(output_path / 'provenance.json', {**sources.to_json(), 'changed_during_run': [str(p) for p in changed]})
		outputs.append(output_path)
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	return outputs


def _check_modules(sources: SourceSnapshot) -> None:
	"""已加载模块的源文件须在快照内或属于 Python 安装，否则运行无法复现"""
	outside = modules_outside_snapshot(sources, list(sys.modules.items()))
	if outside:
		raise RuntimeError('Loaded modules are not covered by the source snapshot:\n  ' + '\n  '.join(outside))
