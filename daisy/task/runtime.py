"""Task 运行期辅助函数"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, cast

from .base import BaseTaskConfig


@dataclass(slots=True, frozen=True)
class TaskHeaderLine:
	"""任务头部附加信息"""

	label: str
	value: str


@dataclass(slots=True)
class TaskRunContext:
	"""任务运行上下文"""

	output_path: Path
	device: str
	commit: str
	seed: int | None = None


def format_output_path(save_path_template: str, config: BaseTaskConfig) -> Path:
	"""按配置字段渲染输出路径模板（不创建目录）"""
	values: dict[str, Any] = dict(config)
	values['date'] = datetime.now().strftime('%Y%m%d')
	return Path(save_path_template.format(**values))


def resolve_output_path(save_path_template: str, config: BaseTaskConfig) -> Path:
	"""解析并创建输出目录"""
	output_path = format_output_path(save_path_template, config)
	output_path.mkdir(parents=True, exist_ok=True)
	return output_path


def prepare_task_run(
	config: BaseTaskConfig,
	device: Any,
	*,
	seed: int | None = None,
	header_lines: Sequence[TaskHeaderLine] = (),
) -> TaskRunContext:
	"""准备任务运行上下文并打印通用头部信息"""
	import daisy

	if seed is not None:
		daisy.util.set_global_seed(seed)

	print('=' * 60)
	print(f'Task: {config.meta.title or config.task_id}')
	if config.run_id:
		print(f'Run: {config.run_id}')
	print(f'Description: {config.meta.description}')
	for line in header_lines:
		print(f'{line.label}: {line.value}')
	print(f'Device: {device}')
	if seed is not None:
		print(f'Seed: {seed}')
	print('=' * 60)

	if config.meta.commit == 'auto':
		config.meta.commit = daisy.util.get_git_commit()
	print(f'Git commit: {config.meta.commit}')

	output_path = resolve_output_path(cast(Any, config).output.save_path, config)
	print(f'Output path: {output_path}')
	return TaskRunContext(
		output_path=output_path,
		device=str(device),
		commit=config.meta.commit,
		seed=seed,
	)


def save_run_snapshot(
	output_path: Path,
	config: BaseTaskConfig,
	run_context: TaskRunContext,
	*,
	extra: dict[str, Any] | None = None,
) -> None:
	"""保存统一的运行快照信息"""
	runtime: dict[str, Any] = {
		'device': run_context.device,
		'commit': run_context.commit,
	}
	if run_context.seed is not None:
		runtime['seed'] = run_context.seed
	if config.imports:
		runtime['imports'] = save_import_files(output_path, config.imports)
	if extra:
		runtime.update(extra)
	save_task_snapshot(output_path, config, extra=runtime)


def save_import_files(output_path: Path, imports: Sequence[Path], *, dirname: str = 'imports') -> dict[str, str]:
	"""复制外部 imports 文件到输出目录，返回源路径到 sha256 的映射"""
	imports_dir = output_path / dirname
	imports_dir.mkdir(parents=True, exist_ok=True)
	digests: dict[str, str] = {}
	for path in imports:
		shutil.copy2(path, imports_dir / path.name)
		digests[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
	return digests


def print_task_completed(output_path: Path) -> None:
	"""打印统一的任务完成信息"""
	print('\n' + '=' * 60)
	print('Task completed!')
	print(f'Output saved to: {output_path}')
	print('=' * 60)


def save_json(path: Path, data: Any) -> None:
	"""保存 JSON 文件"""
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent=2)


def save_task_snapshot(
	output_path: Path,
	config: BaseTaskConfig,
	*,
	extra: dict[str, Any] | None = None,
	filename: str = 'task_snapshot.json',
) -> None:
	"""保存 task 配置快照"""
	# run_default 只影响选择哪些 run，条目原文（含 default）已记录在 run_overrides 中
	snapshot = config.model_dump(mode='json', exclude={'task_file', 'run_default'}, exclude_none=True)
	snapshot['task_id'] = config.task_id
	if not config.run_id:
		snapshot.pop('run_id')
	if config.task_file is not None:
		snapshot['task_file'] = str(config.task_file)
	if extra:
		snapshot['runtime'] = extra
	save_json(output_path / filename, snapshot)
