"""Daisy CLI 入口

用法:
	python -m daisy run <task.toml> [-d device] [-- options]  # 运行任务并支持参数覆盖
	python -m daisy list [-d tasks_dir]                       # 列出所有任务
	python -m daisy example <task_type> [output] [-f]         # 导出示例配置
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Annotated

import tyro
import tyro.conf


@dataclass(slots=True, frozen=True)
class Run:
	"""运行任务

	Args:
		task: 任务配置文件路径
		overrides: 通过 -- 分隔符传入的任务配置覆盖参数
		device: 运行设备 (例如: cuda, cpu)
	"""

	task: Annotated[Path, tyro.conf.Positional]
	overrides: Annotated[tuple[str, ...], tyro.conf.Positional] = ()
	device: Annotated[str | None, tyro.conf.arg(aliases=('-d',))] = None


@dataclass(slots=True, frozen=True)
class ListTasks:
	"""列出所有任务"""

	tasks_dir: Annotated[Path, tyro.conf.arg(aliases=('-d',))] = Path('tasks')


@dataclass(slots=True, frozen=True)
class Example:
	"""导出任务示例配置"""

	task_type: Annotated[str, tyro.conf.Positional]
	output: Annotated[Path | None, tyro.conf.Positional] = None
	force: Annotated[bool, tyro.conf.arg(aliases=('-f',))] = False


type Command = (
	Annotated[Run, tyro.conf.subcommand('run')]
	| Annotated[ListTasks, tyro.conf.subcommand('list')]
	| Annotated[Example, tyro.conf.subcommand('example')]
)


def cmd_list(cmd: ListTasks) -> None:
	"""列出所有任务"""
	from daisy.task import load_config

	tasks_dir = Path(cmd.tasks_dir)
	if not tasks_dir.exists():
		print(f'任务目录不存在: {tasks_dir}')
		return

	task_files = sorted(tasks_dir.glob('*.toml'), reverse=True)

	if not task_files:
		print('没有找到任务文件')
		return

	print(f'{"Task ID":<30} {"Type":<15} {"Title":<25} {"Created":<12}')
	print('-' * 82)

	for f in task_files:
		try:
			cfg = load_config(f)
		except (OSError, ValueError) as e:
			print(f'{f.stem:<30} [Error: {e}]')
			continue
		print(f'{cfg.task_id:<30} {cfg.task_type:<15} {cfg.meta.title[:23]:<25} {cfg.meta.created_at:<12}')


def cmd_example(cmd: Example) -> None:
	"""导出任务示例配置"""
	from daisy.task import export_example_config

	output_path = export_example_config(cmd.task_type, cmd.output, force=cmd.force)
	print(f'已导出示例配置: {output_path}')


def main(argv: Sequence[str] | None = None) -> None:
	args = list(sys.argv[1:] if argv is None else argv)
	command = tyro.cli(Command, args=args or ['--help'], prog='daisy')  # type: ignore[arg-type]

	match command:
		case Run():
			from daisy.task import load_config, run_task

			loaded = load_config(command.task)
			config = tyro.cli(type(loaded), default=loaded, args=command.overrides, prog='daisy run TASK --')
			run_task(config, device=command.device)
		case ListTasks():
			cmd_list(command)
		case Example():
			cmd_example(command)


if __name__ == '__main__':
	main()
