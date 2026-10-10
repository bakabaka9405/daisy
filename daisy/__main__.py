"""Daisy CLI 入口

用法:
	python -m daisy run <task.toml> [-d device] [--cpu-affinity mask] [--runs id ...] [-- options]  # 运行任务并支持参数覆盖
	python -m daisy list [-d tasks_dir]                       # 列出所有任务
	python -m daisy example <task_type> [output] [-f] [-i a.py]  # 导出示例配置
"""

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Annotated

import tyro
import tyro.conf
import psutil

from daisy.task import export_example_config, import_files, load_config, load_run_configs, run_tasks, select_runs


@dataclass(slots=True, frozen=True)
class Run:
	"""运行任务

	Args:
		task: 任务配置文件路径
		overrides: 通过 -- 分隔符传入的任务配置覆盖参数，作用于每个 run（优先级高于 run 条目）
		device: 运行设备 (例如: cuda, cpu)
		cpu_affinity: CPU 亲和度的正整数位掩码，第 i 位对应 CPU i；省略时使用 DAISY_CPU_AFFINITY，两者均未设置时保留当前亲和度
		runs: 只运行这些 id 的 run（按任务文件中的顺序）；default 表示所有默认运行的 run，可与其他 id 组合；不指定时等价于 default
	"""

	task: Annotated[Path, tyro.conf.Positional]
	overrides: Annotated[tuple[str, ...], tyro.conf.Positional] = ()
	device: Annotated[str | None, tyro.conf.arg(aliases=('-d',))] = None
	cpu_affinity: int | None = None
	runs: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class ListTasks:
	"""列出所有任务"""

	tasks_dir: Annotated[Path, tyro.conf.arg(aliases=('-d',))] = Path('tasks')


@dataclass(slots=True, frozen=True)
class Example:
	"""导出任务示例配置

	Args:
		task_type: 任务类型
		output: 输出路径
		force: 覆盖已存在的文件
		imports: 导出前导入的外部 Python 文件（用于注册外部任务）
	"""

	task_type: Annotated[str, tyro.conf.Positional]
	output: Annotated[Path | None, tyro.conf.Positional] = None
	force: Annotated[bool, tyro.conf.arg(aliases=('-f',))] = False
	imports: Annotated[tuple[Path, ...], tyro.conf.arg(aliases=('-i',))] = ()


type Command = (
	Annotated[Run, tyro.conf.subcommand('run')]
	| Annotated[ListTasks, tyro.conf.subcommand('list')]
	| Annotated[Example, tyro.conf.subcommand('example')]
)


def cmd_list(cmd: ListTasks) -> None:
	"""列出所有任务"""
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
	import_files(cmd.imports)
	output_path = export_example_config(cmd.task_type, cmd.output, force=cmd.force)
	print(f'已导出示例配置: {output_path}')


def main(argv: Sequence[str] | None = None) -> None:
	args = list(sys.argv[1:] if argv is None else argv)
	command = tyro.cli(Command, args=args or ['--help'], prog='daisy')  # type: ignore[arg-type]

	match command:
		case Run():
			mask = command.cpu_affinity
			if mask is None and (env_mask := os.environ.get('DAISY_CPU_AFFINITY')) is not None:
				mask = int(env_mask)
			if mask is not None:
				if mask <= 0:
					raise ValueError('CPU affinity mask must be positive.')
				psutil.Process().cpu_affinity([i for i in range(mask.bit_length()) if mask & (1 << i)])

			loaded = select_runs(load_run_configs(command.task), command.runs)
			configs = [tyro.cli(type(c), default=c, args=command.overrides, prog='daisy run TASK --') for c in loaded]
			run_tasks(configs, device=command.device)
		case ListTasks():
			cmd_list(command)
		case Example():
			cmd_example(command)


if __name__ == '__main__':
	main()
