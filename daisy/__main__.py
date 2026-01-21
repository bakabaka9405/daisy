"""Daisy CLI 入口

用法:
	python -m daisy run <task.toml>     # 运行任务
	python -m daisy list                # 列出所有任务
	python -m daisy ui                  # 启动 Web UI
"""

import argparse
import sys
from pathlib import Path


def cmd_run(args):
	"""运行任务"""
	from daisy.task import run_task

	task_file = Path(args.task)
	if not task_file.exists():
		print(f'Error: Task file not found: {task_file}')
		sys.exit(1)

	run_task(task_file, device=args.device)

def cmd_list(args):
	"""列出所有任务"""
	from daisy.task import load_config

	tasks_dir = Path(args.tasks_dir)
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
			print(f'{cfg.task_id:<30} {cfg.task_type:<15} {cfg.meta.title[:23]:<25} {cfg.meta.created_at:<12}')
		except Exception as e:
			print(f'{f.stem:<30} [Error: {e}]')


def cmd_ui(args):
	"""启动 Web UI"""
	try:
		from daisy.task.ui import launch_ui

		launch_ui(args.port)
	except ImportError:
		print('UI 模块需要安装 gradio: pip install gradio')
		sys.exit(1)


def main():
	parser = argparse.ArgumentParser(
		description='Daisy - 深度学习训练任务管理工具',
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	subparsers = parser.add_subparsers(dest='command', help='可用命令')

	# run 命令
	run_parser = subparsers.add_parser('run', help='运行任务')
	run_parser.add_argument('task', help='任务配置文件 (.toml)')
	run_parser.add_argument('--device', '-d', help='运行设备 (cuda/cpu)')

	# list 命令
	list_parser = subparsers.add_parser('list', help='列出所有任务')
	list_parser.add_argument('--tasks-dir', '-d', default='tasks', help='任务目录')

	# ui 命令
	ui_parser = subparsers.add_parser('ui', help='启动 Web UI')
	ui_parser.add_argument('--port', '-p', type=int, default=7860, help='端口号')

	args = parser.parse_args()

	if args.command == 'run':
		cmd_run(args)
	elif args.command == 'list':
		cmd_list(args)
	elif args.command == 'ui':
		cmd_ui(args)
	else:
		parser.print_help()


if __name__ == '__main__':
	main()
