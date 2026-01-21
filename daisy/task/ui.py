"""Gradio Web UI - 动态生成任务类型的 UI"""

import gradio as gr

from datetime import datetime
from pathlib import Path
from typing import Any

import tomli_w

from .registry import TaskRegistry
from .runner import load_config, run_task
from .ui_builder import build_task_ui


def launch_ui(port: int = 7860):
	"""启动 Gradio Web UI"""

	def list_tasks():
		"""列出所有任务"""
		tasks_dir = Path('tasks')
		if not tasks_dir.exists():
			return '没有找到任务'

		task_files = sorted(tasks_dir.glob('*.toml'), reverse=True)
		if not task_files:
			return '没有找到任务'

		lines = [f'{"Task ID":<30} {"Type":<15} {"Title":<20} {"Created"}']
		lines.append('-' * 80)

		for f in task_files[:20]:  # 只显示最近 20 个
			try:
				cfg = load_config(f)
				lines.append(
					f'{cfg.task_id:<30} {cfg.task_type:<15} '
					f'{cfg.meta.title[:18]:<20} {cfg.meta.created_at}'
				)
			except Exception as e:
				lines.append(f'{f.stem:<30} [Error: {str(e)[:30]}]')

		return '\n'.join(lines)

	def run_selected_task(task_file):
		"""运行选中的任务"""
		if not task_file:
			return '请选择任务文件'

		try:
			output_path = run_task(task_file)
			return f'任务完成\n输出目录: {output_path}'
		except Exception as e:
			return f'任务失败: {e}'

	def get_task_files():
		"""获取任务文件列表"""
		tasks_dir = Path('tasks')
		if not tasks_dir.exists():
			return []
		return [str(f) for f in sorted(tasks_dir.glob('*.toml'), reverse=True)]

	def create_task_submit_handler(
		task_type: str,
		config_dict: dict[str, Any],
		config_class: type,
	) -> str:
		"""通用的任务创建提交处理器"""
		try:
			# 生成任务 ID
			today = datetime.now().strftime('%Y%m%d')
			tasks_dir = Path('tasks')
			tasks_dir.mkdir(parents=True, exist_ok=True)

			existing = list(tasks_dir.glob(f'{today}-*.toml'))
			task_num = len(existing) + 1

			# 使用标题生成任务 ID
			title = config_dict.get('meta', {}).get('title', '')
			task_id = f'{today}-{task_num}'
			if title:
				task_id += f'-{title.replace(" ", "_")[:20]}'

			# 设置 created_at
			if 'meta' not in config_dict:
				config_dict['meta'] = {}
			config_dict['meta']['created_at'] = datetime.now().strftime('%Y-%m-%d')

			# 设置默认输出路径
			if 'output' not in config_dict:
				config_dict['output'] = {}
			if 'save_path' not in config_dict['output']:
				config_dict['output']['save_path'] = f'outputs/{task_id}'

			# 清理 None 值
			config_dict = clean_none_values(config_dict)

			# 验证配置
			config = config_class(**config_dict)

			# 保存配置
			output_file = tasks_dir / f'{task_id}.toml'
			data = config.model_dump(exclude={'task_file', 'task_id'}, exclude_none=True)

			with open(output_file, 'wb') as f:
				tomli_w.dump(data, f)

			return f'任务配置已保存: {output_file}\n\n运行命令:\npython -m daisy run {output_file}'
		except Exception as e:
			import traceback
			return f'创建任务失败: {e}\n\n{traceback.format_exc()}'

	def clean_none_values(d: dict) -> dict:
		"""递归清理字典中的 None 值"""
		result = {}
		for k, v in d.items():
			if v is None:
				continue
			if isinstance(v, dict):
				cleaned = clean_none_values(v)
				if cleaned:  # 不保留空字典
					result[k] = cleaned
			else:
				result[k] = v
		return result

	# 获取所有任务类型
	task_types = TaskRegistry.list_task_types()
	if not task_types:
		task_types = ['classification']

	# 构建 UI
	with gr.Blocks(title='Daisy Task Manager') as demo:
		gr.Markdown('# Daisy Task Manager')

		# 为每个任务类型创建 Tab
		for task_type in task_types:
			try:
				runner_cls = TaskRegistry.get_runner(task_type)
				config_class = runner_cls.get_config_class()
				display_name = runner_cls.get_ui_display_name()

				with gr.Tab(f'创建 {display_name} 任务'):
					build_task_ui(
						gr,
						task_type,
						runner_cls,
						config_class,
						create_task_submit_handler,
					)
			except Exception as e:
				with gr.Tab(f'创建 {task_type} 任务'):
					gr.Markdown(f'### 错误\n\n无法加载任务类型 `{task_type}`: {e}')

		# 任务列表 Tab
		with gr.Tab('任务列表'):
			refresh_btn = gr.Button('刷新')
			tasks_display = gr.Textbox(label='任务列表', lines=15, value=list_tasks())
			refresh_btn.click(list_tasks, outputs=tasks_display)

		# 运行任务 Tab
		with gr.Tab('运行任务'):
			task_dropdown = gr.Dropdown(
				choices=get_task_files(),
				label='选择任务',
				allow_custom_value=True,
			)
			refresh_tasks_btn = gr.Button('刷新任务列表')
			run_btn = gr.Button('运行任务', variant='primary')
			run_output = gr.Textbox(label='运行结果', lines=5)

			refresh_tasks_btn.click(
				lambda: gr.update(choices=get_task_files()),
				outputs=task_dropdown,
			)
			run_btn.click(run_selected_task, inputs=task_dropdown, outputs=run_output)

	demo.launch(server_port=port)
