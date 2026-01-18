"""Gradio Web UI"""

import gradio as gr

from datetime import datetime
from pathlib import Path


def launch_ui(port: int = 7860):
	"""启动 Gradio Web UI"""

	from .base import BaseMetaConfig, BaseOutputConfig
	from .registry import TaskRegistry
	from .runner import load_config, run_task
	from .tasks.classification.config import (
		ClassificationConfig,
		DatasetConfig,
		DatasetSplitConfig,
		ModelConfig,
		TrainingConfig,
		TransformConfig,
	)
	import tomli_w

	def get_task_types():
		"""获取所有已注册的任务类型"""
		return TaskRegistry.list_task_types() or ['classification']

	def create_classification_task(
		title,
		description,
		creator,
		dataset_type,
		root,
		sheet,
		column,
		label_offset,
		val_ratio,
		model_name,
		num_classes,
		pretrained,
		epochs,
		batch_size,
		lr,
		warmup_epochs,
		weight_decay,
		train_transform,
		val_transform,
		cmp_obj,
	):
		"""创建并保存分类任务配置"""
		# 生成任务 ID
		today = datetime.now().strftime('%Y%m%d')
		tasks_dir = Path('tasks')
		tasks_dir.mkdir(parents=True, exist_ok=True)

		existing = list(tasks_dir.glob(f'{today}-*.toml'))
		task_num = len(existing) + 1
		task_id = f'{today}-{task_num}'
		if title:
			task_id += f'-{title.replace(" ", "_")[:20]}'

		config = ClassificationConfig(
			task_type='classification',
			meta=BaseMetaConfig(
				title=title,
				description=description,
				creator=creator,
				created_at=datetime.now().strftime('%Y-%m-%d'),
			),
			dataset=DatasetConfig(
				type=dataset_type,
				root=root,
				sheet=sheet if dataset_type == 'sheet' else None,
				column=int(column),
				label_offset=int(label_offset),
				split=DatasetSplitConfig(val_ratio=float(val_ratio)),
			),
			model=ModelConfig(
				name=model_name,
				num_classes=int(num_classes),
				pretrained=pretrained,
			),
			training=TrainingConfig(
				epochs=int(epochs),
				batch_size=int(batch_size),
				lr=float(lr),
				warmup_epochs=int(warmup_epochs),
				weight_decay=float(weight_decay),
				cmp_obj=cmp_obj,
				transform=TransformConfig(train=train_transform, val=val_transform),
			),
			output=BaseOutputConfig(save_path=f'outputs/{task_id}'),
		)

		output_file = tasks_dir / f'{task_id}.toml'
		data = config.model_dump(exclude={'task_file', 'task_id'}, exclude_none=True)

		with open(output_file, 'wb') as f:
			tomli_w.dump(data, f)

		return f'任务配置已保存: {output_file}\n\n运行命令:\npython -m daisy run {output_file}'

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

	# 构建 UI
	with gr.Blocks(title='Daisy Task Manager') as demo:
		gr.Markdown('# Daisy Task Manager')

		with gr.Tab('创建分类任务'):
			with gr.Row():
				with gr.Column():
					gr.Markdown('### 基本信息')
					title = gr.Textbox(label='任务标题')
					description = gr.Textbox(label='描述', lines=2)
					creator = gr.Textbox(label='创建人')

					gr.Markdown('### 数据集配置')
					dataset_type = gr.Dropdown(choices=['sheet', 'folder'], value='sheet', label='数据集类型')
					root = gr.Textbox(label='数据根目录')
					sheet = gr.Textbox(label='标注文件路径 (sheet 类型)')
					column = gr.Number(value=1, label='标签列')
					label_offset = gr.Number(value=0, label='标签偏移')
					val_ratio = gr.Slider(0.05, 0.3, value=0.1, label='验证集比例')

				with gr.Column():
					gr.Markdown('### 模型配置')
					model_name = gr.Dropdown(
						choices=['resnet34', 'resnet50', 'resnet101', 'vgg16_bn', 'efficientnet_b0', 'convnext_tiny'],
						value='resnet34',
						label='模型',
						allow_custom_value=True,
					)
					num_classes = gr.Number(value=3, label='类别数')
					pretrained = gr.Checkbox(value=True, label='使用预训练权重')

					gr.Markdown('### 训练配置')
					epochs = gr.Number(value=30, label='训练轮数')
					batch_size = gr.Number(value=64, label='Batch Size')
					lr = gr.Number(value=1e-3, label='学习率')
					warmup_epochs = gr.Number(value=5, label='Warmup 轮数')
					weight_decay = gr.Number(value=0.05, label='Weight Decay')
					train_transform = gr.Dropdown(
						choices=['rectangle_train', 'rectangle_train_slight', 'stretch_train'], value='rectangle_train', label='训练 Transform'
					)
					val_transform = gr.Dropdown(choices=['rectangle_val', 'stretch_val'], value='rectangle_val', label='验证 Transform')
					cmp_obj = gr.Dropdown(choices=['f1', 'acc', 'prec', 'recall'], value='f1', label='优化目标')

			create_btn = gr.Button('创建任务', variant='primary')
			create_output = gr.Textbox(label='结果', lines=3)

			create_btn.click(
				create_classification_task,
				inputs=[
					title,
					description,
					creator,
					dataset_type,
					root,
					sheet,
					column,
					label_offset,
					val_ratio,
					model_name,
					num_classes,
					pretrained,
					epochs,
					batch_size,
					lr,
					warmup_epochs,
					weight_decay,
					train_transform,
					val_transform,
					cmp_obj,
				],
				outputs=create_output,
			)

		with gr.Tab('任务列表'):
			refresh_btn = gr.Button('刷新')
			tasks_display = gr.Textbox(label='任务列表', lines=15, value=list_tasks())
			refresh_btn.click(list_tasks, outputs=tasks_display)

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
