"""可解释性任务执行器"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from daisy.analysis.mae.interpretability import generate_vit_grad_rollout_heatmap, save_heatmap_visualization
from daisy.model.mae import create_vit_model
from ...base import TaskRunner
from ...data import select_dataset_split
from ...registry import TaskRegistry
from ...runtime import prepare_task_run, print_task_completed, save_run_snapshot
from ...shared import get_mae_finetune_val_transform
from .config import InterpretabilityConfig


@TaskRegistry.register
class InterpretabilityRunner(TaskRunner['InterpretabilityConfig']):
	"""可解释性任务执行器"""

	@classmethod
	def get_task_type(cls) -> str:
		return 'interpretability'

	@classmethod
	def get_config_class(cls) -> type[InterpretabilityConfig]:
		return InterpretabilityConfig

	@classmethod
	def get_ui_display_name(cls) -> str:
		return 'ViT 可解释性'

	def run(self, config: InterpretabilityConfig, device: torch.device) -> Path:
		runtime_cfg = config.runtime
		run_context = prepare_task_run(config, device, seed=runtime_cfg.seed)
		output_path = run_context.output_path

		# 加载数据
		selection = select_dataset_split(config.dataset, split_name=runtime_cfg.split_name)
		dataset = selection.to_dataset()
		save_run_snapshot(output_path, config, run_context)

		files, labels = dataset.getRawData()
		print(f'Total samples: {len(dataset)}')

		# 限制样本数
		if runtime_cfg.max_samples is not None:
			files = files[: runtime_cfg.max_samples]
			labels = labels[: runtime_cfg.max_samples]
			print(f'Limited to {len(files)} samples')

		# 加载模型
		model_cfg = config.model
		if model_cfg.family != 'mae_finetune':
			raise ValueError('Only mae_finetune models are supported for interpretability')

		model = create_vit_model(
			model_cfg.name,
			num_classes=model_cfg.num_classes,
			global_pool=model_cfg.global_pool,
			drop_path_rate=model_cfg.drop_path,
			img_size=model_cfg.img_size,
		)

		checkpoint = torch.load(model_cfg.checkpoint, map_location='cpu', weights_only=False)
		if isinstance(checkpoint, dict) and 'model' in checkpoint:
			checkpoint = checkpoint['model']
		model.load_state_dict(checkpoint)
		model.to(device)
		model.eval()

		print(f'Model: {model_cfg.name}')
		print(f'Checkpoint: {model_cfg.checkpoint}')

		# 准备 transform
		transform = get_mae_finetune_val_transform(input_size=runtime_cfg.input_size)

		# 生成可视化
		vis_dir = output_path / 'visualizations'
		vis_dir.mkdir(exist_ok=True)

		for idx, (file_path, true_label) in enumerate(zip(files, labels)):
			print(f'Processing {idx + 1}/{len(files)}: {file_path.name}')

			# 加载并预处理图像
			from PIL import Image

			img_pil = Image.open(file_path).convert('RGB')
			img_tensor = transform(img_pil)

			# 预测
			with torch.no_grad():
				output = model(img_tensor.unsqueeze(0).to(device))
				pred_label = output.argmax(dim=1).item()

			# 确定目标类别
			if runtime_cfg.target_class_mode == 'pred':
				target_class = pred_label
			else:
				target_class = true_label

			# 生成热力图
			heatmap, overlay = generate_vit_grad_rollout_heatmap(model, img_tensor, target_class=target_class, input_size=runtime_cfg.input_size)

			# 准备原图
			img_np = np.array(img_pil.resize((runtime_cfg.input_size, runtime_cfg.input_size))) / 255.0

			# 保存
			save_path = vis_dir / f'{file_path.stem}.png'
			save_heatmap_visualization(heatmap, overlay, img_np, save_path, true_label, pred_label)

		print(f'Saved {len(files)} visualizations to {vis_dir}')
		print_task_completed(output_path)
		return output_path
