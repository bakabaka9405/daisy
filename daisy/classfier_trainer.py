import daisy
import torch
from daisy.dataset import IndexDataset
from torch.optim import AdamW
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from timm.data.loader import MultiEpochsDataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math
from pathlib import Path
from typing import Literal


def fast_train_smile(
	device: torch.device,
	model: torch.nn.Module,
	num_classes: int,
	epochs: int,
	lr: float,
	dataset: IndexDataset | tuple[IndexDataset, IndexDataset],
	batch_size: int = 128,
	accum_iter: int = 1,
	train_transform: torch.nn.Module | None = None,
	val_transform: torch.nn.Module | None = None,
	val_ratio: float = 0.1,
	warmup_epochs: int = 0,
	num_workers: int | tuple[int, int] = 10,
	weight_decay: float = 1e-4,
	smoothing: float = 0.1,
	use_amp: bool = True,
	use_scheduler: bool = True,
	clip_grad: bool = False,
	max_norm: float = 1.0,
	pin_memory: bool = True,
	early_stop: bool = False,
	early_stop_epoch: int = 5,
	save_path: Path | None = None,
	keep_count: int = 0,
	cmp_obj: Literal['acc', 'prec', 'recall', 'f1'] = 'f1',
):
	if train_transform is None:
		train_transform = daisy.util.transform.get_rectangle_train_transform()
	if val_transform is None:
		val_transform = daisy.util.transform.get_rectangle_val_transform()

	if save_path is not None:
		save_path.mkdir(parents=True, exist_ok=True)

	if isinstance(dataset, tuple):
		train_dataset, val_dataset = dataset
	else:
		train_dataset, val_dataset = daisy.dataset.dataset_split.default_data_split(dataset, val_ratio=val_ratio)
	torch.cuda.empty_cache()

	train_dataset.setTransform(train_transform)
	val_dataset.applyTransform(val_transform)

	if isinstance(num_workers, int):
		num_workers = (num_workers, num_workers)

	print('loading dataloaders...')
	train_loader = MultiEpochsDataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers[0],
		pin_memory=pin_memory,
	)

	val_loader = MultiEpochsDataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers[1],
		pin_memory=pin_memory,
	)

	model.to(device)

	optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)

	def lr_func(epoch: int):
		return min(epoch / (warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / epochs * math.pi) + 1))

	lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

	print('ready to train...')

	scaler = torch.GradScaler(enabled=use_amp)

	best_epoch = 0
	best = {
		'acc': 0.0,
		'prec': 0.0,
		'recall': 0.0,
		'f1': 0.0,
	}
	for epoch in range(epochs):
		model.train()
		train_losses = 0.0
		y_pred = []
		y_true = []
		for i, (images, label) in enumerate(train_loader):
			images, label = images.to(device, non_blocking=True), label.to(device, non_blocking=True)

			if use_amp:
				with torch.autocast('cuda'):
					outputs = model(images)
					loss = criterion(outputs, label) / accum_iter
			else:
				outputs = model(images)
				loss = criterion(outputs, label) / accum_iter

			preds = torch.argmax(outputs, dim=1)
			y_pred.extend(preds.cpu().numpy())
			y_true.extend(label.cpu().numpy())
			scaler.scale(loss).backward()
			if (i + 1) % accum_iter == 0 or (i + 1) == len(train_loader):
				if clip_grad:
					scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad()
			train_losses += loss.item() * accum_iter

		if use_scheduler:
			lr_scheduler.step()
		train_losses /= len(train_loader)

		train_acc = accuracy_score(y_true, y_pred)

		y_pred = []
		y_true = []
		val_losses = 0.0
		count = [[0] * num_classes for _ in range(num_classes)]
		model.eval()
		with torch.no_grad():
			for images, label in val_loader:
				images, label = images.to(device, non_blocking=True), label.to(device, non_blocking=True)

				if use_amp:
					with torch.autocast('cuda'):
						outputs = model(images)
						loss = criterion(outputs, label)
				else:
					outputs = model(images)
					loss = criterion(outputs, label)

				val_losses += loss.item()
				preds = torch.argmax(outputs, dim=1)

				y_pred.extend(preds.cpu().numpy())
				y_true.extend(label.cpu().numpy())
				for i, j in zip(label, preds):
					count[i][j] += 1

		val_losses /= len(val_loader)
		val_acc = accuracy_score(y_true, y_pred)
		prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
		recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
		f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

		print(count)
		print(
			f'Epoch {epoch + 1}/{epochs}, LR: {optimizer.param_groups[0]["lr"]:.6f}, Train Loss: {train_losses:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_losses:.4f}, Val Accuracy: {val_acc:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}'
		)

		current = {
			'acc': val_acc,
			'prec': prec,
			'recall': recall,
			'f1': f1,
		}

		better = current[cmp_obj] > best[cmp_obj]

		if save_path is not None:
			torch.save(model.state_dict(), save_path / f'model_epoch_{epoch + 1}.pth')
			# print(f'Model saved at {save_path / f"model_epoch_{epoch + 1}.pth"}')
			if keep_count > 0 and epoch >= keep_count:
				remove_path = save_path / f'model_epoch_{epoch + 1 - keep_count}.pth'
				if remove_path.exists():
					remove_path.unlink()
					# print(f'Removed old model {remove_path}')

			if better:
				torch.save(model.state_dict(), save_path / 'best_model.pth')

		if better:
			best = current
			best_epoch = epoch

		if early_stop and epoch - best_epoch >= early_stop_epoch:
			print(f'Early stopping at epoch {epoch + 1}')
			break
