import daisy
import torch
from daisy.dataset import IndexDataset
from torch.optim import AdamW
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from timm.data.loader import MultiEpochsDataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math


def fast_train_smile(
	device: torch.device,
	model: torch.nn.Module,
	epochs: int,
	lr: float,
	dataset: IndexDataset,
	batch_size: int = 128,
	val_ratio: float = 0.1,
	warmup_epochs: int = 0,
	num_workers: int | tuple[int, int] = 10,
	use_amp: bool = True,
	pin_memory: bool = True,
):
	train_transform = daisy.util.transform.get_rectangle_train_transform()
	val_transform = daisy.util.transform.get_rectangle_val_transform()

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

	optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
	criterion = LabelSmoothingCrossEntropy()

	def lr_func(epoch: int):
		return min((epoch + 1) / (warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / epochs * math.pi) + 1))

	lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

	print('ready to train...')

	scaler = torch.GradScaler(enabled=use_amp)
	for epoch in range(epochs):
		model.train()
		losses = 0.0
		for images, label in train_loader:
			images, label = images.to(device, non_blocking=True), label.to(device, non_blocking=True)

			optimizer.zero_grad()

			if use_amp:
				with torch.autocast('cuda'):
					outputs = model(images)
					loss = criterion(outputs, label)

			else:
				outputs = model(images)
				loss = criterion(outputs, label)

			scaler.scale(loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			losses += loss.item()

		lr_scheduler.step()
		losses /= len(train_loader)
		model.eval()
		y_pred = []
		y_true = []
		count = [[0] * 3 for _ in range(3)]
		with torch.no_grad():
			for images, label in val_loader:
				images, label = images.to(device, non_blocking=True), label.to(device, non_blocking=True)

				if use_amp:
					with torch.autocast('cuda'):
						outputs = model(images)
				else:
					outputs = model(images)
				preds = torch.argmax(outputs, dim=1)

				y_pred.extend(preds.cpu().numpy())
				y_true.extend(label.cpu().numpy())
				for i, j in zip(label, preds):
					count[i][j] += 1

		acc = accuracy_score(y_true, y_pred)
		prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
		recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
		f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

		print(count)
		print(f'Epoch {epoch + 1}/{epochs}, Loss:{losses:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
