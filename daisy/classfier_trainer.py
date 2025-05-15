import daisy
import torch
from daisy.dataset import IndexDataset
from torch.optim import AdamW
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from timm.data.loader import MultiEpochsDataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def fast_train_smile(
	device: torch.device,
	model: torch.nn.Module,
	epochs: int,
	lr: float,
	dataset: IndexDataset,
	num_workers: int | tuple[int, int] = 10,
):
	train_transform = daisy.util.transform.get_rectangle_train_transform()
	val_transform = daisy.util.transform.get_rectangle_val_transform()

	train_dataset, val_dataset = daisy.dataset.dataset_split.default_data_split(dataset, val_ratio=0.1)
	torch.cuda.empty_cache()

	train_dataset.setTransform(train_transform)
	val_dataset.applyTransform(val_transform)

	if isinstance(num_workers, int):
		num_workers = (num_workers, num_workers)

	print('loading dataloaders...')
	train_loader = MultiEpochsDataLoader(
		train_dataset,
		batch_size=128,
		shuffle=True,
		num_workers=num_workers[0],
		pin_memory=True,
	)

	val_loader = MultiEpochsDataLoader(
		val_dataset,
		batch_size=128,
		shuffle=False,
		num_workers=num_workers[1],
		pin_memory=True,
	)

	model.to(device)

	optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
	loss_fn = LabelSmoothingCrossEntropy()

	print('ready to train...')

	scaler = torch.GradScaler()
	for epoch in range(epochs):
		model.train()
		losses = 0.0
		for images, label in train_loader:
			images, label = images.to(device, non_blocking=True), label.to(device, non_blocking=True)

			optimizer.zero_grad()

			with torch.autocast('cuda'):
				outputs = model(images)
				loss = loss_fn(outputs, label)
				losses += loss.item()
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		losses /= len(train_loader)
		model.eval()
		y_pred = []
		y_true = []
		count = [[0] * 3 for _ in range(3)]
		with torch.no_grad():
			for images, label in val_loader:
				images, label = images.to(device, non_blocking=True), label.to(device, non_blocking=True)

				with torch.autocast('cuda'):
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
