import torch
from torch import nn
from pathlib import Path
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	classification_report,
)
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils.model import freeze as FreezeModelLayers
from typing import Literal
import logging
from collections.abc import Iterable


def CalcPredictionScore(labels, preditions) -> tuple[float, float, float, float]:
	acc = accuracy_score(labels, preditions, normalize=True)
	precision = precision_score(labels, preditions, average='macro', zero_division=1)
	recall = recall_score(labels, preditions, average='macro', zero_division=1)
	f1 = f1_score(labels, preditions, average='macro', zero_division=1)
	return float(acc), float(precision), float(recall), float(f1)


def TrainClassifier(
	device: torch.device,
	model: nn.Module,
	num_classes: int,
	train_loader: Iterable,
	val_loader: Iterable,
	total_epoch: int,
	optimizer: torch.optim.Optimizer,
	criterion: nn.Module,
	lr_scheduler=None,
	output_path: Path | None = None,
	save_checkpoint: bool = False,
	save_interval: int = 1,
	logger: logging.Logger | None = None,
	log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO',
) -> None:
	if not logger:
		logger = logging.Logger(__name__)

	logger.setLevel(log_level)
	logger.addHandler(logging.StreamHandler())
	if output_path:
		output_path.mkdir(parents=True, exist_ok=True)
		logger.addHandler(logging.FileHandler(output_path / 'train.log', mode='w'))

	best = {
		'f1': 0,
	}
	logger.info(f'Training start, total epoch: {total_epoch}')
	for epoch in range(total_epoch):
		model.train()
		losses = []
		for img, label in train_loader:
			optimizer.zero_grad()
			label = label.to(device)
			pred = model(img)
			loss = criterion(pred, label)
			loss.backward()
			optimizer.step()
			losses.append(loss.item())
		if lr_scheduler:
			lr_scheduler.step(epoch)
		train_loss = sum(losses) / len(losses)

		if save_checkpoint and output_path and (epoch + 1) % save_interval == 0:
			(output_path / 'checkpoint').mkdir(parents=True, exist_ok=True)
			checkpoint_path = output_path / 'checkpoint' / f'checkpoint_epoch_{epoch + 1}.pth'
			torch.save(model.state_dict(), checkpoint_path)

		model.eval()
		with torch.no_grad():
			losses = []
			preds = []
			labels = []
			for img, label in val_loader:
				label = label.to(device)
				pred = model(img)
				loss = criterion(pred, label)
				losses.append(loss.item())
				preds.extend(pred.argmax(dim=1).cpu().numpy())
				labels.extend(label.cpu().numpy())
			val_loss = sum(losses) / len(losses)
			acc, precision, recall, f1 = CalcPredictionScore(labels, preds)
			logger.info(
				f'Epoch {epoch + 1}/{total_epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Acc: {acc:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}'
			)

			grid = [[0] * num_classes for _ in range(num_classes)]
			for i, j in zip(labels, preds):
				grid[i][j] += 1
			logger.info(grid)

			if f1 > best['f1']:  # type: ignore
				best = {'f1': f1, 'epoch': epoch + 1, 'model': model.state_dict(), 'label': labels, 'pred': preds}

	if best:
		logger.info(f'best F1: {best['f1']:.6f} at epoch {best['epoch']}')
		if output_path:
			torch.save(best['model'], output_path / 'best_model.pth')
		repo = str(classification_report(best['label'], best['pred'], digits=4, zero_division=1))  # type: ignore
		logger.info(f'best classification report:\n{repo}')
		if output_path:
			with open(output_path / 'best_classification_report.txt', 'w') as f:
				f.write(repo)


if __name__ == '__main__':
	import timm
	from Util import EnableCUDNNBenchmark, GetDefaultTrainTransform, GetDefaultValTransform, ChangeModelClassifier, FreezeModel
	from ImageDataLoader import LoadImages
	from DataLoader import DataLoader as SimpleDataLoader

	EnableCUDNNBenchmark()
	blr = 1e-3
	batch_size = 128
	epochs = 200
	warmup_epochs = 50
	save_interval = 40
	device = torch.device('cuda')
	num_classes = [0, 3, 4, 5, 3, 3, 3, 3]
	for i in range(3, 8):
		model = timm.create_model('resnet34', pretrained=True, num_classes=num_classes[i]).to(device)
		print(model)
		dataset_path = Path(rf'c:\Resources\Datasets\Smile-20250305\finetune_task_{i}')
		output_path = Path(rf'C:\Temp\20250311-test\finetune_task_{i}')
		train_dataset = LoadImages(dataset_path / 'train', prefetchLevel=2, device=device)
		val_dataset = LoadImages(dataset_path / 'val', prefetchLevel=2, device=device)
		train_loader = SimpleDataLoader(
			train_dataset,
			batch_size=128,
			transform=GetDefaultTrainTransform(),
		)
		val_loader = SimpleDataLoader(
			val_dataset,
			batch_size=128,
			transform=GetDefaultValTransform(),
		)
		criterion = LabelSmoothingCrossEntropy()
		optimizer = torch.optim.AdamW(
			model.parameters(),
			lr=blr * batch_size / 256,
			betas=(0.9, 0.999),
			weight_decay=0.05,
		)
		lr_scheduler = CosineLRScheduler(optimizer, epochs, warmup_t=warmup_epochs, lr_min=1e-5)

		TrainClassifier(
			device=device,
			model=model,
			num_classes=num_classes[i],
			train_loader=train_loader,
			val_loader=val_loader,
			total_epoch=epochs,
			optimizer=optimizer,
			criterion=criterion,
			lr_scheduler=lr_scheduler,
			output_path=output_path,
			save_checkpoint=True,
			save_interval=save_interval,
		)
