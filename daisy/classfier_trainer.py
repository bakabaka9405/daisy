import daisy
import torch
from torch.optim import AdamW
from timm import create_model
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from timm.data.loader import MultiEpochsDataLoader
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_classfier():
	daisy.util.enable_cudnn_benchmark()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	num_classes = [3, 4, 5, 3, 3, 3, 3]
	label_offset = [-1, 0, 0, -1, -1, -1, -1]

	epochs = 100
	warmup_epochs = 20
	lr = 1e-3

	for i in range(1, 8):
		feeder = daisy.feeder.load_feeder_from_sheet(
			r'C:\Resources\Datasets\微笑图片标注汇总25-3-4\微笑图片标注汇总25-3-4',
			r'C:\Resources\Datasets\微笑图片标注汇总25-3-4\微笑图片标注汇总25-3-4.xlsx',
			have_header=True,
			column=i,
			label_offset=label_offset[i - 1],
		)


		files, labels = feeder.fetch()

		dataset = daisy.dataset.DiskDataset(files, labels)
		train_transform = daisy.util.transform.get_default_train_transform()
		val_transform = daisy.util.transform.get_default_val_transform()

		train_dataset, val_dataset = daisy.dataset.dataset_split.minimum_class_proportional_split(
			dataset,
			val_minimum_size=20,
			force_fetch_minimum_size=True,
		)

		train_dataset = daisy.dataset.dataset_split.trunc_max_class(train_dataset)

		train_dataset.setTransform(train_transform)
		val_dataset.applyTransform(val_transform)

		train_loader = MultiEpochsDataLoader(
			train_dataset,
			batch_size=64,
			shuffle=True,
			num_workers=4,
			pin_memory=True,
		)

		val_loader = MultiEpochsDataLoader(
			val_dataset,
			batch_size=64,
			shuffle=False,
			pin_memory=True,
		)

		model = create_model('resnet34', pretrained=True, num_classes=num_classes[i - 1])
		model.to(device)

		criterion = LabelSmoothingCrossEntropy()

		optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))

		def lr_func(epoch: int):
			return min((epoch + 1) / (warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / max(1, epochs) * math.pi) + 1))

		lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

		for epoch in range(epochs):
			model.train()
			losses = 0.0
			for data, target in train_loader:
				data, target = data.to(device), target.to(device)

				optimizer.zero_grad()
				output = model(data)
				loss = criterion(output, target)
				loss.backward()
				losses += loss.item()
				optimizer.step()

			lr_scheduler.step()

			y_pred = []
			y_true = []

			with torch.no_grad():
				model.eval()
				for data, target in val_loader:
					data, target = data.to(device), target.to(device)
					output = model(data)
					predicted = torch.argmax(output, dim=1)
					y_pred.extend(predicted.cpu().numpy())
					y_true.extend(target.cpu().numpy())

				confusion_matrix = [[0] * num_classes[i - 1] for _ in range(num_classes[i - 1])]
				for t, p in zip(y_true, y_pred):
					confusion_matrix[t][p] += 1

				print(confusion_matrix)

				acc = accuracy_score(y_true, y_pred)
				precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
				recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
				f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
				print(
					f'Epoch {epoch + 1}/{epochs}, Loss: {losses / len(train_loader)}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}'
				)
