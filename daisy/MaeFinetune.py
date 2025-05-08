import torch
from argparse import ArgumentParser, Namespace
from Util import SetGlobalSeed, EnableCUDNNBenchmark
from ImageDataLoader import LoadImages
from pathlib import Path

# from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms

# from MaeModel import MAE_ViT, ViT_Classifier
import math
from DataLoader import DataLoader
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	classification_report,
)
import timm
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy


def BuildArgumentParser() -> ArgumentParser:
	parser = ArgumentParser()
	parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--input_size', type=int, default=224)
	parser.add_argument('--batch_size', type=int, default=128)
	# parser.add_argument('--max_device_batch_size', type=int, default=256)
	parser.add_argument('--base_learning_rate', type=float, default=1e-3)
	parser.add_argument('--weight_decay', type=float, default=0.05)
	parser.add_argument('--mask_ratio', type=float, default=0.75)
	parser.add_argument('--total_epoch', type=int, default=30)
	parser.add_argument('--warmup_epoch', type=int, default=10)
	parser.add_argument('--save_interval', type=int, default=1)
	parser.add_argument('--dataset_path', type=str, default=r'C:\Resources\Datasets\20250301-Smile-Images-Wash\task_1')
	# parser.add_argument('--pretrained_model_path', type=str, default=r'c:\Temp\20250227_test_pretrain\checkpoint_epoch_400.pth')
	parser.add_argument('--output_path', type=str, default=r'C:\Temp\20250304-Smile-Images-Wash\task_1')
	return parser


def GetArguments() -> Namespace:
	return BuildArgumentParser().parse_args()


def Prepare(args: Namespace) -> None:
	if args.seed > 0:
		SetGlobalSeed(args.seed)
	else:
		EnableCUDNNBenchmark()

	args.device = torch.device(args.device)
	args.dataset_path = Path(args.dataset_path)
	args.train_dataset = LoadImages(args.dataset_path / 'train', 2, args.device)
	args.val_dataset = LoadImages(args.dataset_path / 'val', 2, args.device)
	print('dataset loaded')
	args.output_path = Path(args.output_path)
	args.output_path.mkdir(parents=True, exist_ok=True)
	# args.pretrained_model = torch.load(args.pretrained_model_path, map_location='cpu', weights_only=False)
	# print('pretrained model loaded')


def FinetuneMain(args: Namespace | None = None):
	if args is None:
		args = GetArguments()
	Prepare(args)
	# print(str(args))
	device: torch.device = args.device
	train_loader = DataLoader(args.train_dataset, args.batch_size, GetTrainTransform())
	val_loader = DataLoader(args.val_dataset, args.batch_size, GetValTransform(), shuffled=False, trunc_end=False)
	# print(args.dataset.transform)
	print('loader ready')
	# # writer = SummaryWriter()
	# model = ViT_Classifier(
	# 	encoder=args.pretrained_model.encoder,
	# 	num_classes=3,
	# ).to(device)

	model = timm.create_model('mambaout_femto', pretrained=True, num_classes=3).to(device)
	criterion = LabelSmoothingCrossEntropy()

	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=args.base_learning_rate * args.batch_size / 256,
		betas=(0.9, 0.999),
		weight_decay=args.weight_decay,
	)

	def lr_func(epoch: int):
		return min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))

	lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

	print('ready to train')
	for epoch in range(args.total_epoch):
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
		lr_scheduler.step()
		train_loss = sum(losses) / len(losses)

		# checkpoint_path = args.output_path / f'checkpoint_epoch_{epoch + 1}.pth'
		# torch.save(
		# 	{
		# 		'epoch': epoch + 1,
		# 		'model_state_dict': model.state_dict(),
		# 		'optimizer_state_dict': optimizer.state_dict(),
		# 		'loss': sum(losses) / len(losses),
		# 	},
		# 	checkpoint_path,
		# )

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
			acc, precision, recall, f1 = CalcScore(labels, preds)
			print(
				f'Epoch {epoch + 1}/{args.total_epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Acc: {acc:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}'
			)

			grid = [[0] * 3 for _ in range(3)]
			# cnt = grid.copy()
			for i, j in zip(labels, preds):
				grid[i][j] += 1
			print(grid)


if __name__ == '__main__':
	FinetuneMain()
