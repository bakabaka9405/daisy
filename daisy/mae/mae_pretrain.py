import torch
from argparse import ArgumentParser, Namespace
from util import SetGlobalSeed, EnableCUDNNBenchmark
from ImageDataLoader import LoadImages
from pathlib import Path

# from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from lib.mae.mae_model import MAE_ViT
import math
from DataLoader import DataLoader


class ZeroOneNormalize:
	def __call__(self, tensor: torch.Tensor):
		return tensor.float().div(255)


def GetTrainTransform():
	return transforms.Compose(
		[
			transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
			transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
			transforms.Resize((224, 224)),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)


def BuildArgumentParser() -> ArgumentParser:
	parser = ArgumentParser()
	parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--input_size', type=int, default=224)
	parser.add_argument('--batch_size', type=int, default=32)
	# parser.add_argument('--max_device_batch_size', type=int, default=256)
	parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
	parser.add_argument('--weight_decay', type=float, default=0.05)
	parser.add_argument('--mask_ratio', type=float, default=0.75)
	parser.add_argument('--total_epoch', type=int, default=400)
	parser.add_argument('--warmup_epoch', type=int, default=50)
	parser.add_argument('--save_interval', type=int, default=50)
	parser.add_argument('--dataset_path', type=str, default=r'C:\Resources\Datasets\Smile-20250223\finetune_task_1\train')
	parser.add_argument('--output_path', type=str, default=r'C:\Temp\20250227_test_pretrain')
	return parser


def GetArguments() -> Namespace:
	return BuildArgumentParser().parse_args()


def Prepare(args: Namespace) -> None:
	if args.seed > 0:
		SetGlobalSeed(args.seed)
	else:
		EnableCUDNNBenchmark()

	args.device = torch.device(args.device)
	# args.load_batch_size = min(args.batch_size, args.max_device_batch_size)
	args.dataset_path = Path(args.dataset_path)
	args.dataset = LoadImages(args.dataset_path, 2, args.device)
	print('dataset loaded')
	args.output_path = Path(args.output_path)
	args.output_path.mkdir(parents=True, exist_ok=True)


def PretrainMain(args: Namespace | None = None):
	if args is None:
		args = GetArguments()
	Prepare(args)
	# print(str(args))
	device: torch.device = args.device
	loader = DataLoader(args.dataset, args.batch_size, GetTrainTransform())
	# print(args.dataset.transform)
	print('loader ready')
	# writer = SummaryWriter()
	model = MAE_ViT().to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=args.base_learning_rate * args.batch_size / 256,
		betas=(0.9, 0.95),
		weight_decay=args.weight_decay,
	)

	def lr_func(epoch: int):
		return min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))

	lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

	print('ready to train')
	for epoch in range(args.total_epoch):
		model.train()
		losses = []
		for img, _ in loader:
			optimizer.zero_grad()
			img = img.to(device)
			predicted_img, mask = model(img)
			loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
			loss.backward()
			optimizer.step()
			losses.append(loss.item())
		lr_scheduler.step()
		print(f'Epoch {epoch + 1}/{args.total_epoch}, Loss: {sum(losses) / len(losses)}')

		if (epoch + 1) % args.save_interval == 0:
			checkpoint_path = args.output_path / f'checkpoint_epoch_{epoch + 1}.pth'
			torch.save(
				model,
				checkpoint_path,
			)
			# print(f'Checkpoint saved at {checkpoint_path}')


if __name__ == '__main__':
	PretrainMain()
