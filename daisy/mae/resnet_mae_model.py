import torch
from torch import nn, Tensor
from collections.abc import Callable
from torchvision.models.resnet import ResNet
import numpy
import cv2


class ResNetMaeEncoder(nn.Module):
	def __init__(
		self,
		model_t: type[ResNet] | Callable[..., ResNet],
		mask_ratio: float = 0.75,
		**model_kwargs,
	):
		super().__init__()

		model = model_t(**model_kwargs)
		self.conv1 = model.conv1
		self.bn1 = model.bn1
		self.relu = model.relu
		self.maxpool = model.maxpool
		self.layer1 = model.layer1
		self.layer2 = model.layer2
		self.layer3 = model.layer3
		self.layer4 = model.layer4
		self.mask_ratio = mask_ratio

	def _patchify(self, x: Tensor) -> Tensor:
		# x: [B, 3, 224, 224]
		# patches: [B, 3, 32*32, 7, 7]

		x = x.unfold(2, 32, 32).unfold(3, 7, 7)
		x = x.contiguous().view(x.size(0), x.size(1), -1, 7, 7)

		return x

	def _unpatchify(self, x: Tensor) -> Tensor:
		# x: [B, 3, 32*32, 7, 7]
		# patches: [B, 3, 224, 224]

		x = x.view(x.size(0), x.size(1), 32, 32, 7, 7)
		x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
		x = x.view(x.size(0), x.size(1), 224, 224)

		return x

	def _random_mask(self, x: Tensor) -> Tensor:
		x = self._patchify(x)
		mask = (torch.rand(x.size(0), x.size(2)) < 0.75).to(x.device)
		mask = mask.view(x.size(0), 1, x.size(2), 1, 1)
		mask = mask.expand(-1, 3, -1, 7, 7)
		x = x.masked_fill(mask, 0)
		x = self._unpatchify(x)
		return x

	def forward(self, x: Tensor):
		x = self._random_mask(x)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		return x


class SimpleDCNN(nn.Module):
	def __init__(self):
		super().__init__()

		# self.linear = nn.Linear(1024, 8 * 7 * 7)  # 先将512维向量线性映射到适合卷积的维度 (例如: 8通道, 7x7)
		# self.relu = nn.ReLU(inplace=True)

		self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(256)
		self.relu1 = nn.ReLU(inplace=True)

		self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU(inplace=True)

		self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU(inplace=True)

		self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False)
		self.bn4 = nn.BatchNorm2d(3)
		self.relu4 = nn.ReLU(inplace=True)

		self.deconv5 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
		self.sigmoid = nn.Sigmoid()  # 确保输出在0-1之间，适合图像

	def forward(self, x):
		# x = self.linear(x)
		# x = self.relu(x)
		# print(x.shape)

		x = x.view(x.size(0), 512, 7, 7)  # 调整形状为 (batch_size, 8, 7, 7)

		x = self.deconv1(x)
		x = self.bn1(x)
		x = self.relu1(x)

		x = self.deconv2(x)
		x = self.bn2(x)
		x = self.relu2(x)

		x = self.deconv3(x)
		x = self.bn3(x)
		x = self.relu3(x)

		x = self.deconv4(x)
		x = self.bn4(x)
		x = self.relu4(x)

		x = self.deconv5(x)
		x = self.sigmoid(x)  # 使用 sigmoid 将像素值限制在 0-1 之间

		return x


class CnnMaeDecoder(nn.Module):
	model: nn.Module

	def __init__(
		self,
		model: type[nn.Module] | Callable[..., nn.Module] = SimpleDCNN,
		**model_kwargs,
	):
		super().__init__()

		self.model = model(**model_kwargs)

	def forward(self, x: Tensor):
		return self.model(x)


class CnnMae(nn.Module):
	encoder: ResNetMaeEncoder
	decoder: CnnMaeDecoder

	def __init__(
		self,
		encoder: ResNetMaeEncoder,
		decoder: CnnMaeDecoder,
	):
		super().__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, x: Tensor):
		x = self.encoder(x)
		x = self.decoder(x)
		return x


def show_tensor(x: Tensor):
	img = x.cpu().numpy()
	img = img.squeeze(0).transpose(1, 2, 0)
	img = (img * 255).astype(numpy.uint8)
	cv2.imshow('output', img)
	cv2.waitKey(0)


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	from torchvision.models import resnet34
	from pathlib import Path
	import daisy
	from daisy.dataset import DiskDataset
	from timm.data.loader import MultiEpochsDataLoader

	imgs = []
	for i in Path(r'C:\Resources\Datasets\35785').iterdir():
		imgs.append(i)

	dataset = DiskDataset(imgs, [0] * len(imgs))
	transform = daisy.util.transform.get_default_val_transform()
	dataset.setTransform(transform)

	loader = MultiEpochsDataLoader(
		dataset,
		batch_size=128,
		shuffle=True,
		num_workers=8,
		pin_memory=True,
	)

	# show_tensor(x[0:1])
	encoder = ResNetMaeEncoder(
		model_t=resnet34,
		mask_ratio=0.75,
	).to(device)
	decoder = CnnMaeDecoder().to(device)

	model = CnnMae(
		encoder=encoder,
		decoder=decoder,
	).to(device)

	print(model)

	criterion = nn.MSELoss()

	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

	scaler = torch.GradScaler()
	for _ in range(400):
		losses = 0.0
		for x, _ in loader:
			x = x.to(device)
			optimizer.zero_grad()

			with torch.autocast(device_type='cuda'):
				output = model(x)
				loss = criterion(output, x)
				losses += loss.item()

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		losses /= len(loader)
		print(f'loss: {losses}')

	# 测试模型
	model.eval()
	with torch.no_grad():
		img = next(iter(loader))[0][0].to(device)
		output = model(img)
		show_tensor(output[0:1])
