import torch
from torch import Tensor
import numpy
from timm.layers.weight_init import trunc_normal_
from timm.models.vision_transformer import Block
from einops import rearrange
from einops.layers.torch import Rearrange


def GenerateRandomIndexes(size: int) -> tuple[numpy.ndarray, numpy.ndarray]:
	"""
	生成一个随机的索引序列，包括正向索引和反向索引

	:param size: 索引序列的长度
	"""
	forward_indexes = numpy.random.permutation(size)
	backward_indexes = numpy.argsort(forward_indexes)
	return forward_indexes, backward_indexes


def TensorGather2D(input: Tensor, indexes2D: Tensor) -> Tensor:
	"""
	将 input 的第 0 维度的数据按照 indexes2D 的数据进行索引，返回一个新的 Tensor

	`output[i][j][k] = input[indexes2D[i][j]][j][k]`

	:param input: (T, B, C)
	:param indexes2D: (T, B)
	:return: (T, B, C)
	"""
	indexes = indexes2D.unsqueeze(2).expand(-1, -1, input.size(2))
	return torch.gather(input, 0, indexes)


class PatchShuffle(torch.nn.Module):
	maskRatio: float

	def __init__(self, maskRatio: float):
		super().__init__()
		self.maskRatio = maskRatio

	def forward(self, patches: Tensor) -> tuple[Tensor, Tensor]:
		T, B, C = patches.size()
		indexes2D = [GenerateRandomIndexes(T) for _ in range(B)]
		# 相当于转置了一下，每个 patch 的索引放在最后一个维度
		# 目前不清楚这样做在性能上的优势
		forwardIndexes2D = torch.as_tensor(numpy.stack([i[0] for i in indexes2D], axis=-1), dtype=torch.long).to(patches.device)
		backwardIndexes2D = torch.as_tensor(numpy.stack([i[1] for i in indexes2D], axis=-1), dtype=torch.long).to(patches.device)

		shuffledPatches = TensorGather2D(patches, forwardIndexes2D)  # 随机打乱了数据的patch，这样所有的patch都被打乱了
		unmaskedPatches = shuffledPatches[: int(T * (1 - self.maskRatio))]  # 得到未mask的pacth [T*0.25, B, C]

		return unmaskedPatches, backwardIndexes2D


class MAE_Encoder(torch.nn.Module):
	cls_token: torch.nn.Parameter
	pos_embedding: torch.nn.Parameter
	shuffle: PatchShuffle
	patchify: torch.nn.Conv2d
	transformer: torch.nn.Sequential
	layer_norm: torch.nn.LayerNorm

	def __init__(
		self,
		image_size=224,
		patch_size=16,
		embed_dim=192,
		num_layer=4,
		num_head=3,
		mask_ratio=0.75,
	):
		super().__init__()
		self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
		self.pos_embedding = torch.nn.Parameter(torch.randn((image_size // patch_size) ** 2, 1, embed_dim))
		self.shuffle = PatchShuffle(mask_ratio)
		self.patchify = torch.nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)  # (3, dim, patch, patch) 的卷积
		self.transformer = torch.nn.Sequential(*[Block(embed_dim, num_head) for _ in range(num_layer)])
		self.layer_norm = torch.nn.LayerNorm(embed_dim)

		# 初始化类别编码和向量编码
		trunc_normal_(self.cls_token, std=0.02)
		trunc_normal_(self.pos_embedding, std=0.02)

	def forward(self, img: Tensor):
		patches = self.patchify(img)
		patches = rearrange(patches, 'b c h w -> (h w) b c')
		patches = patches + self.pos_embedding
		patches, backwardIndexes2D = self.shuffle(patches)
		patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
		patches = rearrange(patches, 't b c -> b t c')
		features = self.layer_norm(self.transformer(patches))
		features = rearrange(features, 'b t c -> t b c')

		return features, backwardIndexes2D


class MAE_Decoder(torch.nn.Module):
	mask_token: torch.nn.Parameter
	pos_embedding: torch.nn.Parameter
	transformer: torch.nn.Sequential
	head: torch.nn.Linear
	patch2img: Rearrange

	def __init__(
		self,
		image_size=224,
		patch_size=16,
		embed_dim=192,
		num_layer=4,
		num_head=3,
	):
		super().__init__()
		self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, embed_dim))
		self.transformer = torch.nn.Sequential(*[Block(embed_dim, num_head) for _ in range(num_layer)])
		self.head = torch.nn.Linear(embed_dim, 3 * patch_size**2)
		self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size // patch_size)
		trunc_normal_(self.mask_token, std=0.02)
		trunc_normal_(self.pos_embedding, std=0.02)

	def forward(self, features: Tensor, backward_indexes: Tensor) -> tuple[Tensor, Tensor]:
		T = features.size(0)
		backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
		features = torch.cat(
			[
				features,
				self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1),
			],
			dim=0,
		)
		features = TensorGather2D(
			features,
			backward_indexes,
		)
		features = features + self.pos_embedding  # 加上了位置编码的信息

		features = rearrange(features, 't b c -> b t c')
		features = self.transformer(features)
		features = rearrange(features, 'b t c -> t b c')
		features = features[1:]  # remove global feature 去掉全局信息，得到图像信息

		patches = self.head(features)  # 用head得到patchs
		mask = torch.zeros_like(patches)
		mask[T:] = 1  # mask其他的像素全部设为 1
		mask = TensorGather2D(mask, backward_indexes[1:] - 1)
		img = self.patch2img(patches)  # 得到 重构之后的 img
		mask = self.patch2img(mask)

		return img, mask


class MAE_ViT(torch.nn.Module):
	encoder: MAE_Encoder
	decoder: MAE_Decoder

	def __init__(
		self,
		image_size=224,
		patch_size=16,
		embed_dim=192,
		num_encoder_layer=12,
		num_encoder_head=3,
		num_decoder_layer=4,
		num_decoder_head=3,
		mask_ratio=0.75,
	):
		super().__init__()
		self.encoder = MAE_Encoder(image_size, patch_size, embed_dim, num_encoder_layer, num_encoder_head, mask_ratio)
		self.decoder = MAE_Decoder(image_size, patch_size, embed_dim, num_decoder_layer, num_decoder_head)

	def forward(self, img: Tensor) -> tuple[Tensor, Tensor]:
		features: Tensor
		backward_indexes: Tensor
		predicted_img: Tensor
		mask: Tensor

		features, backward_indexes = self.encoder(img)
		predicted_img, mask = self.decoder(features, backward_indexes)
		return predicted_img, mask


class ViT_Classifier(torch.nn.Module):
	def __init__(self, encoder: MAE_Encoder, num_classes=10) -> None:
		super().__init__()
		self.cls_token = encoder.cls_token
		self.pos_embedding = encoder.pos_embedding
		self.patchify = encoder.patchify
		self.transformer = encoder.transformer
		self.layer_norm = encoder.layer_norm
		self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

	def forward(self, img: Tensor) -> Tensor:
		patches = self.patchify(img)
		patches = rearrange(patches, 'b c h w -> (h w) b c')
		patches = patches + self.pos_embedding
		patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
		patches = rearrange(patches, 't b c -> b t c')
		features = self.layer_norm(self.transformer(patches))
		features = rearrange(features, 'b t c -> t b c')
		logits = self.head(features[0])
		return logits