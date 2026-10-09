"""位置编码工具

参考:
- https://github.com/facebookresearch/mae
- MoCo v3: https://github.com/facebookresearch/moco-v3
"""

import math

import icontract
import numpy as np
import torch

from daisy.util import to_2tuple


def _num_checkpoint_pos_tokens(model, checkpoint_model: dict[str, torch.Tensor]) -> int:
	"""checkpoint 位置编码中除 cls 等额外 token 之外的位置 token 数。"""
	num_extra_tokens = model.pos_embed.shape[-2] - model.patch_embed.num_patches
	return checkpoint_model['pos_embed'].shape[-2] - num_extra_tokens


def _is_perfect_square(n: int) -> bool:
	return math.isqrt(n) ** 2 == n


def _grid_area(grid_size: int | tuple[int, int]) -> int:
	gh, gw = to_2tuple(grid_size)
	return int(gh) * int(gw)


@icontract.require(lambda embed_dim: embed_dim % 4 == 0)
@icontract.ensure(lambda embed_dim, grid_size, cls_token, result: result.shape == (int(cls_token) + _grid_area(grid_size), embed_dim))
def get_2d_sincos_pos_embed(
	embed_dim: int,
	grid_size: int | tuple[int, int],
	cls_token: bool = False,
) -> np.ndarray:
	"""
	生成 2D 正弦余弦位置编码

	Args:
		embed_dim: 嵌入维度
		grid_size: 网格大小（整数或 (grid_h, grid_w) 元组）
		cls_token: 是否包含 cls token

	Returns:
		pos_embed: [grid_h*grid_w, embed_dim] 或 [1+grid_h*grid_w, embed_dim]
	"""
	gh, gw = to_2tuple(grid_size)
	grid_h, grid_w = int(gh), int(gw)
	grid_h_arr = np.arange(grid_h, dtype=np.float32)
	grid_w_arr = np.arange(grid_w, dtype=np.float32)
	grid = np.meshgrid(grid_w_arr, grid_h_arr)  # w goes first
	grid = np.stack(grid, axis=0)

	grid = grid.reshape([2, 1, grid_h, grid_w])
	pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
	if cls_token:
		pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
	return pos_embed


@icontract.require(lambda grid: grid.shape[0] == 2)
@icontract.require(lambda embed_dim: embed_dim % 4 == 0)
@icontract.ensure(lambda embed_dim, grid, result: result.shape == (grid[0].size, embed_dim))
def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
	"""从网格生成 2D 位置编码"""
	# use half of dimensions to encode grid_h
	emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
	emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

	emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
	return emb


@icontract.require(lambda embed_dim: embed_dim % 2 == 0)
@icontract.ensure(lambda embed_dim, pos, result: result.shape == (pos.size, embed_dim))
def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
	"""
	从位置生成 1D 正弦余弦位置编码

	Args:
		embed_dim: 输出维度
		pos: 位置列表，大小为 (M,)

	Returns:
		out: (M, D)
	"""
	omega = np.arange(embed_dim // 2, dtype=float)
	omega /= embed_dim / 2.0
	omega = 1.0 / 10000**omega  # (D/2,)

	pos = pos.reshape(-1)  # (M,)
	out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

	emb_sin = np.sin(out)  # (M, D/2)
	emb_cos = np.cos(out)  # (M, D/2)

	emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
	return emb


@icontract.require(
	lambda model, checkpoint_model, source_grid_size: (
		'pos_embed' not in checkpoint_model
		or source_grid_size is None
		or math.prod(source_grid_size) == _num_checkpoint_pos_tokens(model, checkpoint_model)
	),
	'Explicit source_grid_size does not match checkpoint positional token count.',
)
@icontract.require(
	lambda model, checkpoint_model, source_grid_size: (
		'pos_embed' not in checkpoint_model
		or source_grid_size is not None
		or _is_perfect_square(_num_checkpoint_pos_tokens(model, checkpoint_model))
	),
	'Cannot infer square source grid size; source_grid_size must be explicitly provided for non-square checkpoints.',
)
def interpolate_pos_embed(
	model,
	checkpoint_model: dict[str, torch.Tensor],
	source_grid_size: tuple[int, int] | None = None,
) -> None:
	"""
	插值位置编码以支持不同输入分辨率。

	Args:
		model: 目标模型
		checkpoint_model: checkpoint 中的模型状态字典
		source_grid_size: 源 checkpoint 的网格大小 (grid_h, grid_w)。
			若为 None 则尝试推断正方形网格；若源网格为矩形（即使 token 总数为完全平方数），必须显式指定。
	"""
	if 'pos_embed' in checkpoint_model:
		pos_embed_checkpoint = checkpoint_model['pos_embed']
		embedding_size = pos_embed_checkpoint.shape[-1]
		num_extra_tokens = model.pos_embed.shape[-2] - model.patch_embed.num_patches

		if source_grid_size is None:
			orig_h = math.isqrt(_num_checkpoint_pos_tokens(model, checkpoint_model))
			orig_grid_size = (orig_h, orig_h)
		else:
			orig_grid_size = source_grid_size

		target_grid_size: tuple[int, int] = model.patch_embed.grid_size
		# class_token and dist_token are kept unchanged
		if orig_grid_size != target_grid_size:
			print(f'Position interpolate from {orig_grid_size[0]}x{orig_grid_size[1]} to {target_grid_size[0]}x{target_grid_size[1]}')
			extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
			# only the position tokens are interpolated
			pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
			pos_tokens = pos_tokens.reshape(-1, orig_grid_size[0], orig_grid_size[1], embedding_size).permute(0, 3, 1, 2)
			pos_tokens = torch.nn.functional.interpolate(
				pos_tokens,
				size=target_grid_size,
				mode='bicubic',
				align_corners=False,
				antialias=False,
			)
			pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
			new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
			checkpoint_model['pos_embed'] = new_pos_embed
