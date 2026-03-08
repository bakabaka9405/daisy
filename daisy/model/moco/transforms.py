"""MoCo 对比学习数据增强

TwoCropsTransform: 对一张图像应用两个 transform，返回 [view1, view2]。
- v2: 对称增强 — 同一个 transform 用两次
- v3: 非对称增强 — 两个不同 transform
"""

from torch import Tensor


class TwoCropsTransform:
	"""对一张图像应用两个 transform，返回 [view1, view2]"""

	def __init__(self, transform1, transform2=None):
		self.transform1 = transform1
		self.transform2 = transform2 or transform1

	def __call__(self, x) -> list[Tensor]:
		return [self.transform1(x), self.transform2(x)]
