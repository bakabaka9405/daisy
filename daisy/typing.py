import numpy as np
from numpy.typing import NDArray, ArrayLike
from torch import Tensor
from dataclasses import dataclass, replace
from typing import Self

type VectorI32 = np.ndarray[tuple[int], np.dtype[np.int32]]
type VectorI64 = np.ndarray[tuple[int], np.dtype[np.int64]]
type VectorF32 = np.ndarray[tuple[int], np.dtype[np.float32]]
type VectorF64 = np.ndarray[tuple[int], np.dtype[np.float64]]
type MatrixI64 = np.ndarray[tuple[int, int], np.dtype[np.int64]]
type MatrixF32 = np.ndarray[tuple[int, int], np.dtype[np.float32]]
type MatrixF64 = np.ndarray[tuple[int, int], np.dtype[np.float64]]

type ImageU8 = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]

type NDArrayF64 = NDArray[np.float64]


def to_array(x: Tensor | ArrayLike) -> NDArray:
	if isinstance(x, Tensor):
		return x.numpy(force=True)
	return np.asarray(x)


@dataclass(frozen=True)
class Replaceable:
	def replace(self: Self, **changes) -> Self:
		return replace(self, **changes)
