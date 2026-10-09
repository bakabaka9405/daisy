"""方法层接口与算法实现。"""

from daisy.method.base import Method
from daisy.method.classification import ClassificationMethod
from daisy.method.mae import MAEPretrainMethod
from daisy.method.moco import MoCoPretrainMethod
from daisy.method.multihead_classification import MultiHeadClassificationMethod
from daisy.method.supervised import SupervisedMethod

__all__ = [
	'ClassificationMethod',
	'MAEPretrainMethod',
	'Method',
	'MoCoPretrainMethod',
	'MultiHeadClassificationMethod',
	'SupervisedMethod',
]
