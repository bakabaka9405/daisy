"""方法层接口与算法实现。"""

from daisy.method.base import Method
from daisy.method.classification import ClassificationMethod
from daisy.method.mae import MAEPretrainMethod
from daisy.method.moco import MoCoPretrainMethod
from daisy.method.multihead_classification import MultiHeadClassificationMethod
from daisy.method.ordinal import (
	METHODS,
	CEMethod,
	CLMMethod,
	CORALMethod,
	CORNMethod,
	DistanceCEMethod,
	LabelSmoothingCEMethod,
	MSERankHingeMethod,
	MSERankNetMethod,
	OrdinalContrastiveMethod,
	RPSMethod,
	SORDMethod,
	UnimodalMethod,
	WassersteinMethod,
)

__all__ = [
	'METHODS',
	'CEMethod',
	'ClassificationMethod',
	'CLMMethod',
	'CORALMethod',
	'CORNMethod',
	'DistanceCEMethod',
	'LabelSmoothingCEMethod',
	'MAEPretrainMethod',
	'MSERankHingeMethod',
	'MSERankNetMethod',
	'Method',
	'MoCoPretrainMethod',
	'MultiHeadClassificationMethod',
	'OrdinalContrastiveMethod',
	'RPSMethod',
	'SORDMethod',
	'UnimodalMethod',
	'WassersteinMethod',
]
