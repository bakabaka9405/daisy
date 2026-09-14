"""方法层接口与算法实现。"""

from daisy.method.base import Method
from daisy.method.classification import ClassificationMethod
from daisy.method.mae import MAEFinetuneMethod, MAEPretrainMethod
from daisy.method.moco import MoCoPretrainMethod
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
	'MAEFinetuneMethod',
	'MAEPretrainMethod',
	'MSERankHingeMethod',
	'MSERankNetMethod',
	'Method',
	'MoCoPretrainMethod',
	'OrdinalContrastiveMethod',
	'RPSMethod',
	'SORDMethod',
	'UnimodalMethod',
	'WassersteinMethod',
]
