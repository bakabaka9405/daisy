"""模型模块"""

from . import mae
from . import moco
from .multihead import MultiHeadModel
from .ordinal import CLMHead, CORALHead, CORNHead, ClassificationHead, ContrastiveClassificationHead, OrdinalHead, ScalarHead

__all__ = [
	'CLMHead',
	'CORALHead',
	'CORNHead',
	'ClassificationHead',
	'ContrastiveClassificationHead',
	'MultiHeadModel',
	'OrdinalHead',
	'ScalarHead',
	'mae',
	'moco',
]
