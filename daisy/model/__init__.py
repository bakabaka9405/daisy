"""模型模块"""

from . import mae
from . import moco
from .multihead import MultiHeadModel

__all__ = ['MultiHeadModel', 'mae', 'moco']
