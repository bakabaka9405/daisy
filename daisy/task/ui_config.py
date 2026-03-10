"""UI 字段配置"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ComponentType = Literal['textbox', 'number', 'slider', 'dropdown', 'checkbox', 'hidden', 'textarea', 'file']


@dataclass(slots=True, frozen=True)
class UIFieldConfig:
	"""单个字段的 UI 配置"""

	component: ComponentType | None = None
	label: str | None = None
	choices: tuple[str, ...] = ()
	allow_custom: bool = False
	min_value: float | None = None
	max_value: float | None = None
	step: float | None = None
	hidden: bool = False
	info: str | None = None
