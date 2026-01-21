"""UI 元数据定义

用于在 Pydantic Field 的 json_schema_extra 中携带 UI 配置信息。
"""

from dataclasses import dataclass
from typing import Any, Literal


ComponentType = Literal['textbox', 'number', 'slider', 'dropdown', 'checkbox', 'hidden', 'textarea', 'file']


@dataclass
class UIFieldMeta:
	"""UI 字段元数据

	用于描述配置字段在 Gradio UI 中的呈现方式。

	示例用法:
	```python
	name: str = Field(
		default='resnet34',
		json_schema_extra={
			'ui': UIFieldMeta(
				component='dropdown',
				label='模型名称',
				choices=['resnet34', 'resnet50', 'efficientnet_b0'],
				allow_custom=True,
			).to_dict()
		}
	)
	```
	"""

	# 组件类型
	component: ComponentType | None = None
	# 显示标签
	label: str | None = None
	# 下拉选项 (dropdown)
	choices: list[str] | None = None
	# 允许自定义值 (dropdown)
	allow_custom: bool = False
	# 数值范围 (slider/number)
	min_value: float | None = None
	max_value: float | None = None
	step: float | None = None
	# 是否隐藏
	hidden: bool = False
	# 提示信息
	info: str | None = None
	# UI 分组名 (用于将相关字段组织在一起)
	group: str | None = None

	def to_dict(self) -> dict[str, Any]:
		"""转换为字典，用于 json_schema_extra"""
		return {
			k: v
			for k, v in {
				'component': self.component,
				'label': self.label,
				'choices': self.choices,
				'allow_custom': self.allow_custom,
				'min_value': self.min_value,
				'max_value': self.max_value,
				'step': self.step,
				'hidden': self.hidden,
				'info': self.info,
				'group': self.group,
			}.items()
			if v is not None and v and v != []
		}

	@classmethod
	def from_dict(cls, data: dict[str, Any]) -> 'UIFieldMeta':
		"""从字典创建"""
		return cls(
			component=data.get('component'),
			label=data.get('label'),
			choices=data.get('choices'),
			allow_custom=data.get('allow_custom', False),
			min_value=data.get('min_value'),
			max_value=data.get('max_value'),
			step=data.get('step'),
			hidden=data.get('hidden', False),
			info=data.get('info'),
			group=data.get('group'),
		)
