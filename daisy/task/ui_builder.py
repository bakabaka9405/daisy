"""Pydantic 配置类到 Gradio UI 的自动转换器"""

from __future__ import annotations

import types
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .ui_schema import UIFieldMeta


def get_ui_meta(field_info: FieldInfo) -> UIFieldMeta | None:
	"""从 Field 的 json_schema_extra 中提取 UI 元数据"""
	extra = field_info.json_schema_extra
	if extra is None:
		return None
	# json_schema_extra 可以是 dict 或 callable
	if callable(extra):
		return None
	if 'ui' in extra and isinstance(extra['ui'], dict):
		return UIFieldMeta.from_dict(extra['ui'])
	return None


def infer_component_type(
	field_type: type,
	field_name: str,
	default_value: Any = None,
) -> tuple[str, dict[str, Any]]:
	"""根据类型推断 Gradio 组件类型

	返回 (component_type, extra_kwargs)
	"""
	origin = get_origin(field_type)
	args = get_args(field_type)

	# 处理 Optional 类型 (Union[X, None])
	if origin is types.UnionType or (origin is not None and hasattr(origin, '__name__') and origin.__name__ == 'Union'):
		# 过滤掉 None
		non_none_args = [a for a in args if a is not type(None)]
		if len(non_none_args) == 1:
			return infer_component_type(non_none_args[0], field_name, default_value)

	# Literal 类型 -> dropdown
	if origin is Literal:
		choices = list(args)
		return 'dropdown', {'choices': choices}

	# bool -> checkbox
	if field_type is bool:
		return 'checkbox', {}

	# int/float -> number
	if field_type in (int, float):
		return 'number', {}

	# str -> textbox
	if field_type is str:
		return 'textbox', {}

	# list[str] -> textbox (逗号分隔)
	if origin is list:
		return 'textbox', {'info': '多个值用逗号分隔'}

	# 嵌套 BaseModel -> 递归处理 (返回特殊标记)
	if isinstance(field_type, type) and issubclass(field_type, BaseModel):
		return 'nested', {'model_class': field_type}

	# 默认 textbox
	return 'textbox', {}


def create_gradio_component(
	gr,
	field_name: str,
	field_info: FieldInfo,
	field_type: type,
	ui_override: dict[str, Any] | None = None,
):
	"""创建单个 Gradio 组件

	返回 (component, component_type)
	"""
	# 获取 UI 元数据
	ui_meta = get_ui_meta(field_info)

	# 合并覆盖配置
	override = ui_override or {}
	if ui_meta:
		override = {**ui_meta.to_dict(), **override}

	# 检查是否隐藏
	if override.get('hidden'):
		return None, 'hidden'

	# 获取默认值
	default_value = field_info.default
	if default_value is None and field_info.default_factory is not None:
		try:
			factory = field_info.default_factory
			if callable(factory):
				default_value = factory()  # type: ignore[call-arg]
		except Exception:
			pass

	# 推断组件类型
	if 'component' in override:
		component_type = override['component']
		extra_kwargs = {}
		# 如果是 dropdown，尝试从类型推断 choices
		if component_type == 'dropdown':
			_, inferred = infer_component_type(field_type, field_name, default_value)
			extra_kwargs = inferred
	else:
		component_type, extra_kwargs = infer_component_type(field_type, field_name, default_value)

	# 嵌套模型不在这里创建组件
	if component_type == 'nested':
		return None, ('nested', extra_kwargs['model_class'])

	# 准备组件参数
	label = override.get('label', field_name)
	info = override.get('info', extra_kwargs.get('info'))

	component = None

	if component_type == 'textbox':
		component = gr.Textbox(
			label=label,
			value=str(default_value) if default_value not in (None, '') else '',
			info=info,
		)

	elif component_type == 'textarea':
		component = gr.Textbox(
			label=label,
			value=str(default_value) if default_value not in (None, '') else '',
			lines=3,
			info=info,
		)

	elif component_type == 'number':
		num_kwargs = {}
		if 'min_value' in override:
			num_kwargs['minimum'] = override['min_value']
		if 'max_value' in override:
			num_kwargs['maximum'] = override['max_value']
		if 'step' in override:
			num_kwargs['step'] = override['step']
		component = gr.Number(
			label=label,
			value=default_value if default_value is not None else 0,
			info=info,
			**num_kwargs,
		)

	elif component_type == 'slider':
		min_val = override.get('min_value', 0)
		max_val = override.get('max_value', 1)
		step_val = override.get('step', 0.01)
		component = gr.Slider(
			label=label,
			minimum=min_val,
			maximum=max_val,
			step=step_val,
			value=default_value if default_value is not None else min_val,
			info=info,
		)

	elif component_type == 'dropdown':
		choices = override.get('choices', extra_kwargs.get('choices', []))
		allow_custom = override.get('allow_custom', False)
		component = gr.Dropdown(
			label=label,
			choices=choices,
			value=default_value if default_value in choices else (choices[0] if choices else None),
			allow_custom_value=allow_custom,
			info=info,
		)

	elif component_type == 'checkbox':
		component = gr.Checkbox(
			label=label,
			value=bool(default_value) if default_value is not None else False,
			info=info,
		)

	elif component_type == 'file':
		component = gr.Textbox(
			label=label,
			value=str(default_value) if default_value not in (None, '') else '',
			info=info or '输入文件路径',
		)

	return component, component_type


def build_config_ui(
	gr,
	config_class: type[BaseModel],
	overrides: dict[str, dict] | None = None,
	prefix: str = '',
	exclude_fields: set[str] | None = None,
) -> dict[str, Any]:
	"""递归构建配置类的 UI

	返回 {field_path: component} 字典
	"""
	overrides = overrides or {}
	exclude_fields = exclude_fields or {'task_type', 'task_file', 'task_id'}

	components: dict[str, Any] = {}

	# 获取字段信息
	for field_name, field_info in config_class.model_fields.items():
		if field_name in exclude_fields:
			continue

		field_path = f'{prefix}{field_name}' if prefix else field_name
		field_type = field_info.annotation

		# 检查是否是嵌套 BaseModel
		if isinstance(field_type, type) and issubclass(field_type, BaseModel):
			# 递归处理嵌套模型
			nested_components = build_config_ui(
				gr,
				field_type,
				overrides,
				prefix=f'{field_path}.',
				exclude_fields=set(),
			)
			components.update(nested_components)
		else:
			# 创建组件
			override = overrides.get(field_path)
			if field_type is None:
				continue
			comp, _ = create_gradio_component(
				gr, field_name, field_info, field_type, override
			)
			if comp is not None:
				components[field_path] = comp

	return components


def collect_values_to_dict(components: dict[str, Any], values: list[Any]) -> dict[str, Any]:
	"""将 UI 值转换为嵌套字典结构

	例如:
		{'model.name': 'resnet34', 'model.num_classes': 3}
		-> {'model': {'name': 'resnet34', 'num_classes': 3}}
	"""
	result: dict[str, Any] = {}
	field_paths = list(components.keys())

	for i, path in enumerate(field_paths):
		value = values[i]

		# 处理特殊值转换
		if isinstance(value, str):
			# 尝试转换为数字
			if value.replace('.', '').replace('-', '').replace('e', '').isdigit():
				try:
					if '.' in value or 'e' in value.lower():
						value = float(value)
					else:
						value = int(value)
				except ValueError:
					pass
			# 处理列表 (逗号分隔)
			elif ',' in value:
				value = [v.strip() for v in value.split(',') if v.strip()]
			# 空字符串转 None
			elif value == '':
				value = None

		# 按路径设置值
		parts = path.split('.')
		current = result
		for part in parts[:-1]:
			if part not in current:
				current[part] = {}
			current = current[part]
		current[parts[-1]] = value

	return result


def _render_groups(
	gr,
	group_names: list[str],
	groups: dict[str, list[tuple[str, FieldInfo, type | None]]],
	group_labels: dict[str, str],
	overrides: dict[str, dict],
	all_components: dict[str, Any],
) -> None:
	"""渲染一组分组的 UI 组件"""
	for group_name in group_names:
		if group_name not in groups:
			continue
		label = group_labels.get(group_name, group_name)
		gr.Markdown(f'### {label}')
		for field_name, field_info, field_type in groups[group_name]:
			if isinstance(field_type, type) and issubclass(field_type, BaseModel):
				nested = build_config_ui(
					gr, field_type, overrides,
					prefix=f'{field_name}.', exclude_fields=set()
				)
				all_components.update(nested)
			else:
				if field_type is None:
					continue
				override = overrides.get(field_name)
				comp, _ = create_gradio_component(
					gr, field_name, field_info, field_type, override
				)
				if comp:
					all_components[field_name] = comp


def build_task_ui(
	gr,
	task_type: str,
	runner_cls,
	config_class: type[BaseModel],
	on_submit: Any,
) -> tuple[list[Any], Any]:
	"""为任务类型构建完整的 UI

	返回 (components_list, submit_button)
	"""
	# 检查是否有自定义 UI
	custom_ui = runner_cls.build_custom_ui(gr)
	if custom_ui is not None:
		return custom_ui

	# 获取字段覆盖
	overrides = runner_cls.get_ui_field_overrides()

	# 按分组组织字段
	groups: dict[str, list[tuple[str, FieldInfo, type | None]]] = {}
	for field_name, field_info in config_class.model_fields.items():
		if field_name in {'task_type', 'task_file', 'task_id'}:
			continue
		field_type = field_info.annotation
		# 确定分组
		if isinstance(field_type, type) and issubclass(field_type, BaseModel):
			group_name = field_name
		else:
			group_name = '_root'

		if group_name not in groups:
			groups[group_name] = []
		groups[group_name].append((field_name, field_info, field_type))

	# 定义分组显示名称
	group_labels = {
		'meta': '基本信息',
		'dataset': '数据集配置',
		'model': '模型配置',
		'training': '训练配置',
		'output': '输出配置',
		'_root': '其他配置',
	}

	all_components: dict[str, Any] = {}

	# 分组顺序
	group_order = ['meta', 'dataset', 'model', 'training', 'output', '_root']

	with gr.Row():
		# 左列
		with gr.Column():
			_render_groups(gr, group_order[:3], groups, group_labels, overrides, all_components)

		# 右列
		with gr.Column():
			_render_groups(gr, group_order[3:], groups, group_labels, overrides, all_components)

	# 创建提交按钮
	submit_btn = gr.Button('创建任务', variant='primary')
	result_output = gr.Textbox(label='结果', lines=3)

	# 绑定提交回调
	component_list = list(all_components.values())

	def submit_wrapper(*values):
		try:
			config_dict = collect_values_to_dict(all_components, list(values))
			config_dict['task_type'] = task_type
			return on_submit(task_type, config_dict, config_class)
		except Exception as e:
			return f'错误: {e}'

	submit_btn.click(submit_wrapper, inputs=component_list, outputs=result_output)

	return component_list, submit_btn
