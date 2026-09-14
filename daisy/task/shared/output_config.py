"""共享的输出配置"""

from ..base import BaseConfig


class LogOutputConfig(BaseConfig):
	"""包含日志开关的输出配置"""

	save_path: str = 'outputs/{task_id}'
	log: bool = True
