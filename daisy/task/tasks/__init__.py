"""任务类型模块

每个任务类型作为一个子模块实现，包含:
- config.py: 任务配置类
- runner.py: 任务执行器类

导入此模块会自动注册所有任务类型。
"""

# 导入各任务模块以触发注册
from . import classification
from . import mae_pretrain
from . import mae_finetune
