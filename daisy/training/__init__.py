"""组件化训练接口。

训练与预测逻辑通过闭包注入，闭包持有 model、criterion 与输入/输出变换：

    import torch.nn.functional as F
    from daisy.training import Trainer

    def process(batch):
        images, targets = batch
        return F.cross_entropy(model(images), targets)

    def predict(batch):
        images, _ = batch
        return model(images)

    trainer = Trainer(model, optimizer, process, device)
    state = trainer.fit(train_loader, epochs=50)
    scores = trainer.inference(val_loader, predict)

Mixup/CutMix 等变换可直接在闭包内使用 ``timm.data.mixup.Mixup`` 组合。
"""

from daisy.training.plugins import (
    BatchPrint,
    BestModel,
    Checkpoint,
    CosineAnnealingLR,
    CSVLog,
    EarlyStop,
    EpochPrint,
    Eval,
    Plugin,
    Timer,
)
from daisy.training.state import TrainState
from daisy.training.trainer import Trainer

__all__ = [
    'Trainer',
    'TrainState',
    'Plugin',
    'CosineAnnealingLR',
    'Eval',
    'BestModel',
    'EarlyStop',
    'Checkpoint',
    'EpochPrint',
    'BatchPrint',
    'CSVLog',
    'Timer',
]
