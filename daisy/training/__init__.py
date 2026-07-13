"""组件化训练接口。

一次性张量变换可直接注册为 hook：

    trainer = Trainer(model, optimizer, nn.MSELoss(), device)
    trainer.after_forward(lambda s, out: out.squeeze(-1))

回归示例：

    from daisy.training import Trainer
    from daisy.training.plugins import CosineAnnealingLR, Eval, BestModel, EpochPrint, CSVLog

    evaluator = Eval(eval_fn=lambda m, d: evaluate_regression(m, val_loader, d))
    trainer = (
        Trainer(model, AdamW(model.parameters(), lr=1e-3), nn.MSELoss(), device)
        .after_forward(lambda s, out: out.squeeze(-1))
        .use(
            CosineAnnealingLR(lr=1e-3, warmup_epochs=5),
            evaluator,
            BestModel(evaluator, watch_metric='pc', save_path='./output'),
            EpochPrint(evaluator, metric_names=['pc', 'mae', 'rmse']),
            CSVLog('./output/history.csv', evaluator),
        )
    )
    state = trainer.fit(train_loader, epochs=50)

分类与 Mixup 示例：

    from daisy.training.plugins import Mixup, EarlyStop

    evaluator = Eval(eval_fn=lambda m, d: evaluate_classifier(m, val_loader, d))
    trainer = (
        Trainer(model, optimizer, SoftTargetCrossEntropy(), device)
        .use(
            CosineAnnealingLR(lr=lr, warmup_epochs=5, per_iteration=True),
            Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=10),
            evaluator,
            BestModel(evaluator, watch_metric='f1', save_path='./output'),
            EarlyStop(evaluator, patience=10, watch_metric='f1'),
            EpochPrint(evaluator),
            CSVLog('./output/history.csv', evaluator),
        )
    )
    state = trainer.fit(train_loader, epochs=100)
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
    Mixup,
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
    'Mixup',
]
