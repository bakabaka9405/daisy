"""Daisy CLI 入口

用法:
    python -m daisy run <task.toml>     # 运行任务
    python -m daisy new [--type TYPE]   # 交互式创建任务
    python -m daisy list                 # 列出所有任务
    python -m daisy ui                   # 启动 Web UI
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path


def cmd_run(args):
    """运行任务"""
    from daisy.task import run_task

    task_file = Path(args.task)
    if not task_file.exists():
        print(f"Error: Task file not found: {task_file}")
        sys.exit(1)

    run_task(task_file, device=args.device)


def cmd_new(args):
    """交互式创建新任务"""
    from daisy.task import TaskRegistry
    from daisy.task.tasks.classification.config import (
        ClassificationConfig,
        DatasetConfig,
        DatasetSplitConfig,
        ModelConfig,
        TrainingConfig,
        TransformConfig,
    )
    from daisy.task.base import BaseMetaConfig, BaseOutputConfig
    import tomli_w

    task_type = args.type

    # 检查任务类型是否已注册
    if not TaskRegistry.is_registered(task_type):
        available = TaskRegistry.list_task_types()
        print(f"Error: Unknown task type '{task_type}'.")
        print(f"Available types: {available}")
        sys.exit(1)

    print(f"创建新 {task_type} 任务")
    print("=" * 40)

    # 元信息
    title = input("任务标题: ").strip()
    description = input("任务描述: ").strip()
    creator = input("创建人: ").strip()

    # 生成任务 ID
    today = datetime.now().strftime('%Y%m%d')
    tasks_dir = Path(args.output_dir)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    existing = list(tasks_dir.glob(f"{today}-*.toml"))
    task_num = len(existing) + 1
    task_id = f"{today}-{task_num}"

    if title:
        task_id += f"-{title.replace(' ', '_')[:20]}"

    # 根据任务类型创建配置
    if task_type == 'classification':
        config = _create_classification_config(
            title, description, creator, task_id,
            ClassificationConfig, DatasetConfig, DatasetSplitConfig,
            ModelConfig, TrainingConfig, TransformConfig,
            BaseMetaConfig, BaseOutputConfig,
        )
    else:
        # 对于其他任务类型，使用通用流程
        print(f"\n任务类型 '{task_type}' 暂不支持交互式创建。")
        print("请手动创建 TOML 配置文件。")
        sys.exit(1)

    # 保存配置
    output_file = tasks_dir / f"{task_id}.toml"
    data = config.model_dump(exclude={'task_file', 'task_id'}, exclude_none=True)

    with open(output_file, 'wb') as f:
        tomli_w.dump(data, f)

    print(f"\n任务配置已保存到: {output_file}")
    print(f"运行任务: python -m daisy run {output_file}")


def _create_classification_config(
    title, description, creator, task_id,
    ClassificationConfig, DatasetConfig, DatasetSplitConfig,
    ModelConfig, TrainingConfig, TransformConfig,
    BaseMetaConfig, BaseOutputConfig,
):
    """交互式创建分类任务配置"""
    # 数据集配置
    print("\n数据集配置:")
    dataset_type = input("数据集类型 (sheet/folder) [sheet]: ").strip() or "sheet"
    root = input("数据根目录: ").strip()

    dataset_cfg = DatasetConfig(type=dataset_type, root=root)  # type: ignore

    if dataset_type == "sheet":
        dataset_cfg.sheet = input("标注文件路径: ").strip()
        dataset_cfg.column = int(input("标签列 [1]: ").strip() or "1")
        dataset_cfg.label_offset = int(input("标签偏移 [0]: ").strip() or "0")

    val_ratio = float(input("验证集比例 [0.1]: ").strip() or "0.1")
    dataset_cfg.split = DatasetSplitConfig(method="ratio", val_ratio=val_ratio)

    # 模型配置
    print("\n模型配置:")
    model_name = input("模型名称 [resnet34]: ").strip() or "resnet34"
    num_classes = int(input("类别数 [2]: ").strip() or "2")
    pretrained = input("使用预训练权重 (y/n) [y]: ").strip().lower() != 'n'

    model_cfg = ModelConfig(name=model_name, num_classes=num_classes, pretrained=pretrained)

    # 训练配置
    print("\n训练配置:")
    epochs = int(input("训练轮数 [30]: ").strip() or "30")
    batch_size = int(input("Batch size [64]: ").strip() or "64")
    lr = float(input("学习率 [1e-3]: ").strip() or "1e-3")

    training_cfg = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        transform=TransformConfig(),
    )

    # 输出配置
    output_cfg = BaseOutputConfig(save_path=f"outputs/{task_id}")

    # 组装配置
    return ClassificationConfig(
        task_type='classification',
        meta=BaseMetaConfig(
            title=title,
            description=description,
            creator=creator,
            created_at=datetime.now().strftime('%Y-%m-%d'),
        ),
        dataset=dataset_cfg,
        model=model_cfg,
        training=training_cfg,
        output=output_cfg,
    )


def cmd_list(args):
    """列出所有任务"""
    from daisy.task import load_config

    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.exists():
        print(f"任务目录不存在: {tasks_dir}")
        return

    task_files = sorted(tasks_dir.glob("*.toml"), reverse=True)

    if not task_files:
        print("没有找到任务文件")
        return

    print(f"{'Task ID':<30} {'Type':<15} {'Title':<25} {'Created':<12}")
    print("-" * 82)

    for f in task_files:
        try:
            cfg = load_config(f)
            print(f"{cfg.task_id:<30} {cfg.task_type:<15} {cfg.meta.title[:23]:<25} {cfg.meta.created_at:<12}")
        except Exception as e:
            print(f"{f.stem:<30} [Error: {e}]")


def cmd_ui(args):
    """启动 Web UI"""
    try:
        from daisy.task.ui import launch_ui
        launch_ui(args.port)
    except ImportError:
        print("UI 模块需要安装 gradio: pip install gradio")
        sys.exit(1)


def main():
    from daisy.task import TaskRegistry

    parser = argparse.ArgumentParser(
        description='Daisy - 深度学习训练任务管理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # run 命令
    run_parser = subparsers.add_parser('run', help='运行任务')
    run_parser.add_argument('task', help='任务配置文件 (.toml)')
    run_parser.add_argument('--device', '-d', help='运行设备 (cuda/cpu)')

    # new 命令
    available_types = TaskRegistry.list_task_types() or ['classification']
    new_parser = subparsers.add_parser('new', help='创建新任务')
    new_parser.add_argument('--output-dir', '-o', default='tasks', help='任务保存目录')
    new_parser.add_argument(
        '--type', '-t',
        default='classification',
        choices=available_types,
        help=f'任务类型 (可选: {available_types})'
    )

    # list 命令
    list_parser = subparsers.add_parser('list', help='列出所有任务')
    list_parser.add_argument('--tasks-dir', '-d', default='tasks', help='任务目录')

    # ui 命令
    ui_parser = subparsers.add_parser('ui', help='启动 Web UI')
    ui_parser.add_argument('--port', '-p', type=int, default=7860, help='端口号')

    args = parser.parse_args()

    if args.command == 'run':
        cmd_run(args)
    elif args.command == 'new':
        cmd_new(args)
    elif args.command == 'list':
        cmd_list(args)
    elif args.command == 'ui':
        cmd_ui(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
