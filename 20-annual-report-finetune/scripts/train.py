"""
训练脚本
支持DAPT、SFT、DPO三种训练模式
"""

import argparse
import yaml
from pathlib import Path
from loguru import logger
import sys

# 添加src到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.trainer import AnnualReportTrainer


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_dapt(trainer, config):
    """执行DAPT训练"""
    dapt_data_path = config.get('data', {}).get('dapt_data_path')
    if not dapt_data_path or not Path(dapt_data_path).exists():
        logger.error(f"DAPT数据文件不存在: {dapt_data_path}")
        return

    logger.info("=== 开始DAPT训练 ===")
    trainer.train_dapt(dapt_data_path)


def train_sft(trainer, config):
    """执行SFT训练"""
    train_path = config.get('data', {}).get('train_path')
    val_path = config.get('data', {}).get('val_path')

    if not train_path or not Path(train_path).exists():
        logger.error(f"SFT训练数据文件不存在: {train_path}")
        return

    logger.info("=== 开始SFT训练 ===")
    trainer.train_sft(train_path, val_path)


def train_dpo(trainer, config):
    """执行DPO训练"""
    dpo_data_path = config.get('data', {}).get('dpo_data_path')
    if not dpo_data_path or not Path(dpo_data_path).exists():
        logger.error(f"DPO数据文件不存在: {dpo_data_path}")
        return

    logger.info("=== 开始DPO训练 ===")
    trainer.train_dpo(dpo_data_path)


def main():
    parser = argparse.ArgumentParser(description="年报分析模型训练脚本")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--stage',
        type=str,
        choices=['dapt', 'sft', 'dpo', 'all'],
        default='all',
        help='训练阶段'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='基座模型路径（可选，覆盖配置文件）'
    )

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    logger.info(f"配置文件: {args.config}")

    # 创建训练器
    trainer = AnnualReportTrainer(config)

    # 设置模型
    model_path = args.model_path or config.get('model', {}).get('path')
    trainer.setup_model(model_path)

    # 根据stage执行训练
    if args.stage == 'dapt':
        train_dapt(trainer, config)
    elif args.stage == 'sft':
        train_sft(trainer, config)
    elif args.stage == 'dpo':
        train_dpo(trainer, config)
    elif args.stage == 'all':
        # 完整训练流程
        train_dapt(trainer, config)
        train_sft(trainer, config)
        train_dpo(trainer, config)

    logger.info("=== 训练完成 ===")


if __name__ == "__main__":
    main()
