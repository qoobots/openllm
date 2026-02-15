#!/usr/bin/env python3
"""
ManufacturingPrediction训练脚本
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.model import ManufacturingPredictionModel
from src.trainer import ManufacturingPredictionTrainer
from src.data_processor import ManufacturingPredictionDataset


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='ManufacturingPrediction微调训练')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # 初始化模型
    model = ManufacturingPredictionModel(
        model_path=config['model']['path'],
        use_lora=config['model']['use_lora']
    )
    
    # 创建数据集
    train_dataset = ManufacturingPredictionDataset(
        config['data']['train_path'],
        model.tokenizer,
        config['data']['max_length']
    )
    
    val_dataset = ManufacturingPredictionDataset(
        config['data']['val_path'],
        model.tokenizer,
        config['data']['max_length']
    )
    
    # 创建训练器
    trainer = ManufacturingPredictionTrainer(model, train_dataset, val_dataset, config['training'])
    
    # 训练
    trainer.train()
    
    print("训练完成!")


if __name__ == "__main__":
    main()
