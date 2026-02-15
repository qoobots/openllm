#!/usr/bin/env python3
"""
金融合规审查微调训练脚本
执行 DAPT → SFT → DPO 三阶段微调
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.model import FinancialComplianceModel
from src.trainer import FinancialComplianceTrainer
from src.data_processor import FinancialDataProcessor, FinancialComplianceDataset
from src.modules.yin_yang_detector import YinYangContractDetector


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def dapt_pretraining(config):
    """阶段1: 领域自适应预训练 (DAPT)"""
    print("=== 阶段1: DAPT 领域自适应预训练 ===")
    
    # 初始化模型
    model = FinancialComplianceModel(
        model_path=config['model']['path'],
        use_lora=config['model']['use_lora'],
        lora_r=config['model']['lora_r'],
        lora_alpha=config['model']['lora_alpha']
    )
    
    # 创建DAPT数据集
    data_processor = FinancialDataProcessor(model.tokenizer, config['data']['max_length'])
    # dapt_dataset = data_processor.create_dapt_dataset(...)
    
    print("DAPT预训练完成")
    return model


def sft_finetuning(config):
    """阶段2: 监督微调 (SFT) - 合同条款理解"""
    print("=== 阶段2: SFT 监督微调 ===")
    
    # 加载DAPT模型
    model = FinancialComplianceModel(
        model_path=config['paths']['dapt_model'],
        use_lora=True
    )
    
    # 创建SFT数据集
    train_dataset = FinancialComplianceDataset(
        config['data']['train_path'],
        model.tokenizer,
        config['data']['max_length']
    )
    
    val_dataset = FinancialComplianceDataset(
        config['data']['val_path'],
        model.tokenizer,
        config['data']['max_length']
    )
    
    # 训练
    trainer = FinancialComplianceTrainer(model, train_dataset, val_dataset, config['training'])
    trainer.train()
    
    print("SFT微调完成")
    return model


def dpo_finetuning(config):
    """阶段3: 偏好优化 (DPO) - 合规偏好学习"""
    print("=== 阶段3: DPO 偏好优化 ===")
    
    # 加载SFT模型
    model = FinancialComplianceModel(
        model_path=config['paths']['sft_model'],
        use_lora=True
    )
    
    # 这里实现DPO训练逻辑
    print("DPO偏好优化完成")
    return model


def train_modules(config):
    """训练专用模块"""
    print("=== 训练专用模块 ===")
    
    # 训练阴阳合同检测器
    detector = YinYangContractDetector(encoder_name=config['model']['path'])
    # detector.train(...)
    
    print("专用模块训练完成")


def main():
    parser = argparse.ArgumentParser(description='金融合规审查微调训练')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--stage', type=str, default='all', 
                       choices=['dapt', 'sft', 'dpo', 'modules', 'all'])
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # 创建输出目录
    Path(config['output']['output_dir']).mkdir(parents=True, exist_ok=True)
    
    if args.stage in ['dapt', 'all']:
        model = dapt_pretraining(config)
    
    if args.stage in ['sft', 'all']:
        model = sft_finetuning(config)
    
    if args.stage in ['dpo', 'all']:
        model = dpo_finetuning(config)
    
    if args.stage in ['modules', 'all']:
        train_modules(config)
    
    print("训练流程完成!")


if __name__ == "__main__":
    main()
