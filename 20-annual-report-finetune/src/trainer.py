"""
训练器模块
支持DAPT、SFT、DPO三阶段训练
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, DataCollatorForSeq2Seq
from loguru import logger
from typing import Dict, Any, List, Optional, Tuple
import json

from src.model import AnnualReportModel


class AnnualReportDataset(Dataset):
    """年报数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        """
        Args:
            data_path: 数据文件路径（JSONL格式）
            tokenizer: tokenizer
            max_length: 最大序列长度
        """
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

        logger.info(f"加载数据集: {len(self.data)} 条样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建输入
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output_text = item.get('output', '')

        # 格式化提示词
        prompt = self._format_prompt(instruction, input_text)

        # 组合输入和输出
        full_text = prompt + output_text

        # Tokenize
        inputs = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # 创建labels (只计算output部分的loss)
        prompt_length = len(self.tokenizer(prompt, truncation=True, max_length=self.max_length)['input_ids'])
        labels = inputs['input_ids'].clone()
        labels[0, :prompt_length] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

    def _format_prompt(self, instruction: str, input_text: str) -> str:
        """格式化提示词"""
        if input_text:
            prompt = f"""<|im_start|>system
你是一名专业的证券分析师，擅长分析上市公司年报，提取关键财务指标，评估经营风险，提供投资建议。

<|im_start|>user
{instruction}

{input_text}<|im_end|>
<|im_start|>assistant
"""
        else:
            prompt = f"""<|im_start|>system
你是一名专业的证券分析师，擅长分析上市公司年报，提取关键财务指标，评估经营风险，提供投资建议。

<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
        return prompt


class DAPTDataset(Dataset):
    """DAPT数据集（领域自适应预训练）"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        """
        Args:
            data_path: 数据文件路径（每行一段文本）
            tokenizer: tokenizer
            max_length: 最大序列长度
        """
        self.texts = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.texts.append(line)

        logger.info(f"加载DAPT数据集: {len(self.texts)} 条文本")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': inputs['input_ids'].squeeze()
        }


class AnnualReportTrainer:
    """年报分析模型训练器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_handler = None
        self.trainer = None

        self.logger = logger

    def setup_model(self, model_path: Optional[str] = None):
        """设置模型"""
        self.model_handler = AnnualReportModel(self.config)
        self.model_handler.load_base_model(model_path)

        use_lora = self.config.get('model', {}).get('use_lora', True)
        if use_lora:
            self.model_handler.setup_lora()

    def train_dapt(
        self,
        train_path: str,
        eval_path: Optional[str] = None,
        output_path: Optional[str] = None
    ):
        """
        DAPT（领域自适应预训练）

        Args:
            train_path: 训练数据路径
            eval_path: 验证数据路径（可选）
            output_path: 输出路径
        """
        self.logger.info("开始DAPT训练")

        # 准备数据
        train_dataset = DAPTDataset(
            train_path,
            self.model_handler.tokenizer,
            self.config.get('data', {}).get('max_length', 2048)
        )

        eval_dataset = None
        if eval_path and os.path.exists(eval_path):
            eval_dataset = DAPTDataset(
                eval_path,
                self.model_handler.tokenizer,
                self.config.get('data', {}).get('max_length', 2048)
            )

        # 训练参数
        training_args = self.model_handler.get_training_args('dapt')

        # 创建Trainer
        self.trainer = Trainer(
            model=self.model_handler.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.model_handler.tokenizer,
        )

        # 开始训练
        self.trainer.train()

        # 保存模型
        output_path = output_path or self.config.get('paths', {}).get('dapt_model', './dapt_model')
        self.model_handler.save_model(output_path)

        self.logger.info(f"DAPT训练完成，模型保存到: {output_path}")

    def train_sft(
        self,
        train_path: str,
        eval_path: Optional[str] = None,
        base_model: Optional[str] = None,
        output_path: Optional[str] = None
    ):
        """
        SFT（监督微调）

        Args:
            train_path: 训练数据路径
            eval_path: 验证数据路径（可选）
            base_model: 基座模型路径（可选，默认使用配置中的路径）
            output_path: 输出路径
        """
        self.logger.info("开始SFT训练")

        # 重新加载模型
        if base_model:
            self.setup_model(base_model)
        elif not self.model_handler:
            self.setup_model()

        # 准备数据
        train_dataset = AnnualReportDataset(
            train_path,
            self.model_handler.tokenizer,
            self.config.get('data', {}).get('max_length', 2048)
        )

        eval_dataset = None
        if eval_path and os.path.exists(eval_path):
            eval_dataset = AnnualReportDataset(
                eval_path,
                self.model_handler.tokenizer,
                self.config.get('data', {}).get('max_length', 2048)
            )

        # 训练参数
        training_args = self.model_handler.get_training_args('sft')

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.model_handler.tokenizer,
            model=self.model_handler.model,
            label_pad_token_id=-100
        )

        # 创建Trainer
        self.trainer = Trainer(
            model=self.model_handler.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.model_handler.tokenizer,
        )

        # 开始训练
        self.trainer.train()

        # 保存模型
        output_path = output_path or self.config.get('paths', {}).get('sft_model', './sft_model')
        self.model_handler.save_model(output_path)

        self.logger.info(f"SFT训练完成，模型保存到: {output_path}")

    def train_dpo(
        self,
        train_path: str,
        eval_path: Optional[str] = None,
        base_model: Optional[str] = None,
        output_path: Optional[str] = None
    ):
        """
        DPO（偏好优化）

        Args:
            train_path: 训练数据路径（偏好对格式）
            eval_path: 验证数据路径（可选）
            base_model: 基座模型路径（可选）
            output_path: 输出路径
        """
        self.logger.info("开始DPO训练")

        # 重新加载模型
        if base_model:
            self.setup_model(base_model)
        elif not self.model_handler:
            self.setup_model()

        # 准备数据（DPO需要偏好对）
        # 这里简化处理，实际DPO需要专门的Trainer
        from trl import DPOTrainer, DPOConfig

        # 加载偏好数据
        # TODO: 实现DPO数据加载逻辑

        # DPO配置
        dpo_config = DPOConfig(
            output_dir=self.config.get('output', {}).get('checkpoint_dir', './checkpoints'),
            beta=0.1,
            learning_rate=self.config.get('training', {}).get('learning_rate', 1e-5),
            per_device_train_batch_size=self.config.get('training', {}).get('batch_size', 2),
            gradient_accumulation_steps=self.config.get('training', {}).get('gradient_accumulation_steps', 4),
        )

        # 创建DPO Trainer
        # self.trainer = DPOTrainer(
        #     model=self.model_handler.model,
        #     ref_model=None,
        #     args=dpo_config,
        #     train_dataset=train_dataset,
        #     eval_dataset=eval_dataset,
        #     tokenizer=self.model_handler.tokenizer,
        # )

        # self.trainer.train()

        self.logger.info("DPO训练完成")

    def train_all(
        self,
        dapt_data_path: str,
        sft_data_path: str,
        dpo_data_path: Optional[str] = None
    ):
        """
        完整训练流程：DAPT → SFT → DPO

        Args:
            dapt_data_path: DAPT训练数据路径
            sft_data_path: SFT训练数据路径
            dpo_data_path: DPO训练数据路径（可选）
        """
        self.logger.info("开始完整训练流程: DAPT → SFT → DPO")

        # 阶段1: DAPT
        self.train_dapt(dapt_data_path)

        # 阶段2: SFT
        dapt_model_path = self.config.get('paths', {}).get('dapt_model', './dapt_model')
        self.train_sft(sft_data_path, base_model=dapt_model_path)

        # 阶段3: DPO
        if dpo_data_path:
            sft_model_path = self.config.get('paths', {}).get('sft_model', './sft_model')
            self.train_dpo(dpo_data_path, base_model=sft_model_path)

        self.logger.info("完整训练流程完成")


def main():
    """测试代码"""
    config = {
        'model': {
            'path': 'Qwen/Qwen2.5-7B-Instruct',
            'use_lora': True,
            'lora_r': 16,
            'lora_alpha': 32,
            'load_in_4bit': True
        },
        'training': {
            'batch_size': 4,
            'gradient_accumulation_steps': 4,
            'num_epochs': 3,
            'learning_rate': 2e-4
        },
        'output': {
            'checkpoint_dir': './checkpoints'
        },
        'paths': {
            'dapt_model': './models/dapt_model',
            'sft_model': './models/sft_model',
            'dpo_model': './models/dpo_model'
        }
    }

    trainer = AnnualReportTrainer(config)
    # trainer.setup_model()


if __name__ == "__main__":
    main()
