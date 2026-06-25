"""
模型定义模块
支持PEFT微调（LoRA/QLoRA）
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from loguru import logger
from typing import Optional, Dict, Any


class AnnualReportModel:
    """年报分析模型"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_config = None

        self.logger = logger

    def load_base_model(self, model_path: Optional[str] = None):
        """
        加载基座模型

        Args:
            model_path: 模型路径，默认从配置中读取
        """
        model_path = model_path or self.config.get('model', {}).get('path')
        if not model_path:
            raise ValueError("模型路径未配置")

        self.logger.info(f"加载基座模型: {model_path}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        use_lora = self.config.get('model', {}).get('use_lora', True)

        if use_lora:
            # QLoRA配置（节省显存）
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=self.config.get('model', {}).get('load_in_8bit', False),
                load_in_4bit=self.config.get('model', {}).get('load_in_4bit', False),
                trust_remote_code=True
            )

            # 准备模型进行PEFT训练
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            # 全量微调
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

        self.logger.info(f"模型加载完成，参数量: {self.model.num_parameters() / 1e9:.2f}B")

    def setup_lora(self):
        """配置LoRA"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get('model', {}).get('lora_r', 16),
            lora_alpha=self.config.get('model', {}).get('lora_alpha', 32),
            lora_dropout=self.config.get('model', {}).get('lora_dropout', 0.05),
            target_modules=self.config.get('model', {}).get(
                'target_modules',
                ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            ),
            bias="none"
        )

        self.peft_config = lora_config
        self.model = get_peft_model(self.model, lora_config)

        # 打印可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())

        self.logger.info(
            f"LoRA配置: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
            f"可训练参数: {trainable_params / 1e6:.2f}M / {all_params / 1e6:.2f}M "
            f"({trainable_params / all_params * 100:.2f}%)"
        )

    def load_peft_model(self, model_path: str, base_model_path: Optional[str] = None):
        """
        加载PEFT微调后的模型

        Args:
            model_path: PEFT模型路径
            base_model_path: 基座模型路径
        """
        from peft import PeftModel

        base_model_path = base_model_path or self.config.get('model', {}).get('path')
        if not base_model_path:
            raise ValueError("基座模型路径未配置")

        self.logger.info(f"加载PEFT模型: {model_path}")
        self.logger.info(f"基座模型: {base_model_path}")

        # 加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # 加载PEFT适配器
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model = self.model.merge_and_unload()

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.logger.info("PEFT模型加载完成")

    def save_model(self, output_path: str, save_base: bool = False):
        """
        保存模型

        Args:
            output_path: 输出路径
            save_base: 是否保存基座模型（PEFT模式下只保存适配器）
        """
        self.logger.info(f"保存模型到: {output_path}")

        if save_base:
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
        else:
            # PEFT模式：只保存适配器
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(output_path)
            else:
                # 如果是merge后的模型，保存全部
                self.model.save_pretrained(output_path)

            self.tokenizer.save_pretrained(output_path)

        self.logger.info("模型保存完成")

    def get_training_args(self, stage: str = "sft") -> TrainingArguments:
        """
        获取训练参数

        Args:
            stage: 训练阶段 (dapt, sft, dpo)

        Returns:
            TrainingArguments: 训练参数
        """
        training_config = self.config.get('training', {})

        # 不同阶段有不同的默认参数
        stage_defaults = {
            'dapt': {
                'num_train_epochs': 1,
                'per_device_train_batch_size': 4,
                'learning_rate': 5e-5,
                'max_grad_norm': 1.0
            },
            'sft': {
                'num_train_epochs': 3,
                'per_device_train_batch_size': 4,
                'learning_rate': 2e-4,
                'max_grad_norm': 1.0
            },
            'dpo': {
                'num_train_epochs': 2,
                'per_device_train_batch_size': 2,
                'learning_rate': 1e-5,
                'max_grad_norm': 1.0
            }
        }

        defaults = stage_defaults.get(stage, stage_defaults['sft'])

        args = TrainingArguments(
            output_dir=self.config.get('output', {}).get('checkpoint_dir', './checkpoints'),
            num_train_epochs=training_config.get('num_epochs', defaults['num_train_epochs']),
            per_device_train_batch_size=training_config.get('batch_size', defaults['per_device_train_batch_size']),
            per_device_eval_batch_size=training_config.get('batch_size', defaults['per_device_train_batch_size']),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
            learning_rate=training_config.get('learning_rate', defaults['learning_rate']),
            weight_decay=training_config.get('weight_decay', 0.01),
            warmup_steps=training_config.get('warmup_steps', 100),
            logging_steps=training_config.get('logging_steps', 10),
            eval_steps=training_config.get('eval_steps', 500),
            save_steps=training_config.get('save_steps', 1000),
            fp16=training_config.get('use_fp16', True),
            bf16=training_config.get('use_bf16', False),
            max_grad_norm=training_config.get('max_grad_norm', defaults['max_grad_norm']),
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            report_to=["tensorboard"],
        )

        return args


def main():
    """测试代码"""
    # 测试配置
    config = {
        'model': {
            'path': 'Qwen/Qwen2.5-7B-Instruct',
            'use_lora': True,
            'lora_r': 16,
            'lora_alpha': 32,
            'load_in_8bit': False,
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
        }
    }

    model = AnnualReportModel(config)
    model.load_base_model()
    model.setup_lora()


if __name__ == "__main__":
    main()
