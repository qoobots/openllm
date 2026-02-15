#!/usr/bin/env python3
"""
批量生成缺失的文件脚本
"""

import os
from pathlib import Path


# 项目列表
PROJECTS = [
    "13-legal-contract-finetune",
    "14-medical-insurance-finetune",
    "15-financial-anti-fraud-finetune",
    "16-legal-multilang-finetune",
    "17-manufacturing-prediction-finetune",
    "18-medical-primarycare-finetune",
    "19-financial-fairness-finetune",
]

BASE_DIR = Path("d:/05workspaces/openllm")


# trainer.py 模板
TRAINER_TEMPLATE = '''import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb


class {ClassName}Trainer:
    """{ClassName}微调训练器"""
    
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model.model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        num_training_steps = len(self.train_loader) * config['num_epochs']
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.get('use_fp16', True))
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {{epoch}}")
        
        for step, batch in enumerate(progress_bar):
            batch = {{k: v.to(self.device) for k, v in batch.items()}}
            
            with torch.cuda.amp.autocast(enabled=self.config.get('use_fp16', True)):
                outputs = self.model(**batch)
                loss = outputs.loss
            
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({{'loss': loss.item()}})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self, epoch):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            batch = {{k: v.to(self.device) for k, v in batch.items()}}
            
            with torch.cuda.amp.autocast(enabled=self.config.get('use_fp16', True)):
                outputs = self.model(**batch)
                loss = outputs.loss
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self):
        """本地训练"""
        wandb.init(project="{project_name}", config=self.config)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate(epoch)
            
            print(f"Epoch {{epoch}}: Train Loss = {{train_loss:.4f}}, Val Loss = {{val_loss:.4f}}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_pretrained(self.config['output_dir'])
                print(f"Saved best model with val_loss: {{val_loss:.4f}}")
        
        wandb.finish()
'''

# data_processor.py 模板
DATA_PROCESSOR_TEMPLATE = '''import json
import torch
from torch.utils.data import Dataset


class {ClassName}Dataset(Dataset):
    """{ClassName}数据集"""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入
        prompt = self._format_prompt(item)
        response = item.get('response', '')
        
        # Tokenize
        inputs = self.tokenizer(
            prompt + response,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 创建labels
        labels = inputs['input_ids'].clone()
        prompt_length = len(self.tokenizer(prompt)['input_ids'])
        labels[0, :prompt_length] = -100
        
        return {{
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }}
    
    def _format_prompt(self, item):
        """格式化提示词"""
        return f"Task: {{item.get('input', 'Unknown')}}\\nOutput: "
'''

# train.py 模板
TRAIN_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""
{ClassName}训练脚本
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.model import {ClassName}Model
from src.trainer import {ClassName}Trainer
from src.data_processor import {ClassName}Dataset


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='{ClassName}微调训练')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # 初始化模型
    model = {ClassName}Model(
        model_path=config['model']['path'],
        use_lora=config['model']['use_lora']
    )
    
    # 创建数据集
    train_dataset = {ClassName}Dataset(
        config['data']['train_path'],
        model.tokenizer,
        config['data']['max_length']
    )
    
    val_dataset = {ClassName}Dataset(
        config['data']['val_path'],
        model.tokenizer,
        config['data']['max_length']
    )
    
    # 创建训练器
    trainer = {ClassName}Trainer(model, train_dataset, val_dataset, config['training'])
    
    # 训练
    trainer.train()
    
    print("训练完成!")


if __name__ == "__main__":
    main()
'''

# infer.py 模板
INFER_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""
{ClassName}推理脚本
"""

import argparse
import yaml

from src.model import {ClassName}Model


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='{ClassName}推理')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.txt')
    
    args = parser.parse_args()
    
    # 加载模型
    model = {ClassName}Model(args.model_path, use_lora=True)
    
    # 生成结果
    result = model.generate(args.input, max_length=512, temperature=0.3)
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"推理结果已保存到 {{args.output}}")
    print("\\n结果：")
    print(result)


if __name__ == "__main__":
    main()
'''

# __init__.py 模板
INIT_TEMPLATE = '''from .model import {ClassName}Model
from .trainer import {ClassName}Trainer
from .data_processor import {ClassName}Dataset

__all__ = ['{ClassName}Model', '{ClassName}Trainer', '{ClassName}Dataset']
'''

# 项目配置
PROJECT_CONFIG = {
    "13-legal-contract-finetune": {
        "class_name": "LegalContract",
        "project_name": "legal-contract-finetune"
    },
    "14-medical-insurance-finetune": {
        "class_name": "MedicalInsurance",
        "project_name": "medical-insurance-finetune"
    },
    "15-financial-anti-fraud-finetune": {
        "class_name": "FinancialAntiFraud",
        "project_name": "financial-anti-fraud-finetune"
    },
    "16-legal-multilang-finetune": {
        "class_name": "LegalMultiLang",
        "project_name": "legal-multilang-finetune"
    },
    "17-manufacturing-prediction-finetune": {
        "class_name": "ManufacturingPrediction",
        "project_name": "manufacturing-prediction-finetune"
    },
    "18-medical-primarycare-finetune": {
        "class_name": "MedicalPrimaryCare",
        "project_name": "medical-primarycare-finetune"
    },
    "19-financial-fairness-finetune": {
        "class_name": "FinancialFairness",
        "project_name": "financial-fairness-finetune"
    },
}


def generate_file(project_path, relative_path, content):
    """生成文件"""
    full_path = project_path / relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 生成: {full_path}")


def main():
    print("开始生成缺失的文件...")
    
    for project in PROJECTS:
        project_path = BASE_DIR / project
        config = PROJECT_CONFIG[project]
        class_name = config['class_name']
        project_name = config['project_name']
        
        print(f"\n处理项目: {project}")
        
        # 生成 src/trainer.py
        trainer_content = TRAINER_TEMPLATE.format(
            ClassName=class_name,
            project_name=project_name
        )
        generate_file(project_path, "src/trainer.py", trainer_content)
        
        # 生成 src/data_processor.py
        data_processor_content = DATA_PROCESSOR_TEMPLATE.format(ClassName=class_name)
        generate_file(project_path, "src/data_processor.py", data_processor_content)
        
        # 生成 scripts/train.py
        train_content = TRAIN_SCRIPT_TEMPLATE.format(ClassName=class_name)
        generate_file(project_path, "scripts/train.py", train_content)
        
        # 生成 scripts/infer.py
        infer_content = INFER_SCRIPT_TEMPLATE.format(ClassName=class_name)
        generate_file(project_path, "scripts/infer.py", infer_content)
        
        # 生成 src/__init__.py
        init_content = INIT_TEMPLATE.format(ClassName=class_name)
        generate_file(project_path, "src/__init__.py", init_content)
    
    print("\n✅ 所有文件生成完成!")
    print(f"共处理 {len(PROJECTS)} 个项目")
    print(f"每个项目生成了 5 个文件")
    print(f"总计生成 {len(PROJECTS) * 5} 个文件")


if __name__ == "__main__":
    main()
