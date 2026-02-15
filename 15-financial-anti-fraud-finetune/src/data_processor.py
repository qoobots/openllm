import json
import torch
from torch.utils.data import Dataset


class FinancialAntiFraudDataset(Dataset):
    """FinancialAntiFraud数据集"""
    
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
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }
    
    def _format_prompt(self, item):
        """格式化提示词"""
        return f"Task: {item.get('input', 'Unknown')}\nOutput: "
