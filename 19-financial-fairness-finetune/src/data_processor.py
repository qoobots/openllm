import json
import torch
from torch.utils.data import Dataset


class FinancialFairnessDataset(Dataset):
    """金融公平性数据集"""
    
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
        
        prompt = f"""公平性评估任务：
申请类型：{item.get('application_type', '贷款申请')}
申请人信息：{item.get('applicant_info', '')}
敏感属性：{item.get('sensitive_attributes', '')}

请评估决策的公平性：

公平性评估："""
        
        response = item.get('fairness_analysis', '')
        
        inputs = self.tokenizer(
            prompt + response,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = inputs['input_ids'].clone()
        prompt_length = len(self.tokenizer(prompt)['input_ids'])
        labels[0, :prompt_length] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }
