import json
import torch
from torch.utils.data import Dataset


class LegalContractDataset(Dataset):
    """法律合同数据集"""
    
    def __init__(self, data_path, tokenizer, max_length=1024):
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
        
        prompt = f"""合同审查任务：
合同类型：{item.get('contract_type', '')}
合同方：{item.get('parties', '')}
法律体系：{item.get('jurisdiction', '')}

合同内容：
{item.get('contract_content', '')}

请审查合规性并识别风险：

合规审查："""
        
        response = item.get('compliance_analysis', '')
        
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
