import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class FinancialComplianceDataset(Dataset):
    """金融合规审查数据集
    
    支持合同审查、条款识别、合规标注
    """
    
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
        
        # 构建输入格式
        prompt = self._format_prompt(item)
        response = item.get('compliance_analysis', '')
        
        # Tokenize
        inputs = self.tokenizer(
            prompt + response,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 创建labels (只计算response部分的loss)
        prompt_length = len(self.tokenizer(prompt, truncation=True, max_length=self.max_length)['input_ids'])
        labels = inputs['input_ids'].clone()
        labels[0, :prompt_length] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }
    
    def _format_prompt(self, item):
        """格式化提示词"""
        prompt = f"""合同审查任务：
合同类型：{item.get('contract_type', '贷款合同')}
合同金额：{item.get('amount', '未知')}
签署日期：{item.get('date', '未知')}

合同内容：
{item.get('contract_content', '')}

请进行合规性审查，包括：
1. 阴阳合同识别
2. 监管条款映射
3. 算法公平性评估

合规分析："""
        return prompt


class FinancialDataProcessor:
    """金融合规数据处理器"""
    
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def create_dapt_dataset(self, domain_texts_path, output_path):
        """创建领域自适应预训练数据集"""
        # 这里实现DAPT数据处理逻辑
        pass
    
    def create_sft_dataset(self, contract_path, output_path):
        """创建监督微调数据集"""
        # 这里实现SFT数据处理逻辑
        pass
    
    def create_dpo_dataset(self, preference_pairs_path, output_path):
        """创建偏好优化数据集"""
        # 这里实现DPO数据处理逻辑
        pass
