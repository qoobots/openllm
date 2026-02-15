import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class MedicalImageDataset(Dataset):
    """医疗影像数据集
    
    支持X光/CT/MRI多模态影像
    """
    
    def __init__(self, data_path, processor, max_length=512, image_size=224):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图像
        image = Image.open(item['image_path']).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        
        # 构建文本输入
        text_prompt = self._format_prompt(item)
        
        # 处理多模态输入
        inputs = self.processor(
            text=text_prompt,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        
        # 创建labels
        response_text = item.get('diagnosis', '')
        response_inputs = self.processor(
            text=response_text,
            images=None,
            return_tensors="pt",
            max_length=self.max_length
        )
        
        labels = inputs['input_ids'].clone()
        prompt_length = len(inputs['input_ids'][0])
        labels[0, :prompt_length] = -100
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }
    
    def _format_prompt(self, item):
        """格式化医疗诊断提示词"""
        prompt = f"""医疗影像诊断任务：
影像类型：{item.get('image_type', 'X光')}
患者信息：{item.get('patient_info', '未知')}
临床病史：{item.get('medical_history', '无')}

请分析影像并提供诊断意见：
1. 影像所见
2. 诊断结果
3. 鉴别诊断
4. 建议检查

诊断报告："""
        return prompt


class MedicalDataProcessor:
    """医疗数据处理处理器"""
    
    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
    
    def create_rag_dataset(self, medical_images_path, knowledge_base_path):
        """创建RAG-Finetuning数据集"""
        # 融合医疗知识库
        pass
    
    def create_heterogeneity_dataset(self, heterogeneity_data_path):
        """创建设备异构性数据集"""
        # 统一处理不同设备的影像数据
        pass
