import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class MedicalInsuranceModel:
    """医疗AI医保合规微调模型
    
    基于通义千问医疗版，已通过等保三级认证
    """
    
    def __init__(self, model_path, use_lora=True, lora_r=16, lora_alpha=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.use_lora = use_lora
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        if use_lora:
            self.model = prepare_model_for_kbit_training(self.model)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def generate(self, prompt, max_length=512, temperature=0.3):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def save_model(self, output_dir):
        if self.use_lora:
            self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


class AIGCDocumentDetector:
    """AIGC文档鉴伪引擎
    
    检测文本中的"隐形水印"和"深度伪造"特征
    """
    
    def __init__(self, model_path):
        from transformers import AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def detect_watermark(self, text):
        """检测隐形水印"""
        # 实现水印检测逻辑
        pass
    
    def detect_deepfake(self, text):
        """检测深度伪造特征"""
        # 实现深度伪造检测逻辑
        pass


class RealTimeViolationAlert:
    """实时违规预警系统
    
    医生开处方、护士收费时违规操作实时预警
    """
    
    def __init__(self, model, knowledge_base):
        self.model = model
        self.knowledge_base = knowledge_base
    
    def check_violation(self, operation, context):
        """检查操作是否违规"""
        # 实时检查违规操作
        pass
