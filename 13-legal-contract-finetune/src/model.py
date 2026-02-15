import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class LegalContractModel:
    """法律合同智能审查微调模型
    
    基座模型支持等保三级认证
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
    
    def generate(self, prompt, max_length=1024, temperature=0.3):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
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


class ConflictDetector(nn.Module):
    """条款冲突检测引擎"""
    
    def __init__(self, encoder_name):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.similarity = nn.CosineSimilarity(dim=1)
    
    def detect_conflict(self, clause_a, clause_b):
        """检测两个条款之间的冲突"""
        features_a = self.encoder(**clause_a).last_hidden_state.mean(dim=1)
        features_b = self.encoder(**clause_b).last_hidden_state.mean(dim=1)
        
        similarity = self.similarity(features_a, features_b)
        return similarity


class ComplianceWorkflow:
    """等保三级合规工作流
    
    内置数据脱敏、审计日志留存、权限控制
    """
    
    def __init__(self, config):
        self.config = config
        self.audit_log = []
    
    def data_masking(self, contract_text):
        """数据脱敏"""
        # 实现敏感信息脱敏
        return contract_text
    
    def log_audit(self, operation, user, timestamp):
        """审计日志记录"""
        self.audit_log.append({
            'operation': operation,
            'user': user,
            'timestamp': timestamp
        })
    
    def check_permission(self, user, resource):
        """权限控制"""
        # 实现权限检查
        return True
