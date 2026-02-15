import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class MedicalDiagnosisModel:
    """医疗AI诊断微调模型
    
    基于Qwen3-VL-8B-Thinking多模态模型
    支持X光/CT/MRI影像诊断
    """
    
    def __init__(self, model_path, use_lora=True, lora_r=16, lora_alpha=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.use_lora = use_lora
        
        # 加载模型和processor
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 应用LoRA
        if use_lora:
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
    
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        """前向传播"""
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate_diagnosis(self, image, text_prompt, max_length=512, temperature=0.7):
        """生成诊断报告"""
        inputs = self.processor(
            text=text_prompt,
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1,
                do_sample=True
            )
        
        # 解码生成结果
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return generated_text
    
    def save_model(self, output_dir):
        """保存模型"""
        if self.use_lora:
            self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)


class TrafficLightAlertSystem:
    """红绿灯警报系统
    
    绿灯（无误）→ 黄灯（需确认）→ 红灯（强制医生复核）
    """
    
    def __init__(self, confidence_thresholds=(0.9, 0.7)):
        """
        Args:
            confidence_thresholds: (绿灯阈值, 黄灯阈值)
        """
        self.green_threshold = confidence_thresholds[0]
        self.yellow_threshold = confidence_thresholds[1]
    
    def predict_alert(self, diagnosis_result, confidence_score):
        """预测警报级别"""
        if confidence_score >= self.green_threshold:
            return "GREEN", "诊断可信，无需复核"
        elif confidence_score >= self.yellow_threshold:
            return "YELLOW", "需医生确认"
        else:
            return "RED", "强制医生复核"
