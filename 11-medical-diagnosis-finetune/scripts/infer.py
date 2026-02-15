#!/usr/bin/env python3
"""
医疗AI误诊率优化微调系统推理脚本
"""

import argparse
import yaml
from PIL import Image

from src.model import MedicalDiagnosisModel, TrafficLightAlertSystem


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='医疗AI诊断推理')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--patient_info', type=str, default='')
    parser.add_argument('--output', type=str, default='diagnosis.txt')
    
    args = parser.parse_args()
    
    # 加载模型
    model = MedicalDiagnosisModel(args.model_path, use_lora=True)
    
    # 加载图像
    image = Image.open(args.image_path).convert('RGB')
    
    # 构建输入提示词
    text_prompt = f"""医疗影像诊断任务：
患者信息：{args.patient_info}

请分析影像并提供诊断意见：
1. 影像所见
2. 诊断结果
3. 鉴别诊断
4. 建议检查

诊断报告："""
    
    # 生成诊断报告
    diagnosis = model.generate_diagnosis(image, text_prompt, max_length=512, temperature=0.3)
    
    # 使用红绿灯警报系统评估
    alert_system = TrafficLightAlertSystem(confidence_thresholds=(0.9, 0.7))
    alert_level, alert_message = alert_system.predict_alert(diagnosis, 0.85)
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(f"诊断报告：\n{diagnosis}\n\n")
        f.write(f"警报级别：{alert_level}\n")
        f.write(f"警报说明：{alert_message}\n")
    
    print(f"诊断结果已保存到 {args.output}")
    print("\n诊断报告：")
    print(diagnosis)
    print(f"\n警报级别：{alert_level}")
    print(f"警报说明：{alert_message}")


if __name__ == "__main__":
    main()
