#!/usr/bin/env python3
"""
金融合规审查推理脚本
"""

import argparse
import yaml
import torch
from src.model import FinancialComplianceModel


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def infer_contract(model_path, contract_content):
    """对合同进行合规性审查"""
    
    # 加载模型
    model = FinancialComplianceModel(model_path, use_lora=True)
    
    # 构建提示词
    prompt = f"""请对以下合同进行合规性审查：

合同内容：
{contract_content}

请分析：
1. 阴阳合同识别
2. 监管条款映射
3. 算法公平性评估

合规分析："""
    
    # 生成结果
    result = model.generate(prompt, max_length=1024, temperature=0.3)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='金融合规审查推理')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--contract', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.txt')
    
    args = parser.parse_args()
    
    # 执行推理
    result = infer_contract(args.model_path, args.contract)
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"推理结果已保存到 {args.output}")
    print("\n审查结果：")
    print(result)


if __name__ == "__main__":
    main()
