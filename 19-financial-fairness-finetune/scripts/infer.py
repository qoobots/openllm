#!/usr/bin/env python3
"""
FinancialFairness推理脚本
"""

import argparse
import yaml

from src.model import FinancialFairnessModel


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='FinancialFairness推理')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.txt')
    
    args = parser.parse_args()
    
    # 加载模型
    model = FinancialFairnessModel(args.model_path, use_lora=True)
    
    # 生成结果
    result = model.generate(args.input, max_length=512, temperature=0.3)
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"推理结果已保存到 {args.output}")
    print("\n结果：")
    print(result)


if __name__ == "__main__":
    main()
