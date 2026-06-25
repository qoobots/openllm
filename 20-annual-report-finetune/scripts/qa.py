"""
问答交互脚本
提供交互式问答功能
"""

import argparse
import yaml
import sys
from pathlib import Path
from loguru import logger

# 添加src到路径
sys.path.append(str(Path(__file__).parent.parent))

from scripts.infer import AnnualReportInference


def main():
    parser = argparse.ArgumentParser(description="年报问答交互脚本")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型路径'
    )
    parser.add_argument(
        '--report_file',
        type=str,
        required=True,
        help='年报文本文件路径'
    )

    args = parser.parse_args()

    # 加载年报文本
    with open(args.report_file, 'r', encoding='utf-8') as f:
        report_text = f.read()

    # 初始化推理器
    logger.info(f"加载模型: {args.model_path}")
    inference = AnnualReportInference(args.config, args.model_path)

    logger.info("年报已加载，可以开始提问。输入 'quit' 退出。")
    print("\n" + "="*60)

    while True:
        question = input("\n请输入问题: ").strip()

        if question.lower() in ['quit', 'exit', '退出']:
            logger.info("退出问答系统")
            break

        if not question:
            continue

        # 生成回答
        print("\n正在思考...")
        answer = inference.analyze(report_text, question)

        print("\n回答:")
        print("-"*60)
        print(answer)
        print("-"*60)


if __name__ == "__main__":
    main()
