"""
推理脚本
使用训练好的模型进行年报分析
"""

import argparse
import yaml
import torch
from pathlib import Path
from loguru import logger
import sys
import json

# 添加src到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.model import AnnualReportModel


class AnnualReportInference:
    """年报分析推理器"""

    def __init__(self, config_path: str, model_path: str):
        self.config = self._load_config(config_path)
        self.model_path = model_path

        # 初始化模型
        self.model_handler = AnnualReportModel(self.config)
        self.model_handler.load_peft_model(model_path)

        # 推理参数
        self.inference_config = self.config.get('inference', {})

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def analyze(self, report_text: str, question: str = "") -> str:
        """
        分析年报

        Args:
            report_text: 年报文本
            question: 问题（可选）

        Returns:
            str: 分析结果
        """
        # 构建提示词
        if question:
            prompt = f"""<|im_start|>system
你是一名专业的证券分析师，擅长分析上市公司年报，提取关键财务指标，评估经营风险，提供投资建议。

<|im_start|>user
基于以下年报内容回答问题：

年报内容：
{report_text[:3000]}

问题：
{question}<|im_end|>
<|im_start|>assistant
"""
        else:
            prompt = f"""<|im_start|>system
你是一名专业的证券分析师，擅长分析上市公司年报，提取关键财务指标，评估经营风险，提供投资建议。

<|im_start|>user
请分析以下年报，提取关键财务指标，评估经营风险，并提供投资建议。

年报内容：
{report_text[:3000]}<|im_end|>
<|im_start|>assistant
"""

        # Tokenize
        inputs = self.model_handler.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model_handler.model.device)

        # 生成
        with torch.no_grad():
            outputs = self.model_handler.model.generate(
                **inputs,
                max_new_tokens=self.inference_config.get('max_new_tokens', 1024),
                temperature=self.inference_config.get('temperature', 0.7),
                top_p=self.inference_config.get('top_p', 0.9),
                top_k=self.inference_config.get('top_k', 50),
                do_sample=self.inference_config.get('do_sample', True),
                pad_token_id=self.model_handler.tokenizer.pad_token_id,
                eos_token_id=self.model_handler.tokenizer.eos_token_id
            )

        # 解码
        response = self.model_handler.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def extract_financial_indicators(self, report_text: str) -> dict:
        """
        提取财务指标

        Args:
            report_text: 年报文本

        Returns:
            dict: 财务指标
        """
        question = """请从年报中提取以下财务指标：
1. 营业收入
2. 净利润
3. 毛利率
4. 净资产收益率（ROE）
5. 资产负债率

请以JSON格式返回结果。"""

        response = self.analyze(report_text, question)

        # 尝试解析JSON
        try:
            # 提取JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass

        return {'raw_response': response}

    def assess_risks(self, report_text: str) -> list:
        """
        评估经营风险

        Args:
            report_text: 年报文本

        Returns:
            list: 风险列表
        """
        question = """请分析年报中提到的经营风险，包括：
1. 政策风险
2. 市场风险
3. 财务风险
4. 技术风险
5. 管理风险

请列出每项风险的类型、描述和严重程度（高/中/低）。"""

        response = self.analyze(report_text, question)
        return response

    def generate_report(self, report_text: str, company_info: dict = None) -> dict:
        """
        生成完整的分析报告

        Args:
            report_text: 年报文本
            company_info: 公司信息（可选）

        Returns:
            dict: 分析报告
        """
        logger.info("生成完整分析报告...")

        # 提取财务指标
        financial_indicators = self.extract_financial_indicators(report_text)

        # 评估风险
        risks = self.assess_risks(report_text)

        # 综合分析
        analysis = self.analyze(report_text, question="请对公司的经营状况进行综合分析，包括财务健康度、竞争优势、发展前景等。")

        # 投资建议
        recommendation = self.analyze(report_text, question="基于以上分析，请给出投资建议和评级（买入/持有/卖出）。")

        report = {
            'company_info': company_info or {},
            'financial_indicators': financial_indicators,
            'risks': risks,
            'analysis': analysis,
            'recommendation': recommendation
        }

        return report


def main():
    parser = argparse.ArgumentParser(description="年报分析推理脚本")
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
        '--report_text',
        type=str,
        help='年报文本（直接输入）'
    )
    parser.add_argument(
        '--report_file',
        type=str,
        help='年报文本文件路径'
    )
    parser.add_argument(
        '--question',
        type=str,
        default='',
        help='问题（可选）'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='输出文件路径'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['analyze', 'extract', 'risk', 'report'],
        default='analyze',
        help='推理模式'
    )

    args = parser.parse_args()

    # 获取年报文本
    if args.report_file:
        with open(args.report_file, 'r', encoding='utf-8') as f:
            report_text = f.read()
    elif args.report_text:
        report_text = args.report_text
    else:
        logger.error("请提供年报文本（--report_text 或 --report_file）")
        return

    # 初始化推理器
    logger.info(f"加载模型: {args.model_path}")
    inference = AnnualReportInference(args.config, args.model_path)

    # 执行推理
    if args.mode == 'analyze':
        result = inference.analyze(report_text, args.question)
    elif args.mode == 'extract':
        result = inference.extract_financial_indicators(report_text)
    elif args.mode == 'risk':
        result = inference.assess_risks(report_text)
    elif args.mode == 'report':
        result = inference.generate_report(report_text)

    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {args.output}")
    else:
        print("\n" + "="*50)
        print("分析结果:")
        print("="*50)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
