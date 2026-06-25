"""
评估脚本
评估模型性能
"""

import argparse
import yaml
import json
import sys
from pathlib import Path
from loguru import logger
from typing import Dict, List

# 添加src到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.model import AnnualReportModel


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, config_path: str, model_path: str):
        self.config = self._load_config(config_path)
        self.model_path = model_path

        # 初始化模型
        self.model_handler = AnnualReportModel(self.config)
        self.model_handler.load_peft_model(model_path)

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def evaluate_financial_extraction(self, test_data: List[Dict]) -> Dict:
        """
        评估财务指标提取准确率

        Args:
            test_data: 测试数据

        Returns:
            Dict: 评估结果
        """
        correct = 0
        total = 0

        errors = []

        for item in test_data:
            # 提取财务指标
            # TODO: 实现具体的提取逻辑
            pass

        accuracy = correct / total if total > 0 else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'errors': errors
        }

    def evaluate_qa(self, test_data: List[Dict]) -> Dict:
        """
        评估问答准确率

        Args:
            test_data: 测试数据

        Returns:
            Dict: 评估结果
        """
        # TODO: 实现问答评估逻辑
        # 可以使用BLEU、ROUGE、BERTScore等指标
        pass

        return {
            'accuracy': 0.0,
            'bleu': 0.0,
            'rouge': 0.0
        }

    def evaluate_risk_detection(self, test_data: List[Dict]) -> Dict:
        """
        评估风险检测准确率

        Args:
            test_data: 测试数据

        Returns:
            Dict: 评估结果
        """
        # TODO: 实现风险检测评估逻辑
        pass

        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

    def evaluate_all(self, test_data_path: str, output_path: str = None):
        """
        执行完整评估

        Args:
            test_data_path: 测试数据路径
            output_path: 输出文件路径
        """
        # 加载测试数据
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]

        logger.info(f"加载测试数据: {len(test_data)} 条")

        # 执行各项评估
        results = {
            'financial_extraction': self.evaluate_financial_extraction(test_data),
            'qa': self.evaluate_qa(test_data),
            'risk_detection': self.evaluate_risk_detection(test_data)
        }

        # 汇总结果
        logger.info("="*50)
        logger.info("评估结果:")
        logger.info("="*50)

        for metric_name, metric_result in results.items():
            logger.info(f"\n{metric_name}:")
            for key, value in metric_result.items():
                logger.info(f"  {key}: {value}")

        # 与目标值对比
        targets = self.config.get('evaluation', {}).get('targets', {})
        logger.info("\n目标值对比:")
        for metric_name, target_value in targets.items():
            actual_value = results.get(metric_name.replace('_accuracy', ''), {}).get('accuracy', 0)
            status = "✓" if actual_value >= target_value else "✗"
            logger.info(f"  {status} {metric_name}: {actual_value:.4f} (目标: {target_value:.4f})")

        # 保存结果
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"\n评估结果已保存到: {output_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="模型评估脚本")
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
        '--test_data',
        type=str,
        required=True,
        help='测试数据路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='输出文件路径'
    )

    args = parser.parse_args()

    # 初始化评估器
    logger.info(f"加载模型: {args.model_path}")
    evaluator = ModelEvaluator(args.config, args.model_path)

    # 执行评估
    logger.info("开始评估...")
    results = evaluator.evaluate_all(args.test_data, args.output)

    logger.info("\n评估完成！")


if __name__ == "__main__":
    main()
