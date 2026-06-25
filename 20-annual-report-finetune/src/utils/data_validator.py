"""
数据验证工具
"""

import json
from typing import Dict, List, Any
from pathlib import Path
from loguru import logger


class DataValidator:
    """数据验证器"""

    def __init__(self):
        self.logger = logger

    def validate_parsed_report(self, report: Dict) -> tuple[bool, List[str]]:
        """
        验证解析后的年报数据

        Args:
            report: 解析后的年报数据

        Returns:
            tuple: (是否有效, 错误信息列表)
        """
        errors = []

        # 检查必需字段
        required_fields = ['report_id', 'company_name', 'year', 'full_text']
        for field in required_fields:
            if field not in report or not report[field]:
                errors.append(f"缺少必需字段: {field}")

        # 检查年份是否合理
        if 'year' in report:
            year = report['year']
            if not isinstance(year, int) or year < 2000 or year > 2030:
                errors.append(f"年份不合理: {year}")

        # 检查文本长度
        if 'full_text' in report:
            text_length = len(report['full_text'])
            if text_length < 1000:
                errors.append(f"文本长度过短: {text_length}")
            elif text_length < 100:
                errors.append(f"文本长度异常短: {text_length}")

        return len(errors) == 0, errors

    def validate_structured_report(self, report: Dict) -> tuple[bool, List[str]]:
        """
        验证结构化年报数据

        Args:
            report: 结构化年报数据

        Returns:
            tuple: (是否有效, 错误信息列表)
        """
        errors = []

        # 检查必需字段
        required_fields = ['report_id', 'company_name', 'financial_indicators']
        for field in required_fields:
            if field not in report:
                errors.append(f"缺少必需字段: {field}")

        # 检查财务指标是否为空
        if 'financial_indicators' in report:
            if not report['financial_indicators']:
                errors.append("财务指标为空")

        return len(errors) == 0, errors

    def validate_json_file(self, file_path: str) -> tuple[bool, Dict]:
        """
        验证JSON文件

        Args:
            file_path: JSON文件路径

        Returns:
            tuple: (是否有效, 数据字典或错误信息)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return True, data
        except json.JSONDecodeError as e:
            return False, {"error": f"JSON解析失败: {str(e)}"}
        except FileNotFoundError:
            return False, {"error": f"文件不存在: {file_path}"}
        except Exception as e:
            return False, {"error": f"未知错误: {str(e)}"}

    def validate_batch(self, file_paths: List[str], report_type: str = 'parsed') -> Dict[str, Any]:
        """
        批量验证文件

        Args:
            file_paths: 文件路径列表
            report_type: 报告类型 ('parsed' 或 'structured')

        Returns:
            Dict: 验证结果统计
        """
        results = {
            'total': len(file_paths),
            'valid': 0,
            'invalid': 0,
            'errors': []
        }

        for file_path in file_paths:
            is_valid, data = self.validate_json_file(file_path)
            if not is_valid:
                results['invalid'] += 1
                results['errors'].append({
                    'file': file_path,
                    'error': data.get('error', '未知错误')
                })
                continue

            if report_type == 'parsed':
                is_valid, errors = self.validate_parsed_report(data)
            else:
                is_valid, errors = self.validate_structured_report(data)

            if is_valid:
                results['valid'] += 1
            else:
                results['invalid'] += 1
                results['errors'].append({
                    'file': file_path,
                    'errors': errors
                })

        return results


def main():
    """测试代码"""
    validator = DataValidator()

    # 测试验证
    test_report = {
        'report_id': '600519_2023',
        'company_name': '贵州茅台',
        'year': 2023,
        'full_text': 'test' * 100  # 模拟长文本
    }

    is_valid, errors = validator.validate_parsed_report(test_report)
    print(f"是否有效: {is_valid}")
    print(f"错误: {errors}")


if __name__ == "__main__":
    main()
