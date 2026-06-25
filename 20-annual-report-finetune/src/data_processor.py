"""
数据处理主模块
整合PDF解析、数据提取、文本清洗等功能
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from tqdm import tqdm

from src.modules.pdf_parser import PDFParser, ParsedAnnualReport
from src.modules.data_extractor import DataExtractor, StructuredAnnualReport
from src.utils.text_cleaner import TextCleaner
from src.utils.data_validator import DataValidator


class AnnualReportDataProcessor:
    """年报数据处理器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.pdf_parser = PDFParser()
        self.data_extractor = DataExtractor()
        self.text_cleaner = TextCleaner()
        self.data_validator = DataValidator()

        self.logger = logger

    def process_single_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        save_intermediate: bool = True
    ) -> Optional[Dict]:
        """
        处理单个PDF文件

        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            save_intermediate: 是否保存中间结果

        Returns:
            Dict: 处理结果
        """
        try:
            self.logger.info(f"处理PDF: {pdf_path}")

            # 1. 解析PDF
            parsed_report = self.pdf_parser.parse_pdf(pdf_path)

            # 2. 文本清洗
            parsed_report.full_text = self.text_cleaner.clean(parsed_report.full_text)

            # 3. 验证解析结果
            report_dict = {
                'report_id': parsed_report.report_id,
                'company_name': parsed_report.company_name,
                'stock_code': parsed_report.stock_code,
                'year': parsed_report.year,
                'full_text': parsed_report.full_text,
                'sections': [
                    {
                        'title': s.title,
                        'content': self.text_cleaner.clean(s.content),
                        'page_range': s.page_range
                    }
                    for s in parsed_report.sections
                ],
                'tables': [
                    {
                        'title': t.title,
                        'headers': t.headers,
                        'rows': t.rows,
                        'page': t.page
                    }
                    for t in parsed_report.tables
                ],
                'metadata': parsed_report.metadata
            }

            is_valid, errors = self.data_validator.validate_parsed_report(report_dict)
            if not is_valid:
                self.logger.warning(f"解析数据验证失败: {errors}")
                if save_intermediate:
                    self._save_result(
                        report_dict,
                        os.path.join(output_dir, 'invalid', f"{parsed_report.report_id}.json")
                    )
                return None

            # 4. 保存解析结果
            if save_intermediate:
                self._save_result(
                    report_dict,
                    os.path.join(output_dir, 'parsed', f"{parsed_report.report_id}.json")
                )

            # 5. 提取结构化数据
            structured_report = self.data_extractor.extract(report_dict)

            # 6. 验证结构化数据
            structured_dict = {
                'report_id': structured_report.report_id,
                'company_name': structured_report.company_name,
                'stock_code': structured_report.stock_code,
                'year': structured_report.year,
                'financial_indicators': {
                    key: {
                        'name': ind.name,
                        'value': ind.value,
                        'unit': ind.unit
                    }
                    for key, ind in structured_report.financial_indicators.items()
                },
                'business_segments': [
                    {
                        'segment_name': seg.segment_name,
                        'revenue': seg.revenue,
                        'revenue_growth': seg.revenue_growth,
                        'gross_margin': seg.gross_margin,
                        'description': seg.description
                    }
                    for seg in structured_report.business_segments
                ],
                'risks': [
                    {
                        'risk_type': risk.risk_type,
                        'description': risk.description,
                        'severity': risk.severity,
                        'impact': risk.impact
                    }
                    for risk in structured_report.risks
                ],
                'highlights': structured_report.highlights
            }

            is_valid, errors = self.data_validator.validate_structured_report(structured_dict)
            if not is_valid:
                self.logger.warning(f"结构化数据验证失败: {errors}")
            else:
                # 保存结构化数据
                if save_intermediate:
                    self._save_result(
                        structured_dict,
                        os.path.join(output_dir, 'structured', f"{parsed_report.report_id}_structured.json")
                    )

            self.logger.info(f"处理完成: {parsed_report.report_id}")
            return {
                'report_id': parsed_report.report_id,
                'company_name': parsed_report.company_name,
                'status': 'success'
            }

        except Exception as e:
            self.logger.error(f"处理失败 {pdf_path}: {str(e)}")
            return None

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        save_intermediate: bool = True
    ) -> Dict[str, any]:
        """
        批量处理PDF文件

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            save_intermediate: 是否保存中间结果

        Returns:
            Dict: 处理结果统计
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        if save_intermediate:
            os.makedirs(os.path.join(output_dir, 'parsed'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'structured'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'invalid'), exist_ok=True)

        # 获取所有PDF文件
        pdf_files = list(Path(input_dir).glob("*.pdf"))

        if not pdf_files:
            self.logger.warning(f"目录中没有找到PDF文件: {input_dir}")
            return {'total': 0, 'success': 0, 'failed': 0}

        self.logger.info(f"找到 {len(pdf_files)} 个PDF文件")

        # 批量处理
        results = {
            'total': len(pdf_files),
            'success': 0,
            'failed': 0,
            'errors': []
        }

        for pdf_file in tqdm(pdf_files, desc="处理PDF"):
            result = self.process_single_pdf(
                str(pdf_file),
                output_dir,
                save_intermediate
            )

            if result:
                results['success'] += 1
            else:
                results['failed'] += 1
                results['errors'].append(str(pdf_file))

        self.logger.info(f"处理完成: 成功 {results['success']}/{results['total']}")
        return results

    def _save_result(self, data: Dict, output_path: str):
        """保存结果到文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def convert_to_training_format(
        self,
        structured_dir: str,
        output_path: str,
        task_type: str = 'qa'
    ):
        """
        将结构化数据转换为训练格式

        Args:
            structured_dir: 结构化数据目录
            output_path: 输出文件路径
            task_type: 任务类型 ('qa', 'extraction', 'analysis')
        """
        json_files = list(Path(structured_dir).glob("*_structured.json"))

        if not json_files:
            self.logger.warning(f"没有找到结构化数据: {structured_dir}")
            return

        training_samples = []

        for json_file in tqdm(json_files, desc=f"转换为{task_type}格式"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if task_type == 'qa':
                samples = self._create_qa_samples(data)
            elif task_type == 'extraction':
                samples = self._create_extraction_samples(data)
            elif task_type == 'analysis':
                samples = self._create_analysis_samples(data)
            else:
                self.logger.warning(f"未知任务类型: {task_type}")
                continue

            training_samples.extend(samples)

        # 保存为JSONL格式
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        self.logger.info(f"生成 {len(training_samples)} 条训练样本: {output_path}")

    def _create_qa_samples(self, data: Dict) -> List[Dict]:
        """创建问答对样本"""
        samples = []

        company = data['company_name']
        year = data['year']

        # 基于财务指标生成问答
        financial_indicators = data.get('financial_indicators', {})
        for key, indicator in financial_indicators.items():
            question = f"{company}{year}年的{indicator['name']}是多少？"
            answer = f"{company}{year}年的{indicator['name']}为{indicator['value']}{indicator['unit']}。"

            samples.append({
                'instruction': question,
                'input': '',
                'output': answer
            })

        # 基于风险生成问答
        risks = data.get('risks', [])
        if risks:
            risk_types = ', '.join([r['risk_type'] for r in risks])
            question = f"{company}{year}年面临的主要风险有哪些？"
            answer = f"{company}{year}年主要面临{risk_types}等风险。"

            samples.append({
                'instruction': question,
                'input': '',
                'output': answer
            })

        return samples

    def _create_extraction_samples(self, data: Dict) -> List[Dict]:
        """创建信息抽取样本"""
        return [{
            'instruction': f"请从以下年报数据中提取财务指标、业务分部和风险信息。",
            'input': json.dumps(data, ensure_ascii=False),
            'output': json.dumps({
                'financial_indicators': data.get('financial_indicators', {}),
                'business_segments': data.get('business_segments', []),
                'risks': data.get('risks', [])
            }, ensure_ascii=False)
        }]

    def _create_analysis_samples(self, data: Dict) -> List[Dict]:
        """创建分析样本"""
        samples = []

        company = data['company_name']
        year = data['year']

        # 财务健康度分析
        financial_indicators = data.get('financial_indicators', {})
        analysis_parts = [f"{company}{year}年度财务分析："]

        if 'revenue' in financial_indicators:
            revenue = financial_indicators['revenue']['value']
            analysis_parts.append(f"营业收入为{revenue}元。")

        if 'net_profit' in financial_indicators:
            net_profit = financial_indicators['net_profit']['value']
            analysis_parts.append(f"净利润为{net_profit}元。")

        if 'roe' in financial_indicators:
            roe = financial_indicators['roe']['value']
            if roe > 0.15:
                analysis_parts.append(f"净资产收益率(ROE)为{roe*100:.2f}%，盈利能力较强。")
            else:
                analysis_parts.append(f"净资产收益率(ROE)为{roe*100:.2f}%，盈利能力一般。")

        # 风险分析
        risks = data.get('risks', [])
        if risks:
            high_risks = [r for r in risks if r['severity'] == 'high']
            if high_risks:
                analysis_parts.append(f"存在{len(high_risks)}项高风险因素，需重点关注。")

        samples.append({
            'instruction': f"请分析{company}{year}年度的财务状况和经营风险。",
            'input': json.dumps(data, ensure_ascii=False),
            'output': ''.join(analysis_parts)
        })

        return samples


def main():
    """测试代码"""
    processor = AnnualReportDataProcessor()

    # 批量处理
    results = processor.process_batch(
        input_dir="data/raw",
        output_dir="data/processed",
        save_intermediate=True
    )

    print(f"处理结果: {results}")

    # 转换为训练格式
    processor.convert_to_training_format(
        structured_dir="data/processed/structured",
        output_path="data/processed/qa_train.jsonl",
        task_type='qa'
    )


if __name__ == "__main__":
    main()
