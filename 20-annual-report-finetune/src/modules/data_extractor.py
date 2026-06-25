"""
数据提取模块
从年报文本中提取结构化数据，包括财务指标、经营数据等
"""

import re
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger
import jieba


@dataclass
class FinancialIndicator:
    """财务指标"""
    name: str
    value: float
    unit: str = "元"
    year: int = 2023


@dataclass
class BusinessSegment:
    """业务分部"""
    segment_name: str
    revenue: float
    revenue_growth: float
    gross_margin: float
    description: str


@dataclass
class RiskItem:
    """风险项"""
    risk_type: str
    description: str
    severity: str  # high, medium, low
    impact: str


@dataclass
class StructuredAnnualReport:
    """结构化年报数据"""
    report_id: str
    company_name: str
    stock_code: str
    year: int
    financial_indicators: Dict[str, FinancialIndicator]
    business_segments: List[BusinessSegment]
    risks: List[RiskItem]
    highlights: List[str]
    raw_data: Dict[str, Any]


class DataExtractor:
    """年报数据提取器"""

    # 关键财务指标关键词
    FINANCIAL_KEYWORDS = {
        '营业收入': 'revenue',
        '营业总收入': 'total_revenue',
        '净利润': 'net_profit',
        '归母净利润': 'parent_net_profit',
        '扣非净利润': 'non_recurring_net_profit',
        '毛利率': 'gross_margin',
        '净利率': 'net_margin',
        '净资产收益率': 'roe',
        '总资产': 'total_assets',
        '总负债': 'total_liabilities',
        '资产负债率': 'debt_ratio',
        '流动资产': 'current_assets',
        '流动负债': 'current_liabilities',
        '经营活动现金流': 'operating_cash_flow',
        '每股收益': 'eps',
        '市盈率': 'pe_ratio',
        '市净率': 'pb_ratio'
    }

    # 风险类型关键词
    RISK_KEYWORDS = [
        '政策风险', '市场风险', '经营风险', '财务风险',
        '技术风险', '管理风险', '法律风险', '信用风险',
        '汇率风险', '流动性风险', '合规风险'
    ]

    def __init__(self):
        self.logger = logger

    def extract(self, parsed_report: dict) -> StructuredAnnualReport:
        """
        从解析后的年报数据中提取结构化信息

        Args:
            parsed_report: 解析后的年报数据

        Returns:
            StructuredAnnualReport: 结构化数据
        """
        try:
            self.logger.info(f"开始提取结构化数据: {parsed_report['report_id']}")

            # 提取财务指标
            financial_indicators = self._extract_financial_indicators(
                parsed_report['full_text'],
                parsed_report['tables']
            )

            # 提取业务分部信息
            business_segments = self._extract_business_segments(
                parsed_report['full_text'],
                parsed_report['sections']
            )

            # 提取风险信息
            risks = self._extract_risks(
                parsed_report['full_text'],
                parsed_report.get('sections', [])
            )

            # 提取亮点信息
            highlights = self._extract_highlights(
                parsed_report['full_text']
            )

            structured_report = StructuredAnnualReport(
                report_id=parsed_report['report_id'],
                company_name=parsed_report['company_name'],
                stock_code=parsed_report['stock_code'],
                year=parsed_report['year'],
                financial_indicators=financial_indicators,
                business_segments=business_segments,
                risks=risks,
                highlights=highlights,
                raw_data=parsed_report
            )

            self.logger.info(f"结构化数据提取完成")
            return structured_report

        except Exception as e:
            self.logger.error(f"数据提取失败: {str(e)}")
            raise

    def _extract_financial_indicators(
        self,
        full_text: str,
        tables: List[dict]
    ) -> Dict[str, FinancialIndicator]:
        """提取财务指标"""
        indicators = {}

        # 从文本中提取
        for chinese_name, english_name in self.FINANCIAL_KEYWORDS.items():
            # 正则匹配指标
            patterns = [
                rf'{chinese_name}[：:：\s]*([+-]?\d+\.?\d*)[万亿千百元]*',
                rf'{chinese_name}\s*[为是]*\s*([+-]?\d+\.?\d*)',
                rf'{chinese_name}.*?([+-]?\d+\.?\d*)\s*[元%]'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, full_text)
                if matches:
                    try:
                        value = float(matches[0])
                        # 简单的单位转换（实际需要更复杂的处理）
                        if '万' in full_text[max(0, full_text.find(matches[0]) - 10):full_text.find(matches[0]) + 10]:
                            value *= 10000
                        elif '亿' in full_text[max(0, full_text.find(matches[0]) - 10):full_text.find(matches[0]) + 10]:
                            value *= 100000000

                        indicators[english_name] = FinancialIndicator(
                            name=chinese_name,
                            value=value
                        )
                        break
                    except (ValueError, IndexError):
                        continue

        # 从表格中提取（财务数据通常在表格中）
        for table in tables:
            if any(keyword in str(table['headers']) for keyword in ['财务指标', '主要会计数据', '财务摘要']):
                self._extract_from_table(table, indicators)

        return indicators

    def _extract_from_table(self, table: dict, indicators: Dict):
        """从表格中提取财务指标"""
        headers = table['headers']
        rows = table['rows']

        # 查找数值列
        value_columns = []
        for i, header in enumerate(headers):
            if any(char.isdigit() for char in str(header)) or '年' in str(header):
                value_columns.append(i)

        for row in rows:
            if not row:
                continue

            row_text = ' '.join([str(cell) for cell in row])

            for chinese_name, english_name in self.FINANCIAL_KEYWORDS.items():
                if chinese_name in row_text:
                    for col_idx in value_columns:
                        if col_idx < len(row):
                            try:
                                value_str = str(row[col_idx]).replace(',', '').replace('%', '')
                                if value_str and value_str != '-':
                                    value = float(value_str)
                                    if english_name not in indicators:
                                        indicators[english_name] = FinancialIndicator(
                                            name=chinese_name,
                                            value=value
                                        )
                            except (ValueError, IndexError):
                                pass

    def _extract_business_segments(
        self,
        full_text: str,
        sections: List[dict]
    ) -> List[BusinessSegment]:
        """提取业务分部信息"""
        segments = []

        # 查找包含分部信息的章节
        for section in sections:
            if any(keyword in section['title'] for keyword in ['主营业务', '分部报告', '经营情况', '业务结构']):
                text = section['content']

                # 尝试提取分部信息
                segment_patterns = [
                    r'([^\s]{2,10})[：:：]\s*([+-]?\d+\.?\d*)\s*[元万]*',
                    r'([^\s]{2,10})收入.*?([+-]?\d+\.?\d*)',
                ]

                for pattern in segment_patterns:
                    matches = re.findall(pattern, text)
                    for match in matches:
                        if len(match) == 2:
                            try:
                                segment_name = match[0]
                                revenue = float(match[1].replace(',', ''))

                                # 尝试提取增长率
                                growth_match = re.search(
                                    rf'{segment_name}.*?增长.*?([+-]?\d+\.?\d*)%',
                                    text
                                )
                                revenue_growth = float(growth_match.group(1)) if growth_match else 0.0

                                segments.append(BusinessSegment(
                                    segment_name=segment_name,
                                    revenue=revenue,
                                    revenue_growth=revenue_growth,
                                    gross_margin=0.0,  # 需要更复杂的提取
                                    description=""
                                ))
                            except (ValueError, AttributeError, IndexError):
                                pass

        return segments

    def _extract_risks(self, full_text: str, sections: List[dict]) -> List[RiskItem]:
        """提取风险信息"""
        risks = []

        # 查找风险相关章节
        risk_section = None
        for section in sections:
            if '风险' in section['title'] or '可能面对' in section['title']:
                risk_section = section
                break

        if risk_section:
            text = risk_section['content']

            # 提取风险项
            risk_patterns = [
                r'([^\s\n]{2,15}风险)[：:：]\s*([^\n。]+)',
                r'(一、|二、|三、|四、|五、|六、|七、|八、|九、|十、)[^\n]*?([^\s\n]{2,15}风险)'
            ]

            for pattern in risk_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    try:
                        if len(match) == 2:
                            risk_type = match[0] if '风险' in match[0] else match[1]
                            description = match[1] if '风险' in match[0] else ''

                            # 判断风险严重程度
                            severity = 'medium'
                            if any(keyword in description for keyword in ['重大', '严重', '显著']):
                                severity = 'high'
                            elif any(keyword in description for keyword in ['较小', '轻微', '低']):
                                severity = 'low'

                            if description:
                                risks.append(RiskItem(
                                    risk_type=risk_type,
                                    description=description[:200],
                                    severity=severity,
                                    impact=""
                                ))
                    except (ValueError, IndexError):
                        pass

        # 如果没有找到风险章节，从全文中提取
        if not risks:
            for risk_type in self.RISK_KEYWORDS:
                if risk_type in full_text:
                    # 提取风险描述
                    risk_match = re.search(
                        rf'{risk_type}[：:：]\s*([^\n。]{10,200})',
                        full_text
                    )
                    if risk_match:
                        risks.append(RiskItem(
                            risk_type=risk_type,
                            description=risk_match.group(1),
                            severity='medium',
                            impact=""
                        ))

        return risks

    def _extract_highlights(self, full_text: str) -> List[str]:
        """提取年度亮点信息"""
        highlights = []

        highlight_keywords = [
            '主要亮点', '经营亮点', '业绩亮点', '工作亮点',
            '重大进展', '重大突破', '重要成果', '显著成效'
        ]

        for keyword in highlight_keywords:
            if keyword in full_text:
                # 提取亮点段落
                match = re.search(
                    rf'{keyword}[：:：\s]*([^\n]+)',
                    full_text
                )
                if match:
                    highlights.append(match.group(1).strip())

        # 如果没有找到，尝试从管理层讨论中提取
        if not highlights:
            # 提取包含增长、提升等正面词汇的句子
            positive_patterns = [
                r'(实现|完成|达到).{0,50}(增长|提升|突破|新高)',
                r'(同比|环比).{0,30}(增长|上升|改善)',
                r'(荣获|获得|取得).{0,50}(奖项|认证|荣誉)'
            ]

            for pattern in positive_patterns:
                matches = re.findall(pattern, full_text)
                for match in matches:
                    if isinstance(match, tuple):
                        highlight = ''.join(match)
                    else:
                        highlight = match
                    if len(highlight) > 10 and len(highlight) < 100:
                        highlights.append(highlight)

        return highlights[:10]  # 返回前10个亮点

    def extract_batch(self, parsed_reports: List[dict]) -> List[StructuredAnnualReport]:
        """批量提取结构化数据"""
        results = []

        for report in parsed_reports:
            try:
                structured = self.extract(report)
                results.append(structured)
            except Exception as e:
                self.logger.error(f"提取失败 {report.get('report_id', '')}: {str(e)}")
                continue

        return results

    def save_to_json(self, structured_report: StructuredAnnualReport, output_path: str):
        """保存为JSON文件"""
        output_data = {
            'report_id': structured_report.report_id,
            'company_name': structured_report.company_name,
            'stock_code': structured_report.stock_code,
            'year': structured_report.year,
            'financial_indicators': {
                key: {
                    'name': indicator.name,
                    'value': indicator.value,
                    'unit': indicator.unit
                }
                for key, indicator in structured_report.financial_indicators.items()
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

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"结构化数据已保存: {output_path}")


def main():
    """测试代码"""
    extractor = DataExtractor()

    # 测试单个文件提取
    # with open("data/processed/600519_2023.json", 'r', encoding='utf-8') as f:
    #     parsed_data = json.load(f)
    # structured = extractor.extract(parsed_data)
    # extractor.save_to_json(structured, "data/processed/structured/600519_2023_structured.json")


if __name__ == "__main__":
    main()
