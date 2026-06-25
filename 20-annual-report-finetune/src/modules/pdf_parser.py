"""
PDF解析模块
支持多种PDF格式解析，提取年报文本和表格数据
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pdfplumber
import PyPDF2
from loguru import logger


@dataclass
class PDFSection:
    """年报章节"""
    title: str
    content: str
    page_range: Tuple[int, int]


@dataclass
class PDFTable:
    """表格数据"""
    title: str
    headers: List[str]
    rows: List[List[str]]
    page: int


@dataclass
class ParsedAnnualReport:
    """解析后的年报数据"""
    report_id: str
    company_name: str
    stock_code: str
    year: int
    full_text: str
    sections: List[PDFSection]
    tables: List[PDFTable]
    metadata: Dict


class PDFParser:
    """年报PDF解析器"""

    # 常见年报章节标题
    COMMON_SECTIONS = [
        "重要提示",
        "公司简介",
        "会计数据和财务指标摘要",
        "管理层讨论与分析",
        "重要事项",
        "财务报告",
        "股份变动及股东情况",
        "董事、监事、高级管理人员和员工情况",
        "公司治理",
        "财务报告",
        "备查文件目录"
    ]

    def __init__(self):
        self.logger = logger

    def parse_pdf(self, pdf_path: str) -> ParsedAnnualReport:
        """
        解析年报PDF文件

        Args:
            pdf_path: PDF文件路径

        Returns:
            ParsedAnnualReport: 解析后的年报数据
        """
        try:
            self.logger.info(f"开始解析PDF: {pdf_path}")

            # 提取基本信息
            metadata = self._extract_metadata(pdf_path)

            # 提取全文
            full_text = self._extract_full_text(pdf_path)

            # 提取章节
            sections = self._extract_sections(pdf_path)

            # 提取表格
            tables = self._extract_tables(pdf_path)

            parsed_report = ParsedAnnualReport(
                report_id=metadata.get('report_id', ''),
                company_name=metadata.get('company_name', ''),
                stock_code=metadata.get('stock_code', ''),
                year=metadata.get('year', 2023),
                full_text=full_text,
                sections=sections,
                tables=tables,
                metadata=metadata
            )

            self.logger.info(f"PDF解析完成: {metadata.get('company_name', '')}")
            return parsed_report

        except Exception as e:
            self.logger.error(f"PDF解析失败: {str(e)}")
            raise

    def _extract_metadata(self, pdf_path: str) -> Dict:
        """从文件名和PDF内容提取元数据"""
        filename = Path(pdf_path).stem

        metadata = {
            'report_id': filename,
            'source_file': pdf_path
        }

        # 尝试从文件名提取股票代码和年份
        # 文件名格式示例: 600519_2023.pdf 或 贵州茅台_2023年报.pdf
        if '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 2:
                metadata['stock_code'] = parts[0]
                if parts[1].isdigit() or parts[1][:4].isdigit():
                    metadata['year'] = int(parts[1][:4])
                else:
                    metadata['company_name'] = parts[0]

        # 尝试从PDF第一页提取公司名称
        with pdfplumber.open(pdf_path) as pdf:
            first_page = pdf.pages[0]
            text = first_page.extract_text() or ''

            # 提取公司名称（通常是第一页的大标题）
            company_patterns = [
                r'([^。\n]{2,20})股份有限公司',
                r'([^。\n]{2,20})有限公司',
                r'股票代码[：:]\s*(\d{6})',
                r'股票简称[：:]\s*([^\s]+)'
            ]

            for pattern in company_patterns:
                match = re.search(pattern, text)
                if match:
                    if '公司' in pattern:
                        metadata['company_name'] = match.group(1)
                    elif '代码' in pattern:
                        metadata['stock_code'] = match.group(1)
                    elif '简称' in pattern:
                        metadata['company_name'] = match.group(1)

        return metadata

    def _extract_full_text(self, pdf_path: str) -> str:
        """提取PDF全文"""
        text_parts = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                except Exception as e:
                    self.logger.warning(f"第{page_num + 1}页提取失败: {str(e)}")
                    continue

        return '\n'.join(text_parts)

    def _extract_sections(self, pdf_path: str) -> List[PDFSection]:
        """提取年报章节"""
        sections = []

        with pdfplumber.open(pdf_path) as pdf:
            current_section = None
            section_content = []
            start_page = 0

            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ''

                # 检测章节标题
                for section_title in self.COMMON_SECTIONS:
                    if section_title in text:
                        # 保存上一个章节
                        if current_section and section_content:
                            sections.append(PDFSection(
                                title=current_section,
                                content='\n'.join(section_content),
                                page_range=(start_page, page_num)
                            ))

                        # 开始新章节
                        current_section = section_title
                        section_content = []
                        start_page = page_num
                        break

                if current_section:
                    section_content.append(text)

            # 保存最后一个章节
            if current_section and section_content:
                sections.append(PDFSection(
                    title=current_section,
                    content='\n'.join(section_content),
                    page_range=(start_page, len(pdf.pages))
                ))

        return sections

    def _extract_tables(self, pdf_path: str) -> List[PDFTable]:
        """提取PDF中的表格"""
        tables = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()

                if page_tables:
                    for i, table in enumerate(page_tables):
                        if table and len(table) > 1:
                            # 第一行作为表头
                            headers = [str(cell) if cell else '' for cell in table[0]]

                            # 剩余行作为数据
                            rows = []
                            for row in table[1:]:
                                if row:
                                    rows.append([str(cell) if cell else '' for cell in row])

                            tables.append(PDFTable(
                                title=f"表格_{page_num + 1}_{i + 1}",
                                headers=headers,
                                rows=rows,
                                page=page_num + 1
                            ))

        self.logger.info(f"共提取 {len(tables)} 个表格")
        return tables

    def parse_batch(self, pdf_dir: str) -> List[ParsedAnnualReport]:
        """
        批量解析PDF文件

        Args:
            pdf_dir: PDF文件目录

        Returns:
            List[ParsedAnnualReport]: 解析结果列表
        """
        pdf_dir = Path(pdf_dir)
        pdf_files = list(pdf_dir.glob("*.pdf"))

        if not pdf_files:
            self.logger.warning(f"目录中没有找到PDF文件: {pdf_dir}")
            return []

        self.logger.info(f"找到 {len(pdf_files)} 个PDF文件")

        results = []
        for pdf_file in pdf_files:
            try:
                result = self.parse_pdf(str(pdf_file))
                results.append(result)
            except Exception as e:
                self.logger.error(f"解析文件失败 {pdf_file.name}: {str(e)}")
                continue

        self.logger.info(f"成功解析 {len(results)}/{len(pdf_files)} 个文件")
        return results

    def save_to_json(self, parsed_report: ParsedAnnualReport, output_path: str):
        """将解析结果保存为JSON文件"""
        output_data = {
            'report_id': parsed_report.report_id,
            'company_name': parsed_report.company_name,
            'stock_code': parsed_report.stock_code,
            'year': parsed_report.year,
            'full_text': parsed_report.full_text,
            'sections': [
                {
                    'title': section.title,
                    'content': section.content,
                    'page_range': section.page_range
                }
                for section in parsed_report.sections
            ],
            'tables': [
                {
                    'title': table.title,
                    'headers': table.headers,
                    'rows': table.rows,
                    'page': table.page
                }
                for table in parsed_report.tables
            ],
            'metadata': parsed_report.metadata
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"结果已保存到: {output_path}")


def main():
    """测试代码"""
    parser = PDFParser()

    # 测试单个文件解析
    # parsed_report = parser.parse_pdf("data/raw/600519_2023.pdf")
    # parser.save_to_json(parsed_report, "data/processed/600519_2023.json")

    # 测试批量解析
    # results = parser.parse_batch("data/raw")
    # for result in results:
    #     parser.save_to_json(result, f"data/processed/{result.report_id}.json")


if __name__ == "__main__":
    main()
