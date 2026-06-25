"""
年报数据标注工具
基于Streamlit的Web标注界面，支持半自动标注
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st
from loguru import logger
import pandas as pd

# 设置页面配置
st.set_page_config(
    page_title="年报数据标注工具",
    page_icon="📊",
    layout="wide"
)


class AnnotationTool:
    """标注工具类"""

    def __init__(self):
        self.data_dir = Path("data/processed/parsed")
        self.annotated_dir = Path("data/annotated")
        self.current_data = []
        self.current_index = 0
        self.annotations = {}

        # 初始化目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """加载待标注数据"""
        json_files = list(self.data_dir.glob("*.json"))

        if not json_files:
            st.warning(f"在 {self.data_dir} 中没有找到待标注数据")
            return []

        data = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    report['source_file'] = str(json_file)
                    data.append(report)
            except Exception as e:
                logger.error(f"加载文件失败 {json_file}: {str(e)}")
                continue

        return data

    def save_annotation(self, report_id: str, annotation: Dict):
        """保存标注"""
        self.annotated_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.annotated_dir / f"{report_id}_annotated.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)

        logger.info(f"标注已保存: {output_path}")

    def auto_annotate(self, report: Dict) -> Dict:
        """
        自动标注（基于规则和简单模型）

        Args:
            report: 年报数据

        Returns:
            Dict: 自动标注结果
        """
        annotation = {
            'report_id': report['report_id'],
            'company_name': report.get('company_name', ''),
            'stock_code': report.get('stock_code', ''),
            'year': report.get('year', 2023),
            'financial_indicators': {},
            'business_segments': [],
            'risks': [],
            'analyst_comments': '',
            'rating': '',
            'tags': []
        }

        # 提取财务指标（简化版）
        full_text = report.get('full_text', '')

        # 提取收入
        if '营业收入' in full_text or '营业总收入' in full_text:
            import re
            match = re.search(r'(营业收入|营业总收入)[：:\s]*([0-9,.]+)[亿万]*元', full_text)
            if match:
                value = match.group(2).replace(',', '')
                annotation['financial_indicators']['revenue'] = {
                    'name': '营业收入',
                    'value': float(value) if value else 0,
                    'confidence': 0.8
                }

        # 提取净利润
        if '净利润' in full_text:
            match = re.search(r'净利润[：:\s]*([0-9,.]+)[亿万]*元', full_text)
            if match:
                value = match.group(1).replace(',', '')
                annotation['financial_indicators']['net_profit'] = {
                    'name': '净利润',
                    'value': float(value) if value else 0,
                    'confidence': 0.8
                }

        # 提取风险关键词
        risk_keywords = ['政策风险', '市场风险', '经营风险', '财务风险',
                       '技术风险', '管理风险', '法律风险', '信用风险']

        for keyword in risk_keywords:
            if keyword in full_text:
                annotation['risks'].append({
                    'type': keyword,
                    'description': f"检测到{keyword}相关内容",
                    'severity': 'medium',
                    'confidence': 0.7
                })

        # 提取亮点
        if '增长' in full_text or '提升' in full_text:
            annotation['tags'].extend(['增长', '改善'])

        return annotation


def main():
    """主函数"""
    st.title("📊 上市公司年报数据标注工具")

    # 初始化session state
    if 'tool' not in st.session_state:
        st.session_state.tool = AnnotationTool()
        st.session_state.data = st.session_state.tool.load_data()
        st.session_state.current_index = 0
        st.session_state.auto_annotation = None

    tool = st.session_state.tool
    data = st.session_state.data

    # 侧边栏
    with st.sidebar:
        st.header("📁 数据管理")

        # 数据统计
        st.metric("待标注数据", len(data))
        if data:
            progress = (st.session_state.current_index / len(data)) * 100
            st.metric("进度", f"{progress:.1f}%")

        st.divider()

        # 加载数据按钮
        if st.button("🔄 重新加载数据"):
            st.session_state.data = tool.load_data()
            st.session_state.current_index = 0
            st.rerun()

        st.divider()

        # 导出标注
        if st.button("📤 导出所有标注"):
            annotated_files = list(tool.annotated_dir.glob("*_annotated.json"))
            if annotated_files:
                all_annotations = []
                for file in annotated_files:
                    with open(file, 'r', encoding='utf-8') as f:
                        all_annotations.append(json.load(f))

                export_path = tool.annotated_dir / "all_annotations.json"
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(all_annotations, f, ensure_ascii=False, indent=2)

                st.success(f"已导出 {len(all_annotations)} 条标注到 {export_path}")
            else:
                st.warning("暂无标注数据")

    # 主内容区
    if not data:
        st.info("👈 请先将处理后的年报数据放入 `data/processed/parsed/` 目录，然后点击"重新加载数据"")
        return

    # 检查索引是否越界
    if st.session_state.current_index >= len(data):
        st.success("🎉 所有数据已标注完成！")
        return

    current_report = data[st.session_state.current_index]

    # 显示当前进度
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        st.metric("当前报告", current_report.get('company_name', 'Unknown'))
    with col2:
        st.metric("股票代码", current_report.get('stock_code', 'N/A'))
    with col3:
        st.metric("年份", current_report.get('year', 2023))

    st.divider()

    # 标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📄 年报内容", "📊 财务指标", "⚠️ 风险分析", "📝 分析师点评"])

    with tab1:
        # 显示年报基本信息
        st.subheader("年报基本信息")

        col1, col2 = st.columns(2)
        with col1:
            st.text_input("公司名称", value=current_report.get('company_name', ''),
                         key=f"company_{st.session_state.current_index}")
            st.text_input("股票代码", value=current_report.get('stock_code', ''),
                         key=f"stock_{st.session_state.current_index}")
            st.number_input("年份", value=current_report.get('year', 2023),
                           key=f"year_{st.session_state.current_index}")

        # 显示章节
        st.subheader("年报章节")
        sections = current_report.get('sections', [])
        if sections:
            for section in sections[:3]:  # 只显示前3个章节
                with st.expander(f"📑 {section['title']}"):
                    st.text_area("", value=section['content'][:500] + "...",
                                height=150, disabled=True)
        else:
            st.info("未找到章节信息")

        # 显示表格
        st.subheader("表格数据")
        tables = current_report.get('tables', [])
        if tables:
            for i, table in enumerate(tables[:2]):  # 只显示前2个表格
                with st.expander(f"📋 {table['title']}"):
                    if table['headers'] and table['rows']:
                        df = pd.DataFrame(table['rows'], columns=table['headers'])
                        st.dataframe(df)
        else:
            st.info("未找到表格数据")

    with tab2:
        st.subheader("财务指标标注")

        # 自动标注按钮
        if st.button("🤖 自动提取财务指标"):
            auto_annotation = tool.auto_annotate(current_report)
            st.session_state.auto_annotation = auto_annotation
            st.success("自动提取完成！请检查并修正结果")

        # 显示/编辑财务指标
        financial_data = st.session_state.auto_annotation.get('financial_indicators', {}) \
            if st.session_state.auto_annotation else {}

        indicators = [
            ('营业收入', 'revenue'),
            ('净利润', 'net_profit'),
            ('毛利率', 'gross_margin'),
            ('净资产收益率', 'roe'),
            ('资产负债率', 'debt_ratio'),
            ('每股收益', 'eps')
        ]

        col1, col2 = st.columns(2)
        for i, (name_cn, key) in enumerate(indicators):
            with col1 if i % 2 == 0 else col2:
                value = financial_data.get(key, {}).get('value', 0) if financial_data else 0
                st.number_input(
                    f"{name_cn} (元/%)",
                    value=float(value) if isinstance(value, (int, float)) else 0.0,
                    key=f"indicator_{st.session_state.current_index}_{key}"
                )

    with tab3:
        st.subheader("风险分析标注")

        risks = st.session_state.auto_annotation.get('risks', []) \
            if st.session_state.auto_annotation else []

        # 显示已有风险
        if risks:
            for i, risk in enumerate(risks):
                with st.expander(f"⚠️ {risk['type']}", expanded=i < 2):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.text_area("风险描述",
                                    value=risk.get('description', ''),
                                    key=f"risk_desc_{st.session_state.current_index}_{i}")
                    with col2:
                        st.selectbox("严重程度",
                                    options=['low', 'medium', 'high'],
                                    index=['low', 'medium', 'high'].index(risk.get('severity', 'medium')),
                                    key=f"risk_severity_{st.session_state.current_index}_{i}")

        # 添加新风险
        st.divider()
        st.subheader("添加新风险")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            new_risk_type = st.text_input("风险类型", key="new_risk_type")
            new_risk_desc = st.text_area("风险描述", key="new_risk_desc")
        with col3:
            new_risk_severity = st.selectbox("严重程度",
                                           options=['low', 'medium', 'high'],
                                           key="new_risk_severity")

    with tab4:
        st.subheader("分析师点评")

        analyst_comments = st.text_area(
            "分析师点评",
            placeholder="请输入对该公司的分析评价...",
            height=200,
            key=f"comments_{st.session_state.current_index}"
        )

        st.subheader("评级")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.selectbox("推荐评级",
                        options=['强力买入', '买入', '持有', '卖出', '强力卖出'],
                        key=f"rating_{st.session_state.current_index}")

        with col2:
            st.multiselect("标签",
                          options=['高增长', '稳定', '低估值', '高股息', '龙头', '转型', '风险'],
                          key=f"tags_{st.session_state.current_index}")

        with col3:
            st.slider("信心度", 1, 5, 3,
                     key=f"confidence_{st.session_state.current_index}")

    st.divider()

    # 导航按钮
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("⬅️ 上一条", disabled=st.session_state.current_index == 0):
            st.session_state.current_index -= 1
            st.session_state.auto_annotation = None
            st.rerun()

    with col2:
        if st.button("💾 保存当前标注"):
            # 收集所有标注
            annotation = {
                'report_id': current_report['report_id'],
                'company_name': st.session_state[f"company_{st.session_state.current_index}"],
                'stock_code': st.session_state[f"stock_{st.session_state.current_index}"],
                'year': st.session_state[f"year_{st.session_state.current_index}"],
                'analyst_comments': analyst_comments,
                'rating': st.session_state[f"rating_{st.session_state.current_index}"],
                'tags': st.session_state[f"tags_{st.session_state.current_index}"],
                'confidence': st.session_state[f"confidence_{st.session_state.current_index}"]
            }

            tool.save_annotation(current_report['report_id'], annotation)
            st.success("标注已保存！")

    with col3:
        if st.button("下一条 ➡️"):
            # 先保存当前标注
            annotation = {
                'report_id': current_report['report_id'],
                'company_name': st.session_state[f"company_{st.session_state.current_index}"],
                'stock_code': st.session_state[f"stock_{st.session_state.current_index}"],
                'year': st.session_state[f"year_{st.session_state.current_index}"],
                'analyst_comments': analyst_comments,
                'rating': st.session_state[f"rating_{st.session_state.current_index}"],
                'tags': st.session_state[f"tags_{st.session_state.current_index}"],
                'confidence': st.session_state[f"confidence_{st.session_state.current_index}"]
            }
            tool.save_annotation(current_report['report_id'], annotation)

            # 跳到下一条
            st.session_state.current_index += 1
            st.session_state.auto_annotation = None
            st.rerun()


if __name__ == "__main__":
    main()
