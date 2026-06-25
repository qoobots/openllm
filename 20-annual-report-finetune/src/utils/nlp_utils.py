"""
NLP工具函数
"""

import jieba
import jieba.posseg as pseg
from typing import List, Tuple
from collections import Counter


class NLPUtils:
    """NLP工具类"""

    def __init__(self):
        # 添加金融领域词典
        self._add_financial_words()

    def _add_financial_words(self):
        """添加金融领域专业词汇"""
        financial_words = [
            '净利润', '毛利率', '净利率', '净资产收益率',
            '资产负债率', '流动比率', '速动比率',
            '每股收益', '市盈率', '市净率',
            '营业收入', '营业利润', '利润总额',
            '经营活动现金流', '投资活动现金流', '筹资活动现金流',
            '应收账款', '存货', '固定资产', '无形资产'
        ]

        for word in financial_words:
            jieba.add_word(word)

    def extract_keywords(self, text: str, top_k: int = 20) -> List[Tuple[str, int]]:
        """
        提取关键词（基于词频）

        Args:
            text: 输入文本
            top_k: 返回前k个关键词

        Returns:
            List[Tuple[str, int]]: (关键词, 词频)
        """
        # 分词
        words = jieba.cut(text)

        # 过滤停用词和短词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就',
                    '不', '人', '都', '一', '一个', '上', '也', '很',
                    '到', '说', '去', '你', '会', '着', '没有', '看'}

        filtered_words = [
            word for word in words
            if len(word) > 1 and word not in stop_words and word.strip()
        ]

        # 统计词频
        word_freq = Counter(filtered_words)

        return word_freq.most_common(top_k)

    def extract_nouns(self, text: str) -> List[str]:
        """
        提取名词

        Args:
            text: 输入文本

        Returns:
            List[str]: 名词列表
        """
        words = pseg.cut(text)
        nouns = [word for word, flag in words if flag.startswith('n')]
        return nouns

    def extract_numerical_entities(self, text: str) -> List[dict]:
        """
        提取数值实体

        Args:
            text: 输入文本

        Returns:
            List[dict]: 数值实体列表
        """
        import re

        entities = []

        # 提取数字 + 单位
        patterns = [
            (r'(\d+\.?\d*)\s*(元|万元|亿元|万元|万美元|亿美元)', '金额'),
            (r'(\d+\.?\d*)\s*%|(\d+\.?\d*)\s*个百分点', '百分比'),
            (r'(\d{4})\s*年', '年份'),
            (r'(\d{1,2})\s*月', '月份')
        ]

        for pattern, entity_type in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                value = match[0] if match[0] else match[1]
                entities.append({
                    'value': value,
                    'type': entity_type
                })

        return entities


def main():
    """测试代码"""
    nlp_utils = NLPUtils()

    text = "公司2023年实现营业收入1500亿元，净利润300亿元，同比增长20%。"

    keywords = nlp_utils.extract_keywords(text)
    print("关键词:", keywords)

    entities = nlp_utils.extract_numerical_entities(text)
    print("数值实体:", entities)


if __name__ == "__main__":
    main()
