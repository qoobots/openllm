"""
文本清洗工具
"""

import re
from typing import List


class TextCleaner:
    """文本清洗器"""

    def __init__(self):
        pass

    def clean(self, text: str) -> str:
        """
        清洗文本

        Args:
            text: 原始文本

        Returns:
            str: 清洗后的文本
        """
        if not text:
            return ""

        # 1. 移除多余空格和换行
        text = re.sub(r'\s+', ' ', text)

        # 2. 移除页码
        text = re.sub(r'第\s*\d+\s*页', '', text)

        # 3. 移除页眉页脚常见模式
        text = re.sub(r'(\d{4}\s*年\s*年度报告|Annual Report \d{4})', '', text)
        text = re.sub(r'(股票代码：\d+|Stock Code: \d+)', '', text)

        # 4. 移除特殊字符（保留中文、英文、数字、标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：""''（）【】《》、.,!?;:()"()\[\]]', '', text)

        # 5. 统一标点符号
        text = text.replace('，', ',').replace('。', '.').replace('！', '!')
        text = text.replace('？', '?').replace('；', ';').replace('：', ':')

        # 6. 移除重复的标点
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)

        # 7. 移除过短的行
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) > 5]

        return '\n'.join(lines)

    def clean_batch(self, texts: List[str]) -> List[str]:
        """批量清洗文本"""
        return [self.clean(text) for text in texts]


def main():
    """测试代码"""
    cleaner = TextCleaner()

    text = "  公司   2023  年度报告  \n\n\n 股票代码：600519\n第 12 页\n\n"
    cleaned = cleaner.clean(text)
    print(f"原始: {text}")
    print(f"清洗后: {cleaned}")


if __name__ == "__main__":
    main()
