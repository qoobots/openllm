# 法律AI多语言合规微调系统

基于Qwen-7B-Chat的多语言法律审查系统，支持20+语言。

## 系统特性

- **多语言合同审查**: 支持中英文等多语言
- **条款冲突检测**: 识别不同法律体系冲突
- **法律风险量化**: 生成风险热力图
- **碎片化加密**: 确保合同数据零出域

## 微调路径

DAPT → SFT → 联邦学习

## 快速开始

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/config.yaml
```

## 评估指标

- 多语言合同审查准确率 > 95%
- 法律术语翻译准确率 > 99%
- 合规风险预测准确率 > 98%
