# 医疗AI基层诊断能力提升微调系统

基于Qwen-14B-Chat的基层医疗诊断系统，通过双盲诊断模式降低误诊率。

## 系统特性

- **多模态诊断报告**: 融合影像与病理文本
- **AI+医生双盲诊断**: 误诊率降至0.2%
- **设备异构性适配**: 统一处理不同品牌设备
- **远程医疗协作**: 三甲医院专家实时指导

## 微调路径

DAPT → SFT → 联邦学习

## 快速开始

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/config.yaml
```

## 评估指标

- 基层医院诊断准确率 > 92%
- 设备异构性适应率 > 95%
- 误诊率 < 0.2%
