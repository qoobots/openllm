# 制造业联邦学习质检微调系统

基于Llama2-70B + QLoRA量化技术的制造质检系统，采用联邦学习实现跨工厂数据协作。

## 系统特性

- **联邦学习质检**: 跨工厂数据协作，数据不出域
- **轻量化边缘微调**: 支持设备端实时更新
- **缺陷分类器**: 高精度缺陷识别

## 微调路径

DAPT → SFT → 联邦学习

## 快速开始

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/config.yaml
```

## 评估指标

- 质检准确率 > 99.5%
- 跨工厂泛化能力提升30%
