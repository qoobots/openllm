# 金融算法公平性微调系统

基于Qwen-7B-Chat的金融公平性系统，通过DKPO技术优化模型公平性。

## 系统特性

- **公平性偏好注入器**: DKPO技术注入公平标准
- **公平性实时监控器**: 实时评估模型输出
- **联邦学习公平性评估**: 数据不出域共享结果
- **合规审计报告生成器**: 自动生成监管报告

## 微调路径

DAPT → SFT → DKPO

## 快速开始

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/config.yaml
```

## 评估指标

- 公平性偏差 < 0.5%
- 合规审计通过率 > 99%
