# 制造业设备预测性维护微调系统

基于Llama2-70B + QLoRA的预测性维护系统，减少设备停机损失。

## 系统特性

- **多模态设备监控**: 振动/温度/电流多维度数据
- **时序故障预测**: TFT时序模型提前15天预警
- **联邦学习知识库**: 跨工厂共享维护经验
- **维护计划优化**: 自动生成最优维护方案

## 微调路径

DAPT → SFT → 联邦学习

## 快速开始

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/config.yaml
```

## 评估指标

- 故障预测准确率 > 96%
- 预测提前时间 > 15天
- 设备停机时间减少 > 40%
