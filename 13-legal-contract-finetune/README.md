# 法律合同智能审查微调系统

基于ChatGLM3-6B的法律合同审查系统，满足等保三级认证要求。

## 系统特性

- **智能预审模块**: NLP深度解析风险点
- **条款冲突检测引擎**: 识别法律体系冲突
- **等保三级合规工作流**: 数据脱敏、审计日志
- **合规复核工作流**: 支持双人复核

## 微调路径

DAPT → SFT → DPO

## 快速开始

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/config.yaml
```

## 评估指标

- 合同审查准确率 > 99%
- 隐私保护等级: 等保三级
