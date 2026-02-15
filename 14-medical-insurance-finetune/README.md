# 医疗AI医保合规微调系统

基于Qwen-14B-Chat的医疗医保合规系统，实时监控医生操作。

## 系统特性

- **实时违规预警**: 医生开处方、护士收费时实时提示
- **医保政策自动更新**: RAG-Finetuning动态知识更新
- **基层医院支持**: 提升基层医院合规能力

## 微调路径

DAPT → SFT → RAG-Finetuning

## 快速开始

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/config.yaml
```

## 评估指标

- 医保违规识别准确率 > 98%
- 实时预警响应时间 < 500ms
