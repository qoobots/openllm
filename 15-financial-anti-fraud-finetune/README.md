# 金融AI反欺诈微调系统

基于文心一言金融版的金融反欺诈系统，支持多模态风控。

## 系统特性

- **AIGC文档鉴伪**: 检测隐形水印和深度伪造
- **生物特征反欺诈**: 人脸识别与证件OCR比对
- **金融知识图谱**: 12万+法规、300万+判例

## 微调路径

DAPT → SFT → DPO

## 快速开始

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/config.yaml
```

## 评估指标

- 欺诈检测率 > 99%
- 误识率 < 0.1%
- 文档合规率 > 99.5%
