# 金融合规审查微调系统

基于大模型PEFT技术的金融合同合规审查系统，采用DAPT→SFT→DPO三阶段微调路径。

## 系统特性

- **阴阳合同识别引擎**: 检测金额/日期/签名不一致
- **监管条款映射器**: 自动关联银保监2025新规
- **算法公平性监控器**: 实时监测性别/地域歧视
- **合规标识生成器**: 自动插入合规水印

## 微调路径

1. **DAPT**: 金融语料领域自适应
2. **SFT**: 合同条款监督微调
3. **DPO**: 合规偏好优化

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
# 完整训练流程 (DAPT → SFT → DPO)
python scripts/train.py --config configs/config.yaml --stage all

# 单独执行某个阶段
python scripts/train.py --config configs/config.yaml --stage sft
```

### 推理

```bash
python scripts/infer.py \
    --model_path models/financial_compliance_model \
    --contract "合同内容..." \
    --output result.txt
```

## 项目结构

```
10-financial-compliance-finetune/
├── configs/           # 配置文件
├── data/             # 训练数据
├── models/           # 模型输出
├── scripts/          # 训练和推理脚本
├── src/
│   ├── modules/      # 专用模块
│   ├── model.py      # 模型定义
│   ├── trainer.py    # 训练器
│   └── data_processor.py  # 数据处理
└── requirements.txt  # 依赖包
```

## 评估指标

- 条款召回率 > 98%
- 误报率 < 0.5%
- 合规准确率 > 99.5%
