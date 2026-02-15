# OpenLLM - Large Language Model Vertical Domain Fine-tuning Projects

A collection of 10 commercial-ready LLM fine-tuning projects for vertical domains using Python+PyTorch.

## Project Overview

This repository contains 10 LLM fine-tuning projects targeting different vertical domains, strictly following PEFT technical standards and focusing on highly regulated industries such as finance, healthcare, manufacturing, and legal.

### Project List

1. **[Financial Compliance Review Fine-tuning System](./10-financial-compliance-finetune)** - Yin-Yang contract detection, regulatory clause mapping
2. **[Medical AI Misdiagnosis Rate Optimization System](./11-medical-diagnosis-finetune)** - Multimodal diagnosis, traffic light alert system
3. **[Manufacturing Federated Learning Quality Inspection System](./12-manufacturing-quality-finetune)** - Cross-factory QC, edge fine-tuning
4. **[Legal Contract Intelligent Review System](./13-legal-contract-finetune)** - Clause conflict detection, Level 3 compliance
5. **[Medical AI Medical Insurance Compliance System](./14-medical-insurance-finetune)** - Real-time violation alerts, medical insurance policy adaptation
6. **[Financial AI Anti-Fraud System](./15-financial-anti-fraud-finetune)** - AIGC document authentication, biometric anti-fraud
7. **[Legal AI Multi-language Compliance System](./16-legal-multilang-finetune)** - Multi-language contract review, risk quantification
8. **[Manufacturing Predictive Maintenance System](./17-manufacturing-prediction-finetune)** - Time-series fault prediction, maintenance plan optimization
9. **[Medical AI Primary Care Capability Enhancement System](./18-medical-primarycare-finetune)** - Double-blind diagnosis, device heterogeneity adaptation
10. **[Financial Algorithm Fairness System](./19-financial-fairness-finetune)** - Fairness preference injection, real-time monitoring

## Technical Features

### Fine-tuning Techniques
- **LoRA/QLoRA**: Low-Rank Adaptation, reducing fine-tuning cost to under $200
- **DAPT**: Domain Adaptive Pre-training
- **SFT**: Supervised Fine-tuning
- **DPO/DKPO**: Direct Preference Optimization / Debiasing Knowledge Preference Optimization
- **RAG-Finetuning**: Retrieval-Augmented Fine-tuning
- **Federated Learning**: Collaborative training without data leaving the domain

### Supported Base Models
- ERNIE Finance (ERNIE)
- Qwen3-VL-8B-Thinking (Multimodal)
- Qwen-7B/14B-Chat
- Llama2-70B
- ChatGLM3-6B

## Quick Start

### Choose a project and enter the directory

```bash
cd 10-financial-compliance-finetune  # or other projects
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Prepare data

Place training data in the `data/` directory, following the format specified in each project's configuration file.

### Train the model

```bash
python scripts/train.py --config configs/config.yaml --stage all
```

### Inference testing

```bash
python scripts/infer.py --model_path models/model --input input.txt
```

## Documentation

- [Project Overview Document](01-docs/PROJECT_OVERVIEW.md) - Complete tech stack, commercialization path, market size
- [Individual Project README](./XX-project-name/README.md) - Detailed installation and usage instructions
- [Design Document](./01-docs/大模型垂直领域微调业务系统设计.md) - Complete design specifications for 10 systems (Chinese)
- [README_CN.md](./README_CN.md) - Chinese version of this README

## Tech Stack

```yaml
Core Frameworks:
  - PyTorch: >= 2.0.0
  - Transformers: >= 4.35.0
  - PEFT: >= 0.7.0
  - BitsAndBytes: >= 0.41.0

Fine-tuning Techniques:
  - LoRA/QLoRA
  - DAPT
  - SFT
  - DPO/DKPO
  - RAG-Finetuning

Distributed Training:
  - Accelerate
  - Flower (Federated Learning)
  - DeepSpeed

Data Processing:
  - Datasets
  - MONAI (Medical Imaging)
  - OpenCV
```

## Project Structure

Each project follows a unified directory structure:

```
XX-project-name/
├── configs/              # Configuration files
├── data/                 # Training data
├── models/               # Model outputs
├── scripts/              # Training and inference scripts
├── src/
│   ├── modules/          # Specialized modules
│   ├── model.py          # Model definitions
│   ├── trainer.py        # Trainer
│   └── data_processor.py # Data processing
└── requirements.txt      # Dependencies
```

## Evaluation Metrics

| Domain | Core Metric | Target Value |
|--------|-------------|--------------|
| Finance | Clause Recall Rate | > 98% |
| Finance | Fairness Bias | < 0.5% |
| Healthcare | Primary Care Misdiagnosis Reduction | 37% |
| Healthcare | Diagnostic Accuracy | > 92% |
| Manufacturing | QC Accuracy | > 99.5% |
| Manufacturing | Fault Prediction Accuracy | > 96% |
| Legal | Contract Review Accuracy | > 99% |
| Legal | Multi-language Translation Accuracy | > 99% |

## Market Size (2026 Forecast)

- Financial Compliance AI: $47B (CAGR 38%)
- Medical AI Diagnosis: $58B (CAGR 45%)
- Industrial AI Predictive Maintenance: $120B (CAGR 30%)
- Legal AI Contract Review: $280B (CAGR 35%)

## Important Notes

1. **Data Security**: All projects support Level 3 compliance certification
2. **Privacy Protection**: Federated learning ensures data never leaves the domain
3. **Compliance Requirements**: Meets relevant industry regulatory standards
4. **Hardware Requirements**: A100/H100 GPU recommended for training

## License

See [LICENSE](./LICENSE) file.

## Contributing

Issues and Pull Requests are welcome!

## Commercialization Status

These projects are designed for commercial deployment with:

✅ Complete fine-tuning pipelines
✅ Clear evaluation metrics
✅ Specialized domain modules
✅ Defined target customers
✅ Multiple pricing models
✅ Market size forecasts
✅ Compliance certifications (Level 3)
✅ Privacy protection mechanisms
