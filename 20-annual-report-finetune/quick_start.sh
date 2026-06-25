#!/bin/bash
# 快速启动脚本 - 年报分析微调系统

echo "=========================================="
echo "  上市公司年报分析微调系统 - 快速启动"
echo "=========================================="
echo ""

# 创建必要的目录
echo "1. 创建目录结构..."
mkdir -p data/raw data/processed/parsed data/processed/structured data/processed/invalid
mkdir -p data/annotated models/checkpoints models/dapt_model models/sft_model models/annual_report_model
mkdir -p logs
echo "✓ 目录创建完成"
echo ""

# 生成示例训练数据
echo "2. 生成示例训练数据..."
python scripts/generate_sample_data.py \
    --output_path data/processed/sft_train.jsonl \
    --num_companies 50 \
    --samples_per_company 15
echo ""

# 检查生成的数据
echo "3. 检查生成的数据..."
if [ -f "data/processed/sft_train.jsonl" ]; then
    echo "✓ 训练数据文件已生成"
    file_size=$(wc -c < data/processed/sft_train.jsonl)
    line_count=$(wc -l < data/processed/sft_train.jsonl)
    echo "  文件大小: $file_size bytes"
    echo "  样本数量: $line_count 行"
else
    echo "✗ 训练数据文件未生成"
fi
echo ""

# 显示快速开始指南
echo "=========================================="
echo "  快速开始指南"
echo "=========================================="
echo ""
echo "1. 处理真实的PDF年报文件:"
echo "   python scripts/process_pdfs.py --input_dir data/raw --output_dir data/processed"
echo ""
echo "2. 启动数据标注工具:"
echo "   streamlit run annotation_tool/app.py"
echo ""
echo "3. 训练模型（需要先安装依赖）:"
echo "   python scripts/train.py --config configs/config.yaml --stage sft"
echo ""
echo "4. 使用模型进行推理:"
echo "   python scripts/infer.py --model_path models/annual_report_model --report_file data/processed/xxx.json --mode report"
echo ""
echo "5. 交互式问答:"
echo "   python scripts/qa.py --model_path models/annual_report_model --report_file data/processed/xxx.json"
echo ""
echo "=========================================="
echo "  详细文档请查看 README.md"
echo "=========================================="
