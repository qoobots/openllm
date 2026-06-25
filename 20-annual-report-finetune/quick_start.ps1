# 年报分析微调系统 - Windows 快速启动脚本

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  上市公司年报分析微调系统 - 快速启动" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 创建必要的目录
Write-Host "1. 创建目录结构..." -ForegroundColor Yellow
$directories = @(
    "data/raw",
    "data/processed/parsed",
    "data/processed/structured",
    "data/processed/invalid",
    "data/annotated",
    "models/checkpoints",
    "models/dapt_model",
    "models/sft_model",
    "models/annual_report_model",
    "logs"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "✓ 目录创建完成" -ForegroundColor Green
Write-Host ""

# 生成示例训练数据
Write-Host "2. 生成示例训练数据..." -ForegroundColor Yellow
python scripts/generate_sample_data.py --output_path data/processed/sft_train.jsonl --num_companies 50 --samples_per_company 15
Write-Host ""

# 检查生成的数据
Write-Host "3. 检查生成的数据..." -ForegroundColor Yellow
if (Test-Path "data/processed/sft_train.jsonl") {
    Write-Host "✓ 训练数据文件已生成" -ForegroundColor Green
    $fileInfo = Get-Item "data/processed/sft_train.jsonl"
    $lineCount = (Get-Content "data/processed/sft_train.jsonl").Count
    Write-Host "  文件大小: $($fileInfo.Length) bytes"
    Write-Host "  样本数量: $lineCount 行"
} else {
    Write-Host "✗ 训练数据文件未生成" -ForegroundColor Red
}
Write-Host ""

# 显示快速开始指南
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  快速开始指南" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. 处理真实的PDF年报文件:" -ForegroundColor White
Write-Host "   python scripts/process_pdfs.py --input_dir data/raw --output_dir data/processed" -ForegroundColor Gray
Write-Host ""
Write-Host "2. 启动数据标注工具:" -ForegroundColor White
Write-Host "   streamlit run annotation_tool/app.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. 训练模型（需要先安装依赖）:" -ForegroundColor White
Write-Host "   python scripts/train.py --config configs/config.yaml --stage sft" -ForegroundColor Gray
Write-Host ""
Write-Host "4. 使用模型进行推理:" -ForegroundColor White
Write-Host "   python scripts/infer.py --model_path models/annual_report_model --report_file data/processed/xxx.json --mode report" -ForegroundColor Gray
Write-Host ""
Write-Host "5. 交互式问答:" -ForegroundColor White
Write-Host "   python scripts/qa.py --model_path models/annual_report_model --report_file data/processed/xxx.json" -ForegroundColor Gray
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  详细文档请查看 README.md" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
