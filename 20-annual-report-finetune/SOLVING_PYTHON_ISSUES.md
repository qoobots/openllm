# Python 环境问题解决方案

## 错误: "请选择包含有效 Python SDK 的模块"

这个错误通常发生在 VS Code 或 IntelliJ IDEA 中，原因是 IDE 没有正确识别 Python 环境。

---

## IntelliJ IDEA 专用解决方案

### 方案 1: 配置 Python SDK

1. **打开设置**
   - Windows: `File` → `Settings` → `Project: annual-report-finetune` → `Python Interpreter`
   - Mac: `IntelliJ IDEA` → `Preferences` → `Project: annual-report-finetune` → `Python Interpreter`

2. **添加 Python 解释器**
   - 点击 "Add Interpreter" → "Add Local Interpreter"
   - 选择 "Virtual Environment" 或 "System Interpreter"
   - 浏览并选择您的 Python 安装路径

3. **验证配置**
   - 确保选择的 Python 版本 >= 3.8
   - 点击 "OK" 保存配置

### 方案 2: 使用 IDEA 终端运行（最简单）

1. 在 IDEA 底部点击 "Terminal" 标签
2. 运行以下命令：

```powershell
# 生成示例数据
python scripts\generate_sample_data.py --output_path data\processed\sft_train.jsonl --num_companies 50 --samples_per_company 15
```

### 方案 3: 配置运行配置

1. 点击右上角的运行配置下拉框
2. 选择 "Edit Configurations..."
3. 点击 "+" 号，选择 "Python"
4. 配置如下：
   - **Name**: 生成示例数据
   - **Script path**: `$PROJECT_DIR$/scripts/generate_sample_data.py`
   - **Parameters**: `--output_path data/processed/sft_train.jsonl --num_companies 50 --samples_per_company 15`
   - **Python interpreter**: 选择已配置的解释器
5. 点击 "OK"

### IDEA 常见 Python 路径

- `C:\Python310\python.exe`
- `C:\Python39\python.exe`
- `C:\Users\{用户名}\AppData\Local\Programs\Python\Python310\python.exe`

---

## VS Code 解决方案

## 解决方案

### 方案 1: 在 VS Code 中选择 Python 解释器

1. 在 VS Code 中按 `Ctrl + Shift + P` (Windows) 或 `Cmd + Shift + P` (Mac)
2. 输入并选择 "Python: Select Interpreter"
3. 从列表中选择一个 Python 解释器，或点击 "Enter interpreter path..." 选择已安装的 Python

### 方案 2: 安装 Python 扩展

确保已安装 VS Code Python 扩展：
1. 打开扩展面板 (`Ctrl + Shift + X`)
2. 搜索 "Python"
3. 安装由 Microsoft 提供的官方 Python 扩展

### 方案 3: 在命令行中使用（推荐）

如果 VS Code 环境配置有问题，可以直接在命令行中运行：

#### Windows (PowerShell)

```powershell
# 进入项目目录
cd d:\05workspaces\openllm\20-annual-report-finetune

# 生成示例数据
python scripts/generate_sample_data.py --output_path data/processed/sft_train.jsonl --num_companies 50 --samples_per_company 15
```

#### Windows (CMD)

```cmd
cd /d d:\05workspaces\openllm\20-annual-report-finetune
python scripts\generate_sample_data.py --output_path data/processed\sft_train.jsonl --num_companies 50 --samples_per_company 15
```

### 方案 4: 运行快速启动脚本

我已创建了快速启动脚本，可以直接运行：

#### Windows PowerShell

```powershell
cd d:\05workspaces\openllm\20-annual-report-finetune
.\quick_start.ps1
```

#### Linux/Mac

```bash
cd /path/to/openllm/20-annual-report-finetune
chmod +x quick_start.sh
./quick_start.sh
```

## 检查 Python 环境

运行以下命令检查 Python 是否正确安装：

```bash
# 检查 Python 版本
python --version

# 检查 Python 路径
where python  # Windows
which python  # Linux/Mac

# 检查 pip
pip --version
```

## 安装依赖

如果依赖未安装，运行：

```bash
pip install -r requirements.txt
```

## 常见问题

### Q: 找不到 python 命令

**A:** Python 可能未安装或未添加到 PATH。从 https://www.python.org/downloads/ 下载并安装 Python，安装时勾选 "Add Python to PATH"。

### Q: 权限错误

**A:** 可能需要管理员权限。以管理员身份运行 PowerShell/CMD，或使用虚拟环境：

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Q: VS Code 仍然提示选择 Python SDK

**A:** 重启 VS Code，或尝试：
1. 删除 `.vscode` 文件夹
2. 重新打开项目
3. 再次选择 Python 解释器

## 验证安装

运行以下命令验证环境是否配置正确：

```bash
python scripts/generate_sample_data.py --output_data test.jsonl --num_companies 1 --samples_per_company 1
```

如果成功生成数据，说明环境配置正确。
