from setuptools import setup, find_packages

setup(
    name="annual-report-finetune",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.35.0',
        'peft>=0.6.0',
        'accelerate>=0.24.0',
        'PyPDF2>=3.0.0',
        'pdfplumber>=0.10.0',
        'jieba>=0.42.1',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'streamlit>=1.28.0',
        'pyyaml>=6.0',
        'tqdm>=4.66.0',
        'loguru>=0.7.0',
    ],
    python_requires='>=3.8',
)
