"""
PDF处理脚本
批量解析年报PDF文件
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# 添加src到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processor import AnnualReportDataProcessor


def main():
    parser = argparse.ArgumentParser(description="年报PDF处理脚本")
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/pdfs',
        help='PDF文件目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='输出目录'
    )
    parser.add_argument(
        '--save_intermediate',
        action='store_true',
        default=True,
        help='是否保存中间结果'
    )

    args = parser.parse_args()

    # 初始化处理器
    logger.info("初始化数据处理器...")
    processor = AnnualReportDataProcessor()

    # 处理PDF
    logger.info(f"开始处理PDF文件: {args.input_dir}")
    results = processor.process_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        save_intermediate=args.save_intermediate
    )

    # 输出结果
    logger.info("="*50)
    logger.info("处理完成")
    logger.info(f"总计: {results['total']} 个文件")
    logger.info(f"成功: {results['success']} 个")
    logger.info(f"失败: {results['failed']} 个")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"SFT训练数据: {args.output_dir}/sft_train.jsonl")

    if results.get('errors'):
        logger.warning(f"失败的文件:")
        for error in results['errors'][:10]:  # 只显示前10个
            logger.warning(f"  - {error}")

    # 生成训练格式数据
    logger.info("\n生成训练格式数据...")

    # 确保输出目录存在
    output_dir_path = Path(args.output_dir)
    structured_dir = output_dir_path / "structured"
    sft_train_path = output_dir_path / "sft_train.jsonl"

    # 检查是否有结构化数据
    if not structured_dir.exists() or results['total'] == 0:
        logger.warning(f"结构化数据目录不存在或没有处理文件: {structured_dir}")
        logger.info("跳过训练数据生成步骤")
        return
    else:
        # 生成SFT训练数据
        processor.convert_to_training_format(
            structured_dir=str(structured_dir),
            output_path=str(sft_train_path),
            task_type='qa'
        )

        # 验证生成的文件
        if sft_train_path.exists():
            file_size = sft_train_path.stat().st_size
            logger.info(f"✓ 训练数据文件已创建: {sft_train_path} (大小: {file_size} bytes)")
        else:
            logger.warning(f"✗ 训练数据文件未创建: {sft_train_path}")

    logger.info("处理流程完成！")


if __name__ == "__main__":
    main()
