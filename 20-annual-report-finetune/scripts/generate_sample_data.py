"""
生成示例训练数据脚本
在没有真实PDF数据的情况下，生成示例的sft_train.jsonl文件用于测试
"""

import json
import os
from pathlib import Path
from loguru import logger
from typing import List, Dict


def generate_sample_reports() -> List[Dict]:
    """生成示例年报数据"""
    companies = [
        {
            "name": "贵州茅台",
            "code": "600519.SH",
            "year": 2023,
            "revenue": 150560746878.38,
            "net_profit": 74734764045.72,
            "gross_margin": 0.91,
            "roe": 0.25,
            "debt_ratio": 0.18,
            "business": "茅台酒及系列酒的生产与销售",
            "risks": [
                {"type": "政策风险", "description": "白酒行业监管政策变化可能影响经营", "severity": "medium"},
                {"type": "市场风险", "description": "消费市场波动影响产品需求", "severity": "medium"}
            ],
            "highlights": ["营业收入突破1500亿元", "净利润同比增长18%", "品牌价值持续提升"]
        },
        {
            "name": "中国平安",
            "code": "601318.SH",
            "year": 2023,
            "revenue": 1244000000000.00,
            "net_profit": 83750000000.00,
            "gross_margin": 0.35,
            "roe": 0.15,
            "debt_ratio": 0.85,
            "business": "保险、银行、投资等金融服务",
            "risks": [
                {"type": "政策风险", "description": "金融监管政策调整", "severity": "high"},
                {"type": "市场风险", "description": "资本市场波动", "severity": "high"},
                {"type": "信用风险", "description": "投保人信用风险", "severity": "medium"}
            ],
            "highlights": ["综合金融服务能力领先", "数字化转型成效显著", "客户规模稳步增长"]
        },
        {
            "name": "腾讯控股",
            "code": "00700.HK",
            "year": 2023,
            "revenue": 554500000000.00,
            "net_profit": 18800000000.00,
            "gross_margin": 0.48,
            "roe": 0.12,
            "debt_ratio": 0.25,
            "business": "社交、游戏、金融科技、云服务等",
            "risks": [
                {"type": "政策风险", "description": "互联网监管政策变化", "severity": "high"},
                {"type": "技术风险", "description": "技术迭代快速", "severity": "medium"},
                {"type": "竞争风险", "description": "行业竞争加剧", "severity": "high"}
            ],
            "highlights": ["微信用户规模超13亿", "游戏业务稳健增长", "AI技术应用深入"]
        },
        {
            "name": "比亚迪",
            "code": "002594.SZ",
            "year": 2023,
            "revenue": 602300000000.00,
            "net_profit": 30000000000.00,
            "gross_margin": 0.20,
            "roe": 0.18,
            "debt_ratio": 0.65,
            "business": "新能源汽车及电池制造",
            "risks": [
                {"type": "市场风险", "description": "新能源汽车市场竞争激烈", "severity": "high"},
                {"type": "技术风险", "description": "技术路线变化", "severity": "medium"},
                {"type": "政策风险", "description": "补贴政策调整", "severity": "high"}
            ],
            "highlights": ["新能源汽车销量全球第一", "电池技术领先", "海外市场拓展顺利"]
        },
        {
            "name": "宁德时代",
            "code": "300750.SZ",
            "year": 2023,
            "revenue": 400300000000.00,
            "net_profit": 44000000000.00,
            "gross_margin": 0.22,
            "roe": 0.20,
            "debt_ratio": 0.55,
            "business": "动力电池、储能电池研发制造",
            "risks": [
                {"type": "技术风险", "description": "电池技术迭代", "severity": "high"},
                {"type": "市场风险", "description": "原材料价格波动", "severity": "high"},
                {"type": "竞争风险", "description": "行业竞争加剧", "severity": "high"}
            ],
            "highlights": ["全球动力电池市场份额第一", "储能业务快速增长", "研发投入持续加大"]
        }
    ]

    return companies


def create_qa_samples(company: Dict) -> List[Dict]:
    """为每个公司创建问答样本"""
    samples = []

    name = company['name']
    year = company['year']
    code = company['code']

    # 财务指标问答
    samples.append({
        "instruction": f"{name}{year}年的营业收入是多少？",
        "input": "",
        "output": f"{name}{year}年的营业收入为{company['revenue']:.2f}元。"
    })

    samples.append({
        "instruction": f"{name}{year}年的净利润是多少？",
        "input": "",
        "output": f"{name}{year}年的净利润为{company['net_profit']:.2f}元。"
    })

    samples.append({
        "instruction": f"{name}{year}年的毛利率是多少？",
        "input": "",
        "output": f"{name}{year}年的毛利率为{company['gross_margin']*100:.2f}%。"
    })

    samples.append({
        "instruction": f"{name}{year}年的净资产收益率（ROE）是多少？",
        "input": "",
        "output": f"{name}{year}年的净资产收益率为{company['roe']*100:.2f}%。"
    })

    samples.append({
        "instruction": f"{name}{year}年的资产负债率是多少？",
        "input": "",
        "output": f"{name}{year}年的资产负债率为{company['debt_ratio']*100:.2f}%。"
    })

    # 业务问答
    samples.append({
        "instruction": f"{name}的主营业务是什么？",
        "input": "",
        "output": f"{name}的主营业务是{company['business']}。"
    })

    # 风险问答
    risks_text = "、".join([r['type'] for r in company['risks']])
    samples.append({
        "instruction": f"{name}{year}年面临的主要风险有哪些？",
        "input": "",
        "output": f"{name}{year}年主要面临{risks_text}等风险。"
    })

    # 亮点问答
    highlights_text = "、".join(company['highlights'])
    samples.append({
        "instruction": f"{name}{year}年的经营亮点是什么？",
        "input": "",
        "output": f"{name}{year}年的经营亮点包括：{highlights_text}。"
    })

    # 综合分析问答
    analysis = f"""{name}{year}年度财务分析：营业收入为{company['revenue']:.2f}元，净利润为{company['net_profit']:.2f}元，毛利率为{company['gross_margin']*100:.2f}%，净资产收益率为{company['roe']*100:.2f}%，资产负债率为{company['debt_ratio']*100:.2f}%。"""

    if company['roe'] > 0.15:
        analysis += "净资产收益率较高，盈利能力较强。"
    else:
        analysis += "净资产收益率一般，盈利能力有待提升。"

    high_risks = [r for r in company['risks'] if r['severity'] == 'high']
    if high_risks:
        analysis += f"存在{len(high_risks)}项高风险因素，需重点关注。"

    samples.append({
        "instruction": f"请分析{name}{name}年的财务状况和经营风险。",
        "input": "",
        "output": analysis
    })

    return samples


def generate_sft_train_data(output_path: str, num_companies: int = 5, samples_per_company: int = 10):
    """
    生成SFT训练数据

    Args:
        output_path: 输出文件路径
        num_companies: 公司数量
        samples_per_company: 每个公司生成的样本数
    """
    logger.info("开始生成示例训练数据...")

    # 获取示例公司数据
    companies = generate_sample_reports()

    # 如果需要更多公司，复制并修改
    while len(companies) < num_companies:
        base_company = companies[len(companies) % len(generate_sample_reports())]
        new_company = base_company.copy()
        new_company['code'] = f"{new_company['code'][:4]}{len(companies)+1:02d}.{new_company['code'].split('.')[-1]}"
        companies.append(new_company)

    companies = companies[:num_companies]

    # 生成训练样本
    all_samples = []
    for company in companies:
        samples = create_qa_samples(company)
        all_samples.extend(samples)

    # 限制每个公司的样本数
    all_samples = all_samples[:num_companies * samples_per_company]

    # 保存为JSONL格式
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    logger.info(f"✓ 生成 {len(all_samples)} 条训练样本")
    logger.info(f"✓ 输出文件: {output_path}")
    logger.info(f"✓ 文件大小: {output_file.stat().st_size} bytes")

    # 显示部分样本
    logger.info("\n样本示例:")
    for i, sample in enumerate(all_samples[:3]):
        logger.info(f"\n样本 {i+1}:")
        logger.info(f"  指令: {sample['instruction'][:50]}...")
        logger.info(f"  输出: {sample['output'][:50]}...")

    return len(all_samples)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="生成示例SFT训练数据")
    parser.add_argument(
        '--output_path',
        type=str,
        default='data/processed/sft_train.jsonl',
        help='输出文件路径'
    )
    parser.add_argument(
        '--num_companies',
        type=int,
        default=10,
        help='生成多少家公司的数据'
    )
    parser.add_argument(
        '--samples_per_company',
        type=int,
        default=10,
        help='每家公司生成多少条样本'
    )

    args = parser.parse_args()

    # 生成数据
    total_samples = generate_sft_train_data(
        output_path=args.output_path,
        num_companies=args.num_companies,
        samples_per_company=args.samples_per_company
    )

    logger.info(f"\n✓ 示例数据生成完成！")
    logger.info(f"  公司数量: {args.num_companies}")
    logger.info(f"  总样本数: {total_samples}")
    logger.info(f"  输出文件: {args.output_path}")


if __name__ == "__main__":
    main()
