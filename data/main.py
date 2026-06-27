"""
天池竞赛：资金流入流出预测 - 挑战Baseline
数据路径：与 main.py 同目录
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DATA_PATH = './'  # 数据与main.py在同一目录

# ==================== 1. 数据加载 ====================

def load_data():
    """加载所有数据文件"""
    print("正在加载数据...")

    # 用户申购赎回数据（主数据）
    user_balance = pd.read_csv(f'{DATA_PATH}user_balance_table.csv')
    user_balance['report_date'] = pd.to_datetime(user_balance['report_date'], format='%Y%m%d')

    # 收益率表
    interest = pd.read_csv(f'{DATA_PATH}mfd_day_share_interest.csv')
    interest['mfd_date'] = pd.to_datetime(interest['mfd_date'], format='%Y%m%d')

    # 银行间拆借利率表
    shibor = pd.read_csv(f'{DATA_PATH}mfd_bank_shibor.csv')
    shibor['mfd_date'] = pd.to_datetime(shibor['mfd_date'], format='%Y%m%d')

    # 用户基本信息表
    user_profile = pd.read_csv(f'{DATA_PATH}user_profile_table.csv')

    print(f"✓ user_balance: {len(user_balance)} 条记录")
    print(f"✓ interest: {len(interest)} 条记录")
    print(f"✓ shibor: {len(shibor)} 条记录")
    print(f"✓ user_profile: {len(user_profile)} 条记录")

    return user_balance, interest, shibor, user_profile

# ==================== 2. 数据预处理 ====================

def preprocess_data(user_balance):
    """按日汇总申购赎回总量"""
    print("\n正在预处理数据...")

    # 按日期汇总
    daily_data = user_balance.groupby('report_date').agg({
        'total_purchase_amt': 'sum',
        'total_redeem_amt': 'sum'
    }).reset_index()

    daily_data.columns = ['date', 'total_purchase_amt', 'total_redeem_amt']
    daily_data = daily_data.sort_values('date').reset_index(drop=True)

    print(f"数据时间范围: {daily_data['date'].min().date()} ~ {daily_data['date'].max().date()}")
    print(f"共 {len(daily_data)} 天")

    return daily_data

# ==================== 3. 方法1：周期因子法（推荐Baseline） ====================

def period_factor_predict(daily_data, predict_dates):
    """
    周期因子法 - 核心Baseline
    利用星期几的周期性规律进行预测
    """
    print("\n使用【周期因子法】进行预测...")

    # 添加星期特征
    daily_data['weekday'] = daily_data['date'].dt.dayofweek  # 0=周一, 6=周日

    # 使用最近3个月数据（2014年6-8月）计算周期因子，更稳定
    recent_data = daily_data[daily_data['date'] >= '2014-06-01']

    # 计算每个星期几的平均值
    weekday_purchase = recent_data.groupby('weekday')['total_purchase_amt'].mean()
    weekday_redeem = recent_data.groupby('weekday')['total_redeem_amt'].mean()

    # 计算基准值（最近30天均值，更贴近当前水平）
    base_purchase = daily_data['total_purchase_amt'].tail(30).mean()
    base_redeem = daily_data['total_redeem_amt'].tail(30).mean()

    # 计算周期因子
    purchase_factors = weekday_purchase / base_purchase
    redeem_factors = weekday_redeem / base_redeem

    print("周期因子（申购）:")
    for i, f in enumerate(purchase_factors):
        day_name = ['周一','周二','周三','周四','周五','周六','周日'][i]
        print(f"  {day_name}: {f:.4f}")

    # 预测
    predictions = []
    for date in predict_dates:
        weekday = date.dayofweek
        pred_purchase = base_purchase * purchase_factors[weekday]
        pred_redeem = base_redeem * redeem_factors[weekday]
        predictions.append({
            'date': date,
            'total_purchase_amt': pred_purchase,
            'total_redeem_amt': pred_redeem
        })

    return pd.DataFrame(predictions)

# ==================== 4. 方法2：移动平均法（备用） ====================

def moving_average_predict(daily_data, predict_dates):
    """用最近N天均值预测"""
    print("\n使用【移动平均法】进行预测...")

    window = 30
    base_purchase = daily_data['total_purchase_amt'].tail(window).mean()
    base_redeem = daily_data['total_redeem_amt'].tail(window).mean()

    predictions = pd.DataFrame({
        'date': predict_dates,
        'total_purchase_amt': [base_purchase] * len(predict_dates),
        'total_redeem_amt': [base_redeem] * len(predict_dates)
    })

    return predictions

# ==================== 5. 方法3：加权融合（进阶） ====================

def ensemble_predict(daily_data, predict_dates):
    """
    融合周期因子 + 移动平均
    """
    print("\n使用【融合模型】进行预测...")

    # 周期因子预测
    pred_period = period_factor_predict(daily_data, predict_dates)

    # 移动平均基准
    base_purchase = daily_data['total_purchase_amt'].tail(30).mean()
    base_redeem = daily_data['total_redeem_amt'].tail(30).mean()

    # 融合：70%周期因子 + 30%移动平均
    predictions = pred_period.copy()
    predictions['total_purchase_amt'] = (
        0.7 * pred_period['total_purchase_amt'] + 0.3 * base_purchase
    )
    predictions['total_redeem_amt'] = (
        0.7 * pred_period['total_redeem_amt'] + 0.3 * base_redeem
    )

    return predictions

# ==================== 6. 生成提交文件 ====================

def generate_submission(predictions, filename='tc_comp_predict_table.csv'):
    """
    生成符合提交格式的CSV
    格式：report_date, purchase, redeem（无header，日期格式YYYYMMDD）
    """
    print(f"\n正在生成提交文件: {filename}")

    result = predictions.copy()
    result['report_date'] = result['date'].dt.strftime('%Y%m%d')
    result = result[['report_date', 'total_purchase_amt', 'total_redeem_amt']]

    # 保存为CSV（无header，符合提交格式）
    result.to_csv(filename, index=False, header=False)

    print(f"✓ 文件已保存！共 {len(result)} 行")
    print("\n预览（前10行）:")
    print(result.head(10).to_string(index=False))

    return result

# ==================== 7. 可视化 ====================

def visualize(daily_data, predictions):
    """绘制历史数据与预测结果"""
    print("\n正在绘制可视化图表...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 申购
    ax1 = axes[0]
    ax1.plot(daily_data['date'], daily_data['total_purchase_amt'],
             label='历史申购', color='blue', alpha=0.7)
    ax1.plot(predictions['date'], predictions['total_purchase_amt'],
             label='预测申购', color='red', linewidth=2)
    ax1.axvline(x=pd.Timestamp('2014-09-01'), color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('申购金额预测 (Purchase)', fontsize=14)
    ax1.set_xlabel('日期')
    ax1.set_ylabel('金额')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 赎回
    ax2 = axes[1]
    ax2.plot(daily_data['date'], daily_data['total_redeem_amt'],
             label='历史赎回', color='green', alpha=0.7)
    ax2.plot(predictions['date'], predictions['total_redeem_amt'],
             label='预测赎回', color='red', linewidth=2)
    ax2.axvline(x=pd.Timestamp('2014-09-01'), color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('赎回金额预测 (Redeem)', fontsize=14)
    ax2.set_xlabel('日期')
    ax2.set_ylabel('金额')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
    print("✓ 图表已保存: prediction_result.png")
    plt.show()

# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("  天池竞赛：资金流入流出预测 - 挑战Baseline")
    print("=" * 60)

    # 1. 加载数据
    user_balance, interest, shibor, user_profile = load_data()

    # 2. 预处理
    daily_data = preprocess_data(user_balance)

    # 3. 生成预测日期（2014年9月1日-30日）
    predict_dates = pd.date_range(start='2014-09-01', end='2014-09-30', freq='D')
    print(f"\n预测目标: 2014年9月 {len(predict_dates)} 天")

    # 4. 选择预测方法
    # method = 'period'      # 方法1：周期因子法（推荐）
    # method = 'ma'          # 方法2：移动平均法
    method = 'ensemble'    # 方法3：融合模型（效果最好）

    if method == 'period':
        predictions = period_factor_predict(daily_data, predict_dates)
    elif method == 'ma':
        predictions = moving_average_predict(daily_data, predict_dates)
    else:
        predictions = ensemble_predict(daily_data, predict_dates)

    # 5. 生成提交文件
    generate_submission(predictions, 'tc_comp_predict_table.csv')

    # 6. 可视化
    visualize(daily_data, predictions)

    print("\n" + "=" * 60)
    print("  Baseline 运行完成！")
    print("  提交文件: tc_comp_predict_table.csv")
    print("=" * 60)

if __name__ == '__main__':
    main()