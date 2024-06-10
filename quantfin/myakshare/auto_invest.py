import akshare as ak
import pandas as pd
from datetime import datetime, timedelta


# 获取股票的历史数据
def get_stock_data_in_time(stock_code, start_date, end_date):
    stock_data = ak.stock_zh_a_hist(
        symbol=stock_code, period="daily", start_date=start_date, end_date=end_date
    )
    stock_data["date"] = pd.to_datetime(stock_data["日期"])
    stock_data.set_index("date", inplace=True)
    return stock_data


# 定投收益计算
def calculate_stock_auto_investment_income(
    stock_data, investment_amount, investment_weekday, start_date, end_date
):
    # 定投记录
    investment_record = []
    # 持仓记录
    holding_record = []

    current_date = start_date

    while current_date <= end_date:
        # 如果当前日期是用户指定的星期几
        if current_date.weekday() == investment_weekday:
            if current_date in stock_data.index:
                close_price = stock_data.loc[current_date, "收盘"]
                shares_bought = investment_amount / close_price
                investment_record.append(
                    (current_date, investment_amount, shares_bought)
                )
                holding_record.append(shares_bought)

        # 移动到下一天
        current_date += timedelta(days=1)

    investment_value = sum([record[1] for record in investment_record])
    total_shares = sum(holding_record)
    market_value = total_shares * stock_data.iloc[-1]["收盘"]
    profit = market_value - investment_value
    ratio = profit / investment_value

    return investment_value, market_value, profit, ratio


def get_fund_data_in_timezone(fund_code, start_date, end_date):
    """
    获取指定日期范围内的基金净值数据
    """
    fund_nav_df = ak.fund_open_fund_info_em(
        symbol=fund_code, indicator="单位净值走势", period="成立来"
    )
    print(fund_nav_df)
    fund_nav_df["净值日期"] = pd.to_datetime(fund_nav_df["净值日期"])
    fund_nav_df = fund_nav_df[
        (fund_nav_df["净值日期"] >= start_date) & (fund_nav_df["净值日期"] <= end_date)
    ]
    return fund_nav_df


def calculate_qdii_investment_income(fund_data, investment_day, investment_amount):
    """
    计算定投收益
    """
    investment_day = investment_day.lower()
    total_units = 0
    total_investment = 0

    for date, nav in fund_data[["净值日期", "单位净值"]].values:
        if date.strftime("%A").lower() == investment_day:
            units = investment_amount / nav
            total_units += units
            total_investment += investment_amount

    market_value = total_units * fund_data.iloc[-1]["单位净值"]
    profit = market_value - total_investment
    ratio = profit / total_investment
    return total_investment, market_value, profit, ratio


# 主函数
def stock_main():
    stock_code = input("请输入股票代码（如'000001'）：")
    start_date = input("请输入定投开始日期（格式：YYYYMMDD）：")
    end_date = input("请输入定投结束日期（格式：YYYYMMDD）：")
    investment_amount = float(input("请输入每次定投的金额（如1000）："))
    investment_weekday = int(
        input("请输入每周定投的星期几（0代表星期一，1代表星期二，以此类推）：")
    )

    start_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")

    stock_data = get_stock_data_in_time(
        stock_code, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
    )
    total_investment, current_value, profit = calculate_stock_auto_investment_income(
        stock_data, investment_amount, investment_weekday, start_date, end_date
    )

    print(f"总投资金额: {total_investment:.2f} 元")
    print(f"当前股票市值: {current_value:.2f} 元")
    print(f"总收益: {profit:.2f} 元")
    print(f"收益率: {profit / total_investment * 100:.2f}%")


def qdii_main(fund_code, start_date, end_date, investment_day, investment_amount):
    # fund_code = input("请输入QDII基金代码: ")
    # start_date = input("请输入定投开始日期 (YYYY-MM-DD): ")
    # end_date = input("请输入定投结束日期 (YYYY-MM-DD): ")
    # investment_day = input("请输入每周定投的日期 (如: 'wednesday'): ")
    # investment_amount = float(input("请输入每次定投的金额: "))
    fund_code = "000834"
    start_date = "20240301"
    end_date = "20240607"
    investment_day = "wednesday"
    investment_amount = 300

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    fund_data = get_fund_data_in_timezone(fund_code, start_date, end_date)
    total_invested, total_value, profit, ratio = calculate_qdii_investment_income(
        fund_data, investment_day, investment_amount
    )

    print(f"总投资金额: {total_invested:.2f} 元")
    print(f"投资结束时基金总市值: {total_value:.2f} 元")
    print(f"总收益: {profit:.2f} 元")
    print(f"收益率: {ratio * 100:.2f}% 元")

    return total_invested, total_value, profit, ratio


def main():
    qdii_main()


if __name__ == "__main__":
    main()
