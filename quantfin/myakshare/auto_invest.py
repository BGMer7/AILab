import akshare as ak
import pandas as pd
import sys, os
from datetime import datetime, timedelta


# 获取交易日历
def get_trade_calendar():
    return ak.tool_trade_date_hist_sina()


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


def calculate_qdii_investment_income(
    fund_code, start_date, end_date, investment_day, amount
):
    """
    计算定投收益
    """
    investment_day = investment_day.lower()
    fund_data = get_fund_data_in_timezone(fund_code, start_date, end_date)
    a_trade_calendar = get_trade_calendar()
    print("A股交易日", a_trade_calendar)
    # TODO 暂时假定美股的交易日与A股相同
    us_trade_calendar = get_trade_calendar()
    total_units = 0
    total_investment = 0

    """
    TODO 交易日的判断实际上不能简单根据入参的日期判断是否相等，入参的日期只是扣款日
    而要根据扣款日T+2计算份额
    """
    for date in pd.date_range(start=start_date, end=end_date):
        # 如果当前日期是用户指定的定投日
        if date.strftime("%A").lower() == investment_day.lower():
            # 如果当前日期是A股交易日
            if datetime.strptime(date.strftime("%Y-%m-%d"), "%Y-%m-%d").date() in a_trade_calendar.values:
                print(date.strftime("%Y-%m-%d"), "是A股交易日")
                # 找到下一个美股交易日
                next_day = date + timedelta(days=1)
                while datetime.strptime(next_day.strftime("%Y-%m-%d"), "%Y-%m-%d").date() not in us_trade_calendar.values:
                    next_day += timedelta(days=1)
                    
                if date in fund_data["净值日期"].values:
                    next_day_nav = fund_data[fund_data["净值日期"] == next_day][
                        "单位净值"
                    ].values[0]
                    print("amount", amount)
                    units = amount / next_day_nav
                    total_units += units
                    total_investment += amount
            else:
                print(date.strftime("%Y-%m-%d"), "不是A股交易日")
                # 如果当前日期不是A股交易日，寻找下一个A股交易日
                next_a_trade_date = date + timedelta(days=1)
                while datetime.strptime(next_day.strftime("%Y-%m-%d"), "%Y-%m-%d").date() not in us_trade_calendar.values:
                    next_day += timedelta(days=1)

                # 找到对应的美股交易日
                next_us_trade_date = next_a_trade_date + timedelta(days=1)
                while (
                    datetime.strptime(next_us_trade_date.strftime("%Y-%m-%d"), "%Y-%m-%d").date()
                    not in us_trade_calendar.values
                ):
                    next_us_trade_date += timedelta(days=1)

                if date in fund_data["净值日期"].values:
                    next_day_nav = fund_data[
                        fund_data["净值日期"] == next_us_trade_date
                    ]["单位净值"].values[0]
                    units = amount / next_day_nav
                    total_units += units
                    total_investment += amount

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
    fund_code = "000834"
    start_date = "20240301"
    end_date = "20240607"
    investment_day = "wednesday"
    investment_amount = 300

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    total_invested, total_value, profit, ratio = calculate_qdii_investment_income(
        fund_code, start_date, end_date, investment_day, investment_amount
    )

    print(f"总投资金额: {total_invested:.2f} 元")
    print(f"投资结束时基金总市值: {total_value:.2f} 元")
    print(f"总收益: {profit:.2f} 元")
    print(f"收益率: {ratio * 100:.2f}% 元")

    return total_invested, total_value, profit, ratio


if __name__ == "__main__":
    main()
