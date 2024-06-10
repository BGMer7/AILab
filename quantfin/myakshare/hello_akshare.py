import akshare as ak
import pandas as pd
from datetime import datetime


today_date = datetime.now()

# 打印今天日期
history_date = today_date.strftime('%Y%m%d')
current_date = today_date.strftime('%Y%m%d')
print(history_date)

# 设置股票代码和时间范围
stock_code = "600000"  # 以浦发银行为例，股票代码需要加上交易所代码
start_date = "20240101"
end_date = current_date

# 使用 akshare 获取股票历史数据
stock_data = ak.stock_zh_a_hist(
    symbol=stock_code, start_date=start_date, end_date=end_date, adjust="qfq"
)

# 将交易日期转换为日期索引
stock_data.index = pd.to_datetime(stock_data["日期"])

# 打印股票数据的前几行
print(stock_data.head())
