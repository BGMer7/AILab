import tushare as ts

# 设置你的Tushare API token
ts.set_token('bed2454009494f58a58ecac3816ca569ef4afe44a43461c3976da341')


def get_us_stock_calender():
    pro = ts.pro_api()
    trade_cal = pro.trade_cal(exchange='NASDAQ', start_date='20240101', end_date='20241231')
    trade_days = trade_cal[trade_cal["is_open"] == 1]["cal_date"].tolist()
    return trade_days
