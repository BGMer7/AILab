import akshare as ak

# 获取A股上市公司列表
stock_list = ak.stock_info_a_code_name()
# 获取证券公司的股票代码
print(stock_list.head())

stock_industry_category_cninfo_df = ak.stock_industry_category_cninfo(symbol="巨潮行业分类标准")
# 遍历 DataFrame，并输出类目名称
for index, row in stock_industry_category_cninfo_df.iterrows():
    print(row['类目名称'])

# 获取证券公司的财务数据
financial_data = ak.stock_financial_report_sina(symbol="sh600000")

# 选择支付给职工以及为职工支付的现金项数据
cash_payment_to_employees = financial_data.loc["支付给职工以及为职工支付的现金"]

print(cash_payment_to_employees)
