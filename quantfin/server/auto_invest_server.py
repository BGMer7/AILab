from flask import Flask, request, jsonify, render_template
import sys
import os
from flask_cors import CORS

# 添加上级目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from myakshare.auto_invest import qdii_main

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("auto_invest.html")


@app.route("/calculate_investment", methods=["GET"])
def calculate():
    # 从请求参数中获取输入参数
    fund_code = request.args.get("fund_code")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    investment_day = request.args.get("investment_day")
    amount = request.args.get("amount")

    # 调用定投收益计算函数
    total_investment, current_value, profit, ratio = qdii_main(
        fund_code=fund_code,
        start_date=start_date,
        end_date=end_date,
        investment_day=investment_day,
        investment_amount=amount,
    )

    # 返回计算结果
    return jsonify(
        {
            "total_investment": total_investment,
            "current_value": current_value,
            "profit": profit,
            "ratio": ratio,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
