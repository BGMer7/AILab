document.getElementById('investment-form').addEventListener('submit', function (event) {
    event.preventDefault();

    const formData = new FormData(this);
    const data = {
        stock_code: formData.get('stock_code'),
        start_date: formData.get('start_date'),
        end_date: formData.get('end_date'),
        investment_amount: formData.get('investment_amount'),
        investment_day: formData.get('investment_weekday')
    };

    // 构建 URL
    const url = `/calculate_investment?fund_code=${stock_code}&start_date=${start_date}&end_date=${end_date}&investment_day=${data.investment_day}&amount=${investment_amount}`;

    // 发送 GET 请求
    fetch(url)
        .then(response => response.json())
        .then(result => {
            console.log(result);
        })
        .catch(error => console.error('Error:', error));
});
