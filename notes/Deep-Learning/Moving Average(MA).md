[[Autoregressive Transformer(AT)]]
[[Auto Regressive(AR)]]

一般的P阶自回归模型 AR中，
$x_t = f(x_{t-1}, x_{t-2}, \dots, x_{t-k}) + \epsilon$
如果$\epsilon$是一个白噪声，则称为是一个纯AR过程。

自回归模型首先需要确定一个阶数p，表示用几期的历史值来预测当前值。
自回归模型有很多的限制：
（1）自回归模型是用自身的数据进行预测
（2）时间序列数据必须具有平稳性
（3）自回归只适用于预测与自身前期相关的现象（时间序列的自相关性）

### **MA（Moving Average）模型**
在AR模型中，如果 $\epsilon$ 不是一个白噪声，通常认为它是一个q阶的移动平均。
**移动平均模型 (Moving Average)** 是时间序列分析的一种方法，用于描述序列中当前值与过去随机误差项的线性关系。

#### **模型定义**
一个 $q$ 阶移动平均模型 ($MA(q)$) 的数学形式为：
$$
X_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}
$$
- $X_t$：当前时间序列值。
- $\mu$：序列均值。
- $\epsilon_t$：白噪声，服从 $N(0, \sigma^2)$。
- $\theta_1, \theta_2, \dots, \theta_q$：模型系数。
- $q$：滞后项的数量。

#### **特点**
1. 当前值由过去 $q$ 个随机误差项线性组合而成。
2. 模型是对随机波动的平滑操作，适合捕捉短期波动。

#### **应用场景**
- 金融市场的短期价格预测。
- 序列平滑处理（噪声过滤）。

### **ARMA（Autoregressive Moving Average）模型**

**自回归移动平均模型 (ARMA)** 是结合了自回归 (AR) 和移动平均 (MA) 的混合模型，用于描述时间序列中长期趋势和短期波动。

#### **模型定义**
一个 $ARMA(p, q)$ 模型的数学形式为：
$$
X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \dots + \phi_p X_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q}
$$
- $p$：自回归 (AR) 的阶数。
- $q$：移动平均 (MA) 的阶数。
- $\phi_1, \phi_2, \dots, \phi_p$：自回归系数。
- $\theta_1, \theta_2, \dots, \theta_q$：移动平均系数。

#### **特点**
1. 结合了长期自回归关系和短期误差平滑。
2. 要求数据平稳（均值、方差和协方差不随时间变化）。

#### **应用场景**
- 中期和长期时间序列预测。
- 应用于经济学、气象学、金融市场等领域。

### ARIMA（Autoregressive Integrated Moving Average）模型

**自回归积分移动平均模型 (ARIMA)** 是对非平稳时间序列进行建模的一种扩展方法，通过差分操作将非平稳序列转换为平稳序列，再进行 ARMA 建模。

#### 模型定义
一个 $ARIMA(p, d, q)$ 模型的数学形式为：
1. 对序列 $X_t$ 进行 $d$ 阶差分，得到差分序列 $Y_t$：
   $$
   Y_t = X_t - X_{t-1} \quad (\text{d=1})
   $$
2. 对差分序列 $Y_t$ 应用 $ARMA(p, q)$ 模型：
   $$
   Y_t = \phi_1 Y_{t-1} + \dots + \phi_p Y_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q}
   $$

#### **参数解释**
- $p$：自回归阶数。
- $d$：差分次数（使数据平稳的阶数）。
- $q$：移动平均阶数。

#### **特点**
1. 可处理非平稳时间序列。
2. 将差分和 ARMA 模型结合，适应更多实际场景。

#### **应用场景**
- 长期时间序列预测（如 GDP、天气变化）。
- 复杂的经济和金融时间序列分析。

### **ARMA 和 ARIMA 的区别**
1. **数据类型**：
   - ARMA 需要数据是平稳的。
   - ARIMA 适用于非平稳数据，通过差分操作实现平稳化。
2. **额外参数**：
   - ARIMA 多了 $d$ 参数，表示差分次数。

### **模型选择与使用**
1. **数据平稳性检测**：
   - 使用 **ADF（Augmented Dickey-Fuller）检验** 或 **KPSS（Kwiatkowski-Phillips-Schmidt-Shin）检验** 检查数据是否平稳。
2. **模型阶数选择**：
   - 使用 **AIC（Akaike Information Criterion）** 或 **BIC（Bayesian Information Criterion）** 确定最佳 $p$、$q$ 和 $d$。
3. **预测与评估**：
   - 使用历史数据训练模型，通过残差分析验证拟合效果。

### **Python 实现示例（ARIMA 模型）**
```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# 生成时间序列数据（例如随机数）
data = pd.Series([100 + i + (i % 5) * 3 + (i // 10) * 2 + (10 - i % 10) for i in range(50)])

# 拟合 ARIMA 模型
model = ARIMA(data, order=(2, 1, 2))  # ARIMA(p, d, q)
fitted_model = model.fit()

# 输出模型摘要
print(fitted_model.summary())

# 绘制预测结果
forecast = fitted_model.forecast(steps=10)  # 预测未来10步
plt.plot(data, label="Original Data")
plt.plot(range(len(data), len(data) + 10), forecast, label="Forecast", color="red")
plt.legend()
plt.show()
```

### **总结**
1. MA、ARMA、ARIMA 是时间序列建模的重要工具，适用于不同复杂度的序列。
2. ARIMA 更广泛用于非平稳数据的分析，但训练时间和复杂度更高。
3. 实际应用时，选择适当的模型需要根据数据特性和目标任务综合考虑。

ref实战：
https://zhuanlan.zhihu.com/p/457212660