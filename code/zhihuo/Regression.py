import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("D:\\data.csv")
# 绘制散点图,确定使用哪种回归方式
x = data[["充值金额"]]
y = data[["返利"]]
plt.rc("font", family="SimHei", size=13)
matplotlib.rcParams["axes.unicode_minus"] = False
scatter_matrix(data[["充值金额", "返利"]],
               alpha=0.8,
               figsize=(10,10),
               diagonal="kde",
               c="red")
plt.show()
# 转换成多元一次方程
pf = PolynomialFeatures(degree=2)    # 根据上图确定为一元二次方程，因此degree赋值为2
x_2_fit = pf.fit_transform(x)
print(pf)
# 建立模型
lrModel = LinearRegression()
lrModel.fit(x_2_fit, y)
# 模型评分
a = lrModel.score(x_2_fit, y)
print(a)
# 模型预测
x_2_predict = pf.fit_transform([[10], [30], [50], [100], [200]])
print(x_2_predict)