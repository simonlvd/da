# -*- coding: utf-8 -*-
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import seaborn as sns
from matplotlib.pylab import style
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#读取数据
dateparse = lambda dates: pandas.datetime.strptime(dates, '%Y%m%d')
data = pandas.read_csv(
    'D:\\zhongying\\date.csv',
    parse_dates=['date'],
    date_parser=dateparse, 
    index_col='date'
)
# 画图展示
data_month = data['amount_money'].resample('M').sum()
data_train = data_month['2016-05-01', '2017-05-01']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(10,6))
plt.legend(bbox_to_anchor=(1.25, 0.5))
plt.title("赔付金额")
sns.despine()

# 进行一阶差分
diff = data_train.diff(1).dropna()
plt.figure(figsize=(10,6))
plt.plot(diff, 'r', label='Diff')
plt.title('一阶差分')
plt.legend(bbox_to_anchor=(1.25, 0.5))
sns.despine()

# 选择p，q 值
# ic = sm.tsa.arma_order_select_ic(diff, max_ar=20, max_ma=20, ic='aic')
acf = plot_acf(data_train,lags=20)
plt.title('ACF')
acf.show()
pacf = plot_pacf(data_train, lags=20)
plt.title("PACF")
pacf.show()

# 进行模型训练
model = ARIMA(data_train, order=(1,1,1), freq='M')  # p,1阶差分,q
restult = model.fit()
# delta = ARIMAModel.fittedvalues - diff.iloc[:,0]
# score = 1 - delta.var()/diff.var()
# print(score)
print(restult.summary())
# 预测
p = restult.predict('20180531', '20190531', dynamic=True, typ='levels')
plt.figure(figsize=(6, 6))
plt.xticks(rotation=45)
plt.plot(p)
plt.plot(data_train)
plt.show()

# 将一差处理后的预测数据进还原
def revert(diffValues, *lastValue):
    for i in range(len(lastValue)):
        result = []
        lv = lastValue[i]
        for dv in diffValues:
            lv = dv + lv
            result.append(lv)
        diffValues = result
    return diffValues
    
r = revert(p, 11457)
data_index = pd.data_range('2018-05-01', periods=12, freq='M')
predict_data = pd.DataFrame({'time':data_index, 'amount_money':r})
prodict_data.to_csv('F:\\zhongying\\prodict_data.csv',index=Flase)

