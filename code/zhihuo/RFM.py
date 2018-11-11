import numpy as np
import pandas as pd
# 导入数据，处理时间列
data = pd.read_csv('D:\\data.csv')
data['n_time'] = pd.to_datetime(data.time, format='%Y%m%d')
data['time_diff'] = pd.to_datetime('today') - data['n_time']
data['time_diff'] = data['time_diff'].dt.days
# 计算RFM三个值
R_Agg = data.groupby(by=['Customer_id'])['time_diff'].agg({'rencency': np.min})
F_Agg = data.groupby(by=['Customer_id'])['order_id'].agg({'frequency' : np.size})
M_Agg = data.groupby(by=['Customer_id'])['monetary'].agg({'monetary' : np.sum})
new_data = R_Agg.jion(F_Agg).jion(M_Agg)

bins = new_data.rencecy.quantile(q=[0, 0.2, 0.4, 0.6, 0.8, 1], interpolation='nearest')
bins[0] = 0
label = [5, 4, 3, 2, 1]
R_S = pd.cut(new_data.rencecy, bins, labels=label)

bins = new_data.frequency.quantile(q=[0, 0.2, 0.4, 0.6, 0.8, 1], interpolation='nearest')
bins[0] = 0
label = [1, 2, 3, 4, 5]
F_S = pd.cut(new_data.frequency, bins, labels=label)

bins = new_data.monetary.quantile(q=[0, 0.2, 0.4, 0.6, 0.8, 1], interpolation='nearest')
bins[0] = 0
label = [1, 2, 3, 4, 5]
M_S = pd.cut(new_data.monetary, bins, labels=label)
# 计算RFM得分
new_data['R_S'] = R_S
new_data['F_S'] = F_S
new_data['M_S'] = M_S
new_data['RFM'] = 100*R_S.astype(int) + 10*F_S.astype(int) + 1*M_S.astype(int)

bins = new_data.RFM.quantile(q=[0, 0.125, 0.25, 0.375, 0.5,0.625, 0.75, 0.875, 1], interpolation='nearest')
bins[0] = 0
labels = [1, 2, 3, 4, 5, 6, 7, 8]
new_data['level'] = pd.cut(new_data.RFM, bins, labels=labels)
new_data = new_data.reset_index()
new_data.to_csv('D:\\new_data.csv')