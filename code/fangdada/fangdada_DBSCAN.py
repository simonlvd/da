# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np

data = pd.read_csv("D:\\fangdada\\log_data.csv")
data['浏览时长'] = pd.to_datetime(data['开始时间'], format='%Y-%m-%d %H:%M:%S') - pd.to_datetime(data['结束时间'], format='%Y-%m-%d %H:%M:%S')
data['浏览时长'] = data['浏览时长'].map(lambda i: i/np.timedelta64(1, 'h')) # 将时间差转换以小时为单位的时长
R_Agg = data.groupby(by=['用户id'])['浏览时长'].agg({'rencency': np.sum})
F_Agg = data.groupby(by=['用户id'])['访问次数'].agg({'frequency' : np.sum})
M_Agg = data.groupby(by=['用户id'])['咨询次数'].agg({'monetary' : np.sum})
new_data = R_Agg.jion(F_Agg).jion(M_Agg)

x = new_data[['浏览时长', '访问次数', '咨询次数']]
eps = 20
MinPts = 50
model = DBSCAN(eps, MinPts).fit(x)
labels = model.labels_
new_data['labels'] = labels
raito = len(labels[labels[:] == -1]) / len(labels)  #计算噪声点个数占总数的比例
print('Noise raito:', format(raito, '.2%'))
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x, labels))

plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'SimHei'
colors = np.array(['red', 'green', 'blue'])
scatter_matrix(new_data[['浏览时长', '访问次数', '咨询次数']],s=100, alpha=1, c=colors[new_data["labels"]], figsize=(10,10))
plt.suptitle("聚类结果")
plt.show()
new_data.to_csv('D:\\new_data.csv')