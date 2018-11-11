import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
# 导入数据，将特征列进行处理
data = pd.read_csv('D:\\data.txt', sep=' ',encoding='utf-8')
data['time'] = pd.to_datetime(data.starttime, format='%Y-%m-%d %H:%M:%S') - pd.to_datetime(data.endtime, format='%Y-%m-%d %H:%M:%S')
data['time'] = data['time'].map(lambda i: i/np.timedelta64(1, 'h')) # 将时间差转换以小时为单位的时长
R_Agg = data.groupby(by=['Customer_id'])['time'].agg({'rencency': np.sum})
F_Agg = data.groupby(by=['Customer_id'])['order_id'].agg({'frequency' : np.size})
M_Agg = data.groupby(by=['Customer_id'])['monetary'].agg({'monetary' : np.sum})
new_data = R_Agg.jion(F_Agg).jion(M_Agg)
x = new_data[['monetary', 'time', 'frequency']]

# 先用循环来确定k值为多少比较合适
scores = []
for k in range(2,20):
    labels = KMeans(n_clusters=k).fit(x).labels_
    score = metrics.silhouette_score(x, labels)
    scores.append(score)
print(scores)
plt.plot(list(range(2,20)), scores)
plt.xlabel('Number of Clusters Initialized')
plt.ylabel('Scores')
# plt.show()

# 将数据标准化后与未标准化数据都进行分类，再进行轮廓系数评分，确定使用哪种
km = KMeans(n_clusters=5).fit(x)
new_data['cluster'] = km.labels_
# scaler = StandardScaler()
# x_scaler = scaler.fit_transform(x)
x_scaler = StandardScaler().fit_transform(x)
km1 = KMeans(n_clusters=5).fit(x_scaler)
new_data['s_cluster'] = km1.labels_
score_scaler = metrics.silhouette_score(x, new_data['s_cluster'])
score = metrics.silhouette_score(x, new_data.cluster)
print(score_scaler, score)

# 确定后再进行画图展示
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'SimHei'
colors = np.array(['red', 'green', 'blue'])
scatter_matrix(new_data[['monetary', 'time', 'frequency']],s=100, alpha=1, c=colors[new_data["cluster"]], figsize=(10,10))
plt.suptitle("聚类结果")
plt.show()
new_data.to_csv('D:\\new_data.csv')