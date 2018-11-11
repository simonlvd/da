import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.special import distance
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
# 选取数据
features = ['住房面积', '绿化面积', '停车位', '物业费', '距离地铁站距离', '距离商圈距离', '楼栋总数', 'price']
data = pd.read_csv('D:\\DATA.csv', encoding='utf-8')
x = data[features]
x.dropna()
# 标准化处理，消除量纲差距
x[features] = StandardScaler().fit_transform(x[features])
normalized_x = x
# train：=80%，test：20%
norm_train = normalized_x.copy().iloc[0:7836]
norm_test = normalized_x.copy().iloc[7836:]
def predict_price(test_data, feature_columns):
    temp_df = norm_train
    temp_df['distance'] = distance.cdist(temp_df[feature_columns], [test_data[feature_columns]], metric='euclidean')
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    p_price = knn_5.mean()
    return p_price
cols = ['住房面积', '绿化面积', '停车位', '物业费', '距离地铁站距离', '距离商圈距离', '楼栋总数']
norm_test['price'] = norm_test[cols].apply(predict_price,feature_columns=cols,axis=1)
# 计算RMSE误差，评估模型S={[(x1-x)^2 + (x2-x)^2 + ......(xn-x)^2 ]/N}^0.5
norm_test['squared_error'] = (norm_test['p_price'] - norm_test['price']) ** (2)
mse = norm_test['squared_error'].mean()
rmse = mse ** (1/2)
print(rmse)

# cols = ['住房面积', '绿化面积', '停车位', '物业费', '距离地铁站距离', '距离商圈距离', '楼栋总数']
# knn = KNeighborsRegressor()
# knn.fit(norm_train[cols], norm_train['price'])
# features_predict = knn.predict(norm_test[cols])
# features_rmse = mean_squared_error(norm_test['price'],features_predict)
# print(features_rmse)
new_data = pd.read_csv('D:\\NEW_DATA.csv', encoding='utf-8')
new_data['price'] = new_data[cols].apply(predict_price,feature_columns=cols,axis=1)
new_data.to_csv('F:\\new_data.csv', encoding='utf-8')



