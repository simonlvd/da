import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm,skew
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p


# 加载数据集
# train = pd.read_csv('D:\\fangdada\\train_data.csv', encoding='utf-8')
# test = pd.read_csv('D:\\fangdada\\test_data.csv', encoding='utf-8')
train = pd.read_csv('F:\\train.csv')
test = pd.read_csv('F:\\test.csv')
ntrain = train.shape[0]
ntest = test.shape[0]

# 查看是否满足正则分布
sns.distplot(train['prices'], fit=norm)
(mu, sigma) = norm.fit(train['prices'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# 分布图
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
# QQ图
fig = plt.figure()
res = stats.probplot(train['prices'], plot=plt)
plt.show()

# 使用对数进行转换，使数据符合正态分布
y_train = np.log1p(train['prices'])

# 查看新的分布
sns.distplot(train['prices'], fit=norm)
(mu, sigma) = norm.fit(train['prices'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# 画图
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()
# QQ图
fig = plt.figure()
res = stats.probplot(train['prices'], plot=plt)
plt.show()

# 整合数据
ntrain = train.shape[0]
ntest = test.shape[0]
n_data = pd.concat((train, test)).reset_index(drop=True)
n_data.drop(['name','prices'], axis=1, inplace=True)

# 处理数据函数
def data_handle(data):
    '''
    area: 住房面积
    green_area: 绿化面积
    parking: 提车位
    property_fee: 物业费
    distance_subway: 离地铁站距离
    distance_market: 离商圈距离
    locus: 所在区
    :param data:
    :return: data
    '''
    # 处理缺失值
    data['area'] = data['area'].fillna(data['area'].mean())
    data['green_area'] = data['green_area'].fillna(data['green_area'].mean())
    data['parking'] = data['parking'].fillna(data['parking'].mean())
    data['property_fee'] = data['property_fee'].fillna(data['property_fee'].mean())
    data['distance_subway'] = data['distance_subway'].fillna(data['distance_subway'].mean())
    data['distance_market'] = data['distance_market'].fillna(data['distance_market'].mean())
    data['locus'] = data['locus'].fillna(data['locus'].mode()[0])

    # 使用sklearn将离散数据进行标签映射labelencoder编码
    lbl = LabelEncoder()
    lbl.fit(list(data['locus'].values))
    data['locus'] = lbl.transform(list(data['locus'].values))

    # 转换连续型特征列
    numeric_feats = data.dtypes[data.dtypes != "object"].index
    skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    # skewness = pd.DataFrame({'Skew': skewed_feats})
    # skewness = skewness[abs(skewness) > 0.75]
    # skewed_features = skewness.index
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    # 使用box-cox进行变换
    lam = 0.15
    for feat in skewed_feats:
        data[feat] = boxcox1p(data[feat], lam)
        # all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
        data = pd.get_dummies(data)
    return data
all_data = data_handle(data=n_data)

# 创建一个新的训练集和测试集
x_train = all_data[:ntrain]
x_test = all_data[ntrain:]

from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

# 采用kfold拆分法进行交叉检验，建立打分系统，命名为rmsle_cv
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42)
    kf = kf.get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)

#在前期数据处理时有些离散的变量，所以这里使用RobustScaler对结果进行处理
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.7, random_state=3))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# 平均基础模型class
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # 训练模型
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self

    # 预测
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)

averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# 均方根误差
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

averaged_models.fit(x_train.values, y_train)
averaged_train_pred = averaged_models.predict(x_train.values)
averaged_pred = np.expm1(averaged_models.predict(x_test.values))
print(rmsle(y_train, averaged_train_pred))
print(averaged_pred)

# 加载需要预测的数据进行预测
prodict_data = pd.read_csv('D:\\fangdada\\prodict_data.csv', encoding='utf-8')
n_data = data_handle(data=prodict_data)
n_data.drop("name", axis=1, inplace=True)
n_data['prices'] = averaged_models.predict(n_data.values)
n_data.to_csv('D:\\fangdada\\new_prodict_data.csv',index=False)
