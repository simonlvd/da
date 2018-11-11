import pandas as pd
from surprise import KNNBaseline
from surprise import Dataset
from surprise import Reader
# 训练推荐模型，

def getSimModle():
    # 载入数据集
    reader = Reader(rating_scale=(1, 5))
    new_data = Dataset.load_from_df(data['user', 'product', 'rating'], reader)
    trainset = new_data.build_full_trainset()
    # 使用pearson_baseline方式计算相似度  以product为基准计算相似度
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    ##使用KNNBaseline算法
    algo = KNNBaseline(sim_options=sim_options)
    #训练模型
    algo.train(trainset)
    return algo

# 获取id到name的互相映射
def read_item_names():
    # 获取映射
    file_name = ('D:\\gujia\\gujia.item')
    rid_to_name = {}
    name_to_rid = {}
    with open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split(',')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]
    return rid_to_name, name_to_rid

# 基于之前训练的模型 进行相关产品的推荐
def showSimilarMovies(algo, rid_to_name, name_to_rid):
    # 导入需要推荐的数据
    product_data = pd.read_csv('D:\\gujia\\product_name.csv',encoding='utf-8', names=['product'])
    for i in product_data['product']:
        product_name = i
        # 获得产品的raw_id
        raw_id = name_to_rid['product_name']
        #把产品的raw_id转换为模型的内部id
        inner_id = algo.trainset.to_inner_iid(raw_id)
        # 通过模型获取推荐产品 这里设置的是5种
        neighbors = algo.get_neighbors(inner_id, 5)
        # 模型内部id转换为实际产品id
        neighbors_raw_ids = [algo.trainset.to_raw_iid(inner_id) for inner_id in neighbors]
        # 通过产品id列表 获取推荐列表
        neighbors_product = [rid_to_name[raw_id] for raw_id in neighbors_raw_ids]
        print('The 5 nearest neighbors of %s are:' % (product_name))
        for movie in neighbors_product:
            print(movie)


if __name__ == '__main__':
    # rating = 该用户所购买产品的数量/该产品卖出的总次数
    data = pd.read_csv('D:\\gujia\\gujia.csv', encoding='utf-8', names=['user', 'product', 'rating'])

    # 获取id到name的互相映射
    rid_to_name, name_to_rid = read_item_names()
    # 训练推荐模型
    algo = getSimModle()
    ##显示相关产品
    showSimilarMovies(algo, rid_to_name, name_to_rid)



