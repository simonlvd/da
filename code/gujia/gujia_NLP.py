import pandas as pd
from snownlp import sentiment
from snownlp import SnowNLP

def nlp(filepath, neg, pos):
    text = pd.read_excel(filepath)
    contents = text.iloc[:, 0]
    contents_t = contents.values.tolist()
    sentiment.train(neg, pos)
    sent = [SnowNLP(i).sentiments for i in contents_t]
    predict = []
    # 大于0.5则输出1，小于0.5则输出-1
    for i in sent:
        if (i >= 0.5):
            predict.append(1)
        else:
            predict.append(-1)
    text['predict'] = predict
    text.to_excel('G:\\content_data.xlsx')
    return text

# filepath = 'G:\\评论_train.xlsx'
neg = 'G:\\neg.txt'
pos = 'G:\\pos.txt'
# data = nlp(filepath, neg, pos)
# counts=0
# for j in range(len(data.iloc[:,0])): #遍历所有标签，将预测标签和实际标签进行比较，相同则判断正确。
#     if data.iloc[j,2]==data.iloc[j,1]:
#         counts+=1
# print(u"准确率为:%f"%(float(counts)/float(len(data))))#输出本次预测的准确率
filepath1 = 'G:\\评论.xlsx'
nlp(filepath1, neg, pos)
