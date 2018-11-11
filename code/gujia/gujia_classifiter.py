import pandas as pd
import jieba
import jieba.analyse
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
from gensim import corpora, models, similarities
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
"""
第一步：进行分词，画词云图，并将处理好的数据放回，以备后期建模使用
"""

# 先分词, 去掉停用词
def handel_cut(data, stopwords):
    content = data.content.values.tolist()
    content_S = []
    for line in content:
        current_segment = jieba.lcut(line)
        if len(current_segment) > 1 and current_segment !='\r\n':
            content_S.append(current_segment)
    df_content = pd.DataFrame({'content_S': content_S})
    contents = df_content.content_S.values.tolist()
    stopwords = stopwords.stopword.values.tolist()
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean,all_words

def plot(df_all_words):
    words_count = df_all_words.groupby(by=['all_words'])['all_words'].agg({"count":np.size})
    words_count = words_count.reset_index().sort_values(by=["count"],ascending=False)
    matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
    wordcloud = WordCloud(font_path="./data/simhei.ttf",background_color="white",max_font_size=80)
    word_frequence = {x[0]:x[1] for x in words_count.head(50).values}
    wordcloud=wordcloud.fit_words(word_frequence)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

data = pd.read_excel('F:\\content.xlsx')
data = data['content']
data.dropna()
stopwords = pd.read_csv("F:\\stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'], encoding='utf-8')
contents_clean,all_words = handel_cut(data, stopwords)
df_content = pd.DataFrame({'contents_clean':contents_clean})
df_content['content'] = data['content']
df_content.to_excel('F:\\gujia_content.xlsx')
df_all_words=pd.DataFrame({'all_words':all_words})

"""
第二步：用lda和tfidf建模分类，并进行评分选择最优的一个
"""

# 进行数据处理
def data_handel(x):
    words = []
    for line_index in range(len(x)):
        try:
            #x_train[line_index][word_index] = str(x_train[line_index][word_index])
            words.append(' '.join(x[line_index]))
        except:
            print (line_index,x[line_index])
    return words

# 从分好词的数据中拿出一部分，手动分类并保存，用于模型训练
new_data = pd.read_excel('F:\\gujia_content.xlsx')
clean_content = new_data['category', 'contents_clean']
df_train=pd.DataFrame({'contents_clean': new_data['contents_clean'],'label': new_data['category']})
label_mapping = {"质量": 1, "气味": 2, "快递": 3, "服务": 4, "安装":5}
df_train['label'] = df_train['label'].map(label_mapping)
x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values,random_state=1)
words = data_handel(x=x_train)
test_words = data_handel(x=x_test)
#做映射，相当于词袋
dictionary = corpora.Dictionary(clean_content)
corpus = [dictionary.doc2bow(sentence) for sentence in clean_content]

# 使用lda模型
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
# 词向量转换
vec = CountVectorizer(analyzer='word', max_features=50, lowercase=False)
vec.fit(words)

# 调用贝叶斯算法进行模型的分类
classifier = MultinomialNB()
classifier.fit(vec.transform(words), y_train)
score = classifier.score(vec.transform(test_words), y_test)
print(score)

# 使用Tfidf模型训练
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word', max_features=50,  lowercase = False)
vectorizer.fit(words)
classifier = MultinomialNB()
classifier.fit(vectorizer.transform(words), y_train)
sorce = classifier.score(vectorizer.transform(test_words), y_test)
print(score)

# 导入要分类的数据
prodict_data = pd.read_excel('F:\\gujia_content.xlsx')
prodict_data = prodict_data['content']
contents_clean1,all_words1 = handel_cut(prodict_data,stopwords)
contents_clean1 = data_handel(x=contents_clean1)
vectorizer = TfidfVectorizer(analyzer='word', max_features=50,  lowercase = False)
vectorizer.fit(contents_clean1)
classifier = MultinomialNB()
class_f = classifier.predict(contents_clean1)
class_f = pd.DataFrame({'class_f':class_f})
prodict_data['class_f'] = class_f
prodict_data.to_excel('F:\\prodict_data.xlsx')