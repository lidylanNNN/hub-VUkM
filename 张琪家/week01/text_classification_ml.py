import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# 使用pandas读取csv数据集
dataset = pd.read_csv('dataset.csv', sep='\t', header=None)
# print(dataset)

# dataset[0] : 文本， dataset[1] : 标签
# 使用jieba将文本分词

text_token = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
# print(text_token)
# 词频统计器
counter = CountVectorizer()
# 统计(统计出所有的词)
counter.fit(text_token)
# 根据统计出的词将数据集向量化
# print(text_token.values)  text_token.values将表格变为列表
text_vector = counter.transform(text_token.values)

# 训练模型
model1 = KNeighborsClassifier()
model1.fit(text_vector, dataset[1].values)

model2 = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
model2.fit(text_vector, dataset[1].values)

model3 = MultinomialNB(alpha=1.0)
model3.fit(text_vector, dataset[1].values)

# 测试用例
test_text = "张家辉是一个很好的演员"
test_vector = counter.transform([" ".join(jieba.lcut(test_text))])
print("测试文本：", test_text)

# 预测结果
res1 = model1.predict(test_vector)
res2 = model2.predict(test_vector)
res3 = model3.predict(test_vector)
print("knn模型结果：", res1)
print("逻辑回归模型结果：", res2)
print("朴素贝叶斯模型结果：", res3)


