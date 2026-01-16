import pandas as pd
import jieba
# 以下模块可以提取文本的特征
from sklearn.feature_extraction.text import CountVectorizer
# 导入模型knn
from sklearn.neighbors import KNeighborsClassifier
# 导入模型svm
from sklearn.svm import SVC

# 加载数据集
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
# print(dataset)
# 想要统计去重之后指标有哪些，可以使用value_counts()方法
# print(dataset[1].value_counts())

# 数据集处理
# 分词
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
# print(input_sentence)
# print(input_sentence.values)
# 提取特征
vector = CountVectorizer()
vector.fit(input_sentence.values)   #该代码的作用是生成特征矩阵，本质上就是生成每个词出现的次数，也就是词频
# print(vector.vocabulary_)
input_feature = vector.transform(input_sentence.values)     #统计去重之后有多少个词
# print('input_feature:',input_feature)

# 创建模型进行训练-KNN
model_knn = KNeighborsClassifier()
model_knn.fit(input_feature, dataset[1].values)
# 创建其他模型进行训练-SVM
model_svm = SVC()
model_svm.fit(input_feature, dataset[1].values)

# 测试输入
test_query = "帮我导航到月球"
print('输入:', test_query)
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
# 使用模型进行预测
knn_predict = model_knn.predict(test_feature)
print('knn模型预测的结果为:', knn_predict)
svm_predict = model_svm.predict(test_feature)
print('svm模型预测的结果为:', svm_predict)
