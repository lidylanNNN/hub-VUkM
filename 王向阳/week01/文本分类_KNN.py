# 导入结巴分词库，用于中文文本的分词处理
import jieba
# 导入pandas库，用于数据读取和处理
import pandas as pd
# 从sklearn导入CountVectorizer，用于将文本转换为词频特征矩阵
from sklearn.feature_extraction.text import CountVectorizer
# 从sklearn导入KNeighborsClassifier，即K近邻分类模型/KNN
from sklearn.neighbors import KNeighborsClassifier


# 读取整个表格
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)

# sklearn对中文处理
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# 始化CountVectorizer对象（词频特征提取器）：用于将文本转换为词频矩阵
vector = CountVectorizer()

# 统计所有出现的词语，生成词汇表
vector.fit(input_sentence.values)

# 将分词后的文本转换为词频特征矩阵
# 矩阵维度：行数=数据集文本数量 × 列数=词表大小（每列对应一个词语，值为该词在文本中的出现次数）
input_feature = vector.transform(input_sentence.values)

# 使用KNN模型
model = KNeighborsClassifier()

# 训练KNN模型，将词频特征矩阵与数据集第1列的真实标签进行关联学习
model.fit(input_feature, dataset[1].values)

# 预测表格中所有文本的类型
all_pred_results = model.predict(input_feature)

# 遍历表格每一行，输出原始文本和对应的预测类型
for idx in range(len(dataset)):
    # dataset[0][idx] 是表格第一列的文本，all_pred_results[idx] 是对应预测结果
    print("待预测的文本", dataset[0][idx])
    print("KNN模型预测结果: ", all_pred_results[idx])
    print("-" * 50)
