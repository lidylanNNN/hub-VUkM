"""
作业要求：
    使用dataset.csv数据集完成文本分类操作，需要尝试2种不同的模型
"""

import jieba
import torch
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("dataset.csv", sep="\t", names=["text", "label"])
# print(dataset.shape)
# print(dataset.head(5))

# 划分X, y
text, label = dataset["text"], dataset["label"]
text = text.apply(lambda x: " ".join(jieba.lcut(x)))
# print(len(label.unique())) # 12
print(label.unique())

# 划分训练集、测试集，确保可复现,还需要确保训练集和测试集中类别比例一致
train_x, test_x, train_y, test_y = train_test_split(text, label, test_size=0.2, random_state=42)

# 文本向量化

# count_vector = CountVectorizer()
# count_vector.fit(train_x.values)
# text_feature_train = count_vector.transform(train_x.values)
# text_feature_test = count_vector.transform(test_x.values)
# print(text_feature_train.shape)

# CountVectorizer向量化后的准确不高
# knn模型中，k=1时预测正确率: 73.93%
# knn模型中，k=3时预测正确率: 71.03%
# knn模型中，k=5时预测正确率: 71.57%
# knn模型中，k=7时预测正确率: 71.61%
# knn模型中，k=9时预测正确率: 70.25%

tfidf_vector = TfidfVectorizer()
tfidf_vector.fit(train_x)
text_feature_train = tfidf_vector.transform(train_x)
text_feature_test = tfidf_vector.transform(test_x)
print(text_feature_train.shape)

# TfidfVectorizer的结果如下：
# knn模型中，k=1时预测正确率: 81.20%
# knn模型中，k=3时预测正确率: 82.19%
# knn模型中，k=5时预测正确率: 84.17%
# knn模型中，k=7时预测正确率: 84.83%
# knn模型中，k=9时预测正确率: 84.88%


# 搭建 KNN 模型并训练, 选择最好的 k 值
best_knn_model = None
best_acc = 0
for k in [1, 3, 5, 7, 9]:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(text_feature_train, train_y)
    prediction = knn_model.predict(text_feature_test)
    acc = ((test_y == prediction).sum() / len(prediction))
    if acc > best_acc:
        best_acc = acc
        best_knn_model = knn_model
    # 保留两位小数
    print(f"knn模型中，k={k}时预测正确率: {acc * 100:.2f}%")

# 朴素贝叶斯
nb_model = MultinomialNB()
nb_model.fit(text_feature_train, train_y)
# 评估
nb_acc = ((test_y == nb_model.predict(text_feature_test)).sum() / len(test_y))
print(f"朴素贝叶斯模型预测正确率: {nb_acc * 100:.2f}%")
# 朴素贝叶斯模型预测正确率: 87.73%

def text_cls_using_knn(text: str) -> str:
    """
    文本分类（KNN），输入文本完成类别划分
    """
    text_sentence = " ".join(jieba.lcut(text))
    text_feature = tfidf_vector.transform([text_sentence])
    return best_knn_model.predict(text_feature)[0]

def text_cls_using_nb(text: str) -> str:
    """
    文本分类（朴素贝叶斯），输入文本完成类别划分
    """
    text_sentence = " ".join(jieba.lcut(text))
    text_feature = tfidf_vector.transform([text_sentence])
    return nb_model.predict(text_feature)[0]

client = OpenAI(
    # 填充正确的api-key
    api_key="sk-xxxxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_cls_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    try:
        completion = client.chat.completions.create(
            model="qwen-flash",  # 模型的代号

            messages=[
                {"role": "user", "content": f"""对该文本进行分类：{text}。\n 
                文本的类别只从如下选项中进行选择：{"、".join(label.unique())}。\n
                无需多余回答，只给出类别。"""}
            ]
        )

        return completion.choices[0].message.content
    except Exception as e:
        return f"分类失败：{str(e)}"

if __name__ == "__main__":
    test_case1 = "随便放一首歌"
    test_case2 = "帮我导航到天安门"
    test_case3 = "今天天气怎么样，会不会下雨"
    test_case4 = "给我找一个魔兽世界的比赛视频"

    print("\n" + "=" * 50)
    print("测试用例1：", test_case1)
    print("KNN分类结果：", text_cls_using_knn(test_case1))
    print("朴素贝叶斯分类结果：", text_cls_using_nb(test_case1))
    print("LLM分类结果：", text_cls_using_llm(test_case1))

    print("\n" + "=" * 50)
    print("测试用例2：", test_case2)
    print("KNN分类结果：", text_cls_using_knn(test_case2))
    print("朴素贝叶斯分类结果：", text_cls_using_nb(test_case2))
    print("LLM分类结果：", text_cls_using_llm(test_case2))

    print("\n" + "=" * 50)
    print("测试用例3：", test_case3)
    print("KNN分类结果：", text_cls_using_knn(test_case3))
    print("朴素贝叶斯分类结果：", text_cls_using_nb(test_case3))
    print("LLM分类结果：", text_cls_using_llm(test_case3))

    print("\n" + "=" * 50)
    print("测试用例4：", test_case4)
    print("KNN分类结果：", text_cls_using_knn(test_case4))
    print("朴素贝叶斯分类结果：", text_cls_using_nb(test_case4))
    print("LLM分类结果：", text_cls_using_llm(test_case4))
