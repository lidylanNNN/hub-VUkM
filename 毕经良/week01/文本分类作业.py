# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import jieba         # 中文分词工具
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计
from sklearn.neighbors import KNeighborsClassifier             # K近邻算法
from openai import OpenAI  # 用于调用OpenAI API或兼容模式API

# 加载数据集，使用制表符作为分隔符，无列标题，限制读取前10000行
# 假设第一列是文本数据，第二列是标签
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print("dataset: ===============")
print(dataset)
# 打印标签分布情况，了解各类别样本数量
print("dataset[1].value_counts(): ===============")
print(dataset[1].value_counts())

# 使用jieba对中文文本进行分词，并用空格连接成字符串，便于CountVectorizer处理
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
print("input_sententce: ===============")
print(input_sententce)
# 创建词袋模型向量化器，将文本转换为数值特征
vector = CountVectorizer()

# 拟合向量化器，构建词汇表
vector.fit(input_sententce.values)

# 将文本数据转换为特征向量矩阵
# 矩阵形状为 (样本数, 词表大小)，每个元素表示对应词在文档中的出现次数
input_feature = vector.transform(input_sententce.values)
print("input_feature: ===============")
print(input_feature)
# 创建K近邻分类器实例
model = KNeighborsClassifier()
print("dataset[1].values: ===============")
print(dataset[1].values)
# 使用训练数据拟合模型
model.fit(input_feature, dataset[1].values)  # 特征向量和对应的标签


# 配置OpenAI客户端，这里使用阿里云的兼容模式API
client = OpenAI(
    # 如果没有配置环境变量，可以在此处直接指定API密钥
    # 获取API密钥地址：https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-eda26f20c01*************",  # 用于账户绑定和计费

    # 指定API服务提供商的基础URL，这里是阿里云的服务地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_calssify_using_ml(text: str) -> str:
    # 对输入文本进行分词并用空格连接，使其符合训练时的格式
    test_sentence = " ".join(jieba.lcut(text))
    
    # 将预处理后的文本转换为特征向量
    test_feature = vector.transform([test_sentence])
    
    # 使用训练好的模型进行预测
    return model.predict(test_feature)[0]

def text_calssify_using_llm(text: str) -> str:
    # 调用大语言模型进行文本分类
    completion = client.chat.completions.create(
        model="qwen-flash",  # 指定使用的模型名称

        messages=[
            {
                "role": "user", 
                "content": f"""帮我进行文本分类：{text}

            请根据以下类别列表，为上述文本选择最合适的类别：
            FilmTele-Play            (影视播放)
            Video-Play               (视频播放)
            Music-Play              (音乐播放)
            Radio-Listen           (收听广播)
            Alarm-Update        (闹钟更新)
            Travel-Query        (旅行查询)
            HomeAppliance-Control  (家电控制)
            Weather-Query          (天气查询)
            Calendar-Query      (日历查询)
            TVProgram-Play      (电视节目播放)
            Audio-Play       (音频播放)
            Other             (其他)
            """},  # 提供给模型的分类任务指令
        ]
    )
    # 返回模型生成的分类结果
    return completion.choices[0].message.content

if __name__ == "__main__":
    # 主程序入口，用于演示两种文本分类方法
    # pandas 用于表格数据的加载和分析
    # numpy 用于矩阵运算和数值计算
    
    # 测试文本
    test_text = "帮我导航到天安门"
    
    # 使用机器学习方法进行分类
    ml_result = text_calssify_using_ml(test_text)
    print(f"机器学习方法分类结果: {ml_result}")
    
    # 使用大语言模型进行分类
    llm_result = text_calssify_using_llm(test_text)
    print(f"大语言模型分类结果: {llm_result}")
