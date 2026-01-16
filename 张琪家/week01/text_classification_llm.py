from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-833c7e2xxxxx39a3bcf301adf4", # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

test_text = "张家辉是一个很好的演员"

completion = client.chat.completions.create(
        model="qwen-flash",  # 模型的代号

        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{test_text}
                输出的类别只能从如下中进行选择， 除了类别之外不要回答其他内容，请给出最合适的类别。
                FilmTele-Play            
                Video-Play               
                Music-Play              
                Radio-Listen           
                Alarm-Update        
                Travel-Query        
                HomeAppliance-Control  
                Weather-Query          
                Calendar-Query      
                TVProgram-Play      
                Audio-Play       
                Other             
                """},  # 用户的提问
                ]
    )
print(completion.choices[0].message.content)
