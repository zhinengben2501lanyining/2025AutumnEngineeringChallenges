import requests
import json

class HunyuanMT7BClient:
    def __init__(self, base_url="http://10.242.3.186:8000", model_name="Tencent/Hunyuan-MT-7B"):
        """
        初始化 Tencent/Hunyuan-MT-7B 客户端
        
        函数所需:
            base_url (str): vLLM 服务器地址，默认为 http://10.242.3.186:8000
            model_name (str): 模型名称，与启动参数 --served-model-name 一致
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.completions_url = f"{self.base_url}/v1/completions"
        self.chat_completions_url = f"{self.base_url}/v1/chat/completions"
        
    def translate_via_chat(self, text, source_lang="中文", target_lang="英文", temperature=0, max_tokens=512):
        """
        通过 /v1/chat/completions 终结点进行翻译
        
        函数所需:
            text (str): 需要翻译的文本
            source_lang (str): 文本语言
            target_lang (str): 目标语言
            temperature (float): 生成温度，控制随机性
            max_tokens (int): 生成的最大 token 数
            
        返回结果:
            str: 模型输出的结果
        """
        # 请求头:cite[4]:cite[8]
        headers = {
            "Content-Type": "application/json"
        }
        
        # 请求体:cite[5]:cite[9]
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system", 
                    "content": f"你是一名专业的翻译官，请将用户提供的{source_lang}文本准确流畅地翻译成{target_lang}。"
                },
                {
                    "role": "user",
                    "content": f"请将以下{source_lang}内容翻译成{target_lang}：{text}"
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            # 发送 POST 请求:cite[1]:cite[5]
            response = requests.post(self.chat_completions_url, headers=headers, json=payload)
            response.raise_for_status()  # 请求失败抛出异常
            
            result = response.json()
            # 提取结果
            translated_text = result['choices'][0]['message']['content']
            return translated_text
            
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return None
        except KeyError as e:
            print(f"解析响应失败: {e}")
            print(f"原始响应: {response.text}")
            return None
    
    def translate_via_completion(self, text, source_lang="中文", target_lang="英文", temperature=0, max_tokens=512):
        """
        通过 /v1/completions 终结点进行翻译
        
        函数所需:
            text (str): 需要翻译的文本
            source_lang (str): 文本语言
            target_lang (str): 目标语言
            temperature (float): 生成温度，控制随机性
            max_tokens (int): 生成的最大 token 数
            
        返回结果:
            str: 模型输出的结果
        """
        # 请求头
        headers = {
            "Content-Type": "application/json"
        }
        
        # 提示词
        prompt = f"请将以下{source_lang}文本翻译成{target_lang}：{text}"
        
        # 请求体
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            # 发送 POST 请求
            response = requests.post(self.completions_url, headers=headers, json=payload)
            response.raise_for_status()  # 请求失败抛出异常
            
            result = response.json()
            # 提取结果
            translated_text = result['choices'][0]['text']
            return translated_text
            
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return None
        except KeyError as e:
            print(f"解析响应失败: {e}")
            print(f"原始响应: {response.text}")
            return None

if __name__ == "__main__":
    # 初始化客户端
    client = HunyuanMT7BClient()
    
    # 需要翻译的文本
    text_to_translate = "山东师范大学机器人与人工智能实验室 成立于 2021 年，立足于 山东省人工智能基地，致力于机器人技术与人工智能领域的多元化研究。"
    
    print("正在发送翻译请求...")
    print(f"待翻译文本: {text_to_translate}")
    
    # 使用聊天终结点进行翻译
    print("\n使用 /v1/chat/completions 终结点：")
    translated_text = client.translate_via_chat(text_to_translate)
    
    if translated_text:
        print(f"翻译结果: {translated_text}")
    else:
        print("翻译失败")
    
    # 使用补全终结点进行翻译
    print("\n使用 /v1/completions 终结点：")
    translated_text = client.translate_via_completion(text_to_translate)
    
    if translated_text:
        print(f"翻译结果: {translated_text}")
    else:
        print("翻译失败")