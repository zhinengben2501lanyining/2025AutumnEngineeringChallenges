import requests
import json

# 测试模型列表
response = requests.get("http://localhost:8000/v1/models")
print("Available models:", response.json())

# 测试文本生成
data = {
    "model": "Qwen/Qwen2-0.5B",
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
}

response = requests.post(
    "http://localhost:8000/v1/completions",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)
)

print("Generation result:", response.json())