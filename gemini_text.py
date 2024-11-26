import google.generativeai as genai
import json

with open('config.json', 'r') as file:
    config = json.load(file)
geminikey = config.get("mygeminiapi")
genai.configure(api_key=geminikey)

generation_config = {
    "temperature": 2, #1
    "top_p": 0.95, #0.95
    "top_k": 100,   #64
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

response = model.generate_content('請生成一題性格測驗情況是非問題, 回應只需要是非問題本身')
#print(model.count_tokens('請'))
print(response.text)