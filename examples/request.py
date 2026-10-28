from openai import OpenAI
import base64

client = OpenAI(
    base_url="http://localhost:25541/v1",
    api_key="EMPTY",
)

# 图片转 base64
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


image_base64 = encode_image("image.png")

response = client.chat.completions.create(
    model="Qwen3-VL-4B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                },
                {
                    "type": "text",
                    "text": "提取图片文字",
                },
            ],
        }
    ],
    temperature=0.7,
    max_tokens=1024,
)

print(response.choices[0].message.reasoning)
print(response.choices[0].message.content)