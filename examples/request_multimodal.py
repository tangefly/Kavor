import base64
import os
import sys

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:25541/v1",
    api_key="EMPTY",
)

# 视频转 base64
def encode_video(video_path):
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_base64 = encode_image("./data/command.png")
video_base64 = encode_video("./data/bbb_3s_720p.mp4")

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
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:video/mp4;base64,{video_base64}"
                    },
                },
                {
                    "type": "text",
                    "text": "分别描述这个视频和图片展示了什么。",
                },
            ],
        }
    ],
    temperature=0.7,
    max_tokens=1024,
)

print(response.choices[0].message.reasoning)
print(response.choices[0].message.content)
