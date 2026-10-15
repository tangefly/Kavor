"""图文推理示例:和 example.py 只差在 prompt 怎么造。

图片三种给法,区别只在怎么写 "image" 字段的值:
  本地路径        -> PIL.Image(open_image 帮你打开)
  data:...;base64 -> 字符串原样传,load_image 负责解码
  http(s)://      -> 字符串原样传,load_image 负责下载(服务端会去访问这个 URL)
"""
import argparse
import os

from PIL import Image

from kavor import LLM, SamplingParams

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))


def open_image(src: str):
    """本地路径 → PIL.Image;data:/http(s): 开头的字符串原样交给 load_image。"""
    if src.startswith(("data:", "http://", "https://")):
        return src
    return Image.open(src).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/public/home/xiaoxunpeng/Models/Qwen3.5-4B")
    parser.add_argument("--image", type=str, default=os.path.join(EXAMPLE_DIR, "test.png"))
    parser.add_argument("--prompt", type=str, default="描述这张图片")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    args = parser.parse_args()

    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=args.tp_size)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    requests = [
        [{"role": "user", "content": [
            {"type": "image", "image": open_image(args.image)},
            {"type": "text", "text": args.prompt},
        ]}]
    ]

    prompts = []
    for messages in requests:
        if isinstance(messages[0]["content"], list):
            prompts.append(llm.build_mm_inputs(messages, max_pixels=args.max_pixels))
        else:
            prompts.append(llm.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt.prompt if hasattr(prompt, 'prompt') else prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
