"""视频推理示例:同一段视频连着问几个问题,从第二问开始整段视频前缀都走 prefix cache。

和 example_multimodal.py 只差在 prompt 里放的是视频;另外这里显式给了抽帧帧率
(--fps)和原视频 fps,不然后 processor 会按 24fps 猜,抽帧个数和时间戳都会偏。

视频解码走 ffmpeg(这个环境里没有 torchcodec/decord/av),见 video_frames_from_path。
3s 视频按 2fps 抽 6 帧 = 3 个时域 patch(T=帧数/2),720p 每个时域 patch
880 个 vision token,总共两千多 token;抽太密会直接被 max_num_batched_tokens
挡住(多模态 prefill 不切分)。

    python examples/example_video.py --video examples/data/bbb_3s_720p.mp4
"""

import argparse
import os

from transformers.video_utils import VideoMetadata

from kavor import LLM, SamplingParams
from kavor.engine.multimodal import probe_video, video_frames_from_path

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(EXAMPLE_DIR)

DEFAULT_QUESTIONS = [
    "描述这个视频里发生了什么。",
    "画面里最主要的颜色是什么?",
    "镜头是在向哪个方向运动?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="/public/home/xiaoxunpeng/Models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--video", type=str,
                        default=os.path.join(REPO_DIR, "examples/data/bbb_3s_720p.mp4"))
    parser.add_argument("--question", action="append", dest="questions", default=None,
                        help="可以给多次,默认三个问题(用来看 prefix cache 命中)")
    parser.add_argument("--fps", type=float, default=2.0, help="抽帧帧率")
    parser.add_argument("--max_frames", type=int, default=None, help="最多解码多少帧")
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=16,
                        help="演示用 16 就够;CUDA graph 按这个值决定捕获档位,调小启动更快")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=128)
    args = parser.parse_args()

    questions = args.questions or DEFAULT_QUESTIONS

    info = probe_video(args.video)
    frames = video_frames_from_path(args.video, max_frames=args.max_frames)
    print(f"视频: {args.video} {info['width']}x{info['height']} "
          f"{info['fps']:.2f}fps 共 {len(frames)} 帧,按 {args.fps}fps 抽帧")

    llm = LLM(args.model, enforce_eager=False, tensor_parallel_size=args.tp_size,
              max_num_seqs=args.max_num_seqs)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    # 同一段视频 + 不同问题:视频那一段 token 完全一样,块哈希也带上像素摘要(mm_hash_id),
    # 所以第二问开始视频前缀直接命中,不用再跑视觉塔、不用重算那两千多个 token 的 KV。
    processor_kwargs = {
        "videos_kwargs": {"fps": args.fps},
        "video_metadata": [VideoMetadata(fps=info["fps"], total_num_frames=len(frames))],
    }
    inputs = []
    for question in questions:
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": question},
        ]}]
        inputs.append(llm.build_inputs(
            messages, max_pixels=args.max_pixels, processor_kwargs=processor_kwargs))

    first = inputs[0]
    video_tokens = sum(n for _, n in (first.mm_inputs.video_pad_ranges or []))
    print(f"prompt {len(first.token_ids)} token,其中视频占位符 {video_tokens} 个,"
          f"视频 grid(按时域 patch 拆开){first.mm_inputs.video_grid_thw.tolist()}")

    # 一问一答地提交:同一批 prefill 的请求互相看不到对方的 KV,只有前一问跑完、
    # 块哈希已经写进 prefix cache 之后,后一问才可能整段命中视频前缀。
    for question, item in zip(questions, inputs):
        output = llm.generate([item], sampling_params)[0]
        print(f"\n问: {question}\n答: {output['text']!r}")

    block_manager = llm.scheduler.block_manager
    if block_manager.prefix_cache_prompt_tokens:
        print(f"\nprefix cache: 命中 {block_manager.prefix_cache_hit_tokens}/"
              f"{block_manager.prefix_cache_prompt_tokens} token "
              f"({100.0 * block_manager.prefix_cache_hit_tokens / block_manager.prefix_cache_prompt_tokens:.1f}%)")
    llm.exit()


if __name__ == "__main__":
    main()
