"""多轮对话式文本 prefix cache 复用测试。

每一轮是一次独立的 llm.generate:第 i 轮的 prompt 是前 i-1 轮完整对话(含
历史回答)再追加一个新用户问题,由 chat template 重新渲染 —— 这正是多轮对话
服务的真实形态:上一轮的「prompt + 回答」已经算完 KV 并留在块哈希表里,本轮
理论上只有新增的尾巴需要真正 prefill。

  Round 1    对话第 1 轮(长 system 上下文 + 问题 1)   期望命中 0(冷启动)
  Round 2    历史对话 + 问题 2                        期望命中 ≈ 上一轮完整流覆盖的块
  Round 3    历史对话 + 问题 3                        以此类推
  ...
  对照轮      全新无共享的对话                          期望命中 0

两个实现细节:

* 块哈希覆盖序列的完整 token 流(prompt + 生成 token),所以期望值对照的是
  历史请求的完整流,而不只是历史 prompt;
* 命中以完整块为单位,prompt 的最后一个块不参与探测,且 chat template 重新
  渲染在回答结尾处可能产生不同的分词,因此期望值按 can_allocate_prefill 的
  逐块比对逻辑在本地重算,和引擎计数器对不上会标 FAIL 并以退出码 1 结束。

用法:
  python example_prefix_cache.py --model /path/to/model
  python example_prefix_cache.py --model /path/to/model --turns 6 --prefix_tokens 4096
"""

import argparse
import sys
from time import perf_counter

from kavor import LLM, SamplingParams
from transformers import AutoTokenizer

# 两段互不相同的确定性填充文本,分别用于长 system 上下文 / 对照轮
FILLER_A = (
    "Kavor 是一个从零开始实现的轻量级推理引擎,整体结构参考了 vLLM: "
    "Engine 负责接收请求并驱动主循环, Scheduler 决定每一步调度哪些序列做 "
    "prefill 或 decode, BlockManager 以固定大小的块管理 KV 页池, "
    "ModelRunner 负责真正的前向计算。页池在启动时按 gpu_memory_utilization "
    "规划大小,块是物理页,block_table 把序列的逻辑块映射到物理页,序列结束 "
    "后页被回收,但块哈希表里的表项会保留,后续请求只要前缀的完整块逐块 "
    "一致,就能直接复用这些物理页里的 KV,跳过对应部分的 prefill 计算。 "
)

FILLER_B = (
    "GPU 显存带宽通常是大规模推理的主要瓶颈,注意力机制在解码阶段需要为 "
    "每个新 token 读取全部历史 token 的键值缓存,读取量随上下文长度线性 "
    "增长,因此 KV 缓存的布局与复用策略对吞吐影响很大。分页 KV 缓存把 "
    "显存切成固定大小的块,按需分配给不同序列,配合前缀哈希索引,可以让 "
    "共享相同系统提示或相同文档上下文的请求复用已算好的键值,把重复的 "
    "prefill 计算变成一次显存引用计数的增加,从而显著降低首 token 延迟。 "
)

TURN_QUESTIONS = [
    "请用一句话概括上面的资料。",
    "上面的资料提到了哪些组件?简单列举。",
    "把这些组件的职责串成一段话。",
    "你觉得其中哪个组件最关键?为什么?",
    "根据上面的资料提一个后续值得研究的问题。",
    "把我们的讨论总结成三个要点。",
]

CONTROL_QUESTION = "忽略之前的内容,介绍一下你自己。"


def build_system(tokenizer, filler: str, num_tokens: int) -> str:
    """把填充文本重复到约 num_tokens 个 token,作为长 system 上下文。"""
    filler_tokens = max(1, len(tokenizer.encode(filler)))
    return filler * max(1, num_tokens // filler_tokens)


def expected_hit_tokens(refs: list[list[int]], prompt_ids: list[int], block_size: int) -> int:
    """按 BlockManager 的探测逻辑估算命中 token 数:与每条历史完整流
    (prompt + 生成 token)从第 0 块起逐块比对,第一块不同即停;本条
    prompt 的最后一个块不参与探测。"""
    num_probe = (len(prompt_ids) + block_size - 1) // block_size - 1  # range(num_blocks - 1)
    best = 0
    for ref in refs:
        hit_blocks = 0
        for i in range(num_probe):
            ref_block = ref[i * block_size:(i + 1) * block_size]
            if len(ref_block) < block_size:
                break  # 历史流在这块没算满,该块从未写入哈希表
            if ref_block == prompt_ids[i * block_size:(i + 1) * block_size]:
                hit_blocks += 1
            else:
                break
        best = max(best, hit_blocks)
    return best * block_size


def run_turn(llm, tokenizer, title: str, messages: list[dict], refs: list[list[int]],
             block_size: int, max_tokens: int) -> tuple[bool, dict]:
    """渲染一轮对话并生成一次,对照引擎命中计数与本地重算的期望。"""
    bm = llm.scheduler.block_manager
    hit_before, prompt_before = bm.prefix_cache_hit_tokens, bm.prefix_cache_prompt_tokens

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt)
    expected = expected_hit_tokens(refs, prompt_ids, block_size)
    sp = SamplingParams(temperature=0.6, max_tokens=max_tokens, ignore_eos=True)

    print(f"\n=== {title} ===")
    print(f"  prompt {len(prompt_ids)} tok, 期望命中 {expected} tok")
    t0 = perf_counter()
    output = llm.generate([prompt], sp, use_tqdm=False)[0]
    elapsed = perf_counter() - t0

    hit = bm.prefix_cache_hit_tokens - hit_before
    prompt_tokens = bm.prefix_cache_prompt_tokens - prompt_before
    computed = prompt_tokens - hit
    rate = 100.0 * hit / prompt_tokens if prompt_tokens else 0.0
    mark = "PASS" if hit == expected else "FAIL"
    print(f"  命中 {hit} / {prompt_tokens} tok ({rate:.1f}%), 实际 prefill {computed} tok, "
          f"耗时 {elapsed:.2f}s -> {mark}")
    print(f"  回答预览: {output['text'][:60].strip()!r}")

    refs.append(prompt_ids + output["token_ids"])   # 完整流进 refs,供下一轮对照
    return hit == expected, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/tanger/workspace/models/Qwen3.5-9B")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--prefix_tokens", type=int, default=4096,
                        help="system 长上下文长度(token);越长可复用的块越多")
    parser.add_argument("--turns", type=int, default=4, help="对话轮数(不含对照轮)")
    parser.add_argument("--max_tokens", type=int, default=64, help="每轮回答长度")
    args = parser.parse_args()
    assert 1 <= args.turns <= len(TURN_QUESTIONS), f"--turns 支持 1~{len(TURN_QUESTIONS)}"

    # 对话逐轮增长,上限约 prefix + turns * (问题 + 回答)
    conv_budget = args.turns * (64 + args.max_tokens + 128)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=args.tp_size,
              max_model_len=max(4096, args.prefix_tokens + conv_budget))
    # 带线性注意力的模型引擎会把 block_size 自动放大(如 Qwen3.5-9B 是 768 而非 256)
    block_size = llm.scheduler.block_manager.block_size
    num_blocks = args.prefix_tokens // block_size
    assert num_blocks >= 1, (
        f"--prefix_tokens={args.prefix_tokens} 不足一个块(block_size={block_size})")

    system = build_system(tokenizer, FILLER_A, args.prefix_tokens)
    print(f"\nsystem 上下文约 {len(tokenizer.encode(system))} tok "
          f"(block_size={block_size}, 约 {num_blocks} 个完整块), 共 {args.turns} 轮对话 + 1 轮对照")

    messages = [{"role": "system", "content": system}]
    refs: list[list[int]] = []
    results = []
    for turn in range(args.turns):
        messages.append({"role": "user", "content": TURN_QUESTIONS[turn]})
        ok, output = run_turn(llm, tokenizer, f"Round {turn + 1}  对话第 {turn + 1} 轮: "
                              f"历史 {turn} 轮 + 新问题", messages, refs, block_size, args.max_tokens)
        messages.append({"role": "assistant", "content": output["text"]})
        results.append(ok)

    control_messages = [
        {"role": "system", "content": build_system(tokenizer, FILLER_B, args.prefix_tokens)},
        {"role": "user", "content": CONTROL_QUESTION},
    ]
    ok, _ = run_turn(llm, tokenizer, "对照轮  全新对话, 与历史无共享",
                     control_messages, refs, block_size, args.max_tokens)
    results.append(ok)

    bm = llm.scheduler.block_manager
    total_rate = 100.0 * bm.prefix_cache_hit_tokens / bm.prefix_cache_prompt_tokens
    print(f"\n累计: 命中 {bm.prefix_cache_hit_tokens} / {bm.prefix_cache_prompt_tokens} tok "
          f"({total_rate:.1f}%)")
    if all(results):
        print("全部 PASS: 命中数与逐块比对的期望一致")
    else:
        print("存在 FAIL: 引擎命中数与期望不符(见上面对照)")
        sys.exit(1)


if __name__ == "__main__":
    main()
