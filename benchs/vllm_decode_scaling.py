"""vLLM 版 Decode Scaling benchmark:与 benchs/decode_scaling.py(Kavor)同口径对比。

用法(conda vllm 环境):
  python benchs/vllm_decode_scaling.py --model /path/to/model --tp 2 --label vllm
  # 跑完后用 Kavor 脚本把两条曲线叠加画图:
  python benchs/decode_scaling.py --plot-only <kavor/result.json> <vllm/result.json>

与 Kavor 的口径对齐:
  - x 轴 per-user decode tok/s: 各请求 (completion-1)/(last_token_ts-first_token_ts)
    的 median。时间戳取自 vLLM RequestStateStats 的 engine core 单调钟
    (output_processor 内部算 decode_time 用的就是这对时间戳),等价于 Kavor 的
    (finish_time-first_token_time),窗口同样含 prefill 对 decode 的干扰;
  - y 轴 decode tok/s per GPU: Σ(纯 decode step 的 generation tokens)/Σ(step 时长)/TP。
    step 级数据来自 vLLM V1 的 SchedulerIterationDetails
    (num_ctx_tokens/num_generation_tokens/elapsed_ms),elapsed_ms 在 engine core
    侧环绕整个 iteration 计时。含 prefill chunk 的 step 整步剔除——vLLM 默认开
    chunked prefill 时 prefill 会与 decode 混步,混步产出的 decode token 记入
    summary.mixed_gen_tokens 不计入 y 轴(与 Kavor 的 is_prefill 过滤同义,
    Kavor 无混步;通常占比 <5%,见 --disable-chunked-prefill);
  - TTFT: first_token_ts-queued_ts(engine core 侧排队+prefill 时延,Kavor 用
    arrival_time,语义近似)。

要拿到 step 级数据需 enable_logging_iteration_details=True,且 LLM 默认强制
disable_log_stats=True 会关掉它,故这里显式传 disable_log_stats=False;同时把
vllm logger 调到 WARNING,避免逐步 INFO 刷屏(钩子不走日志,不受影响)。

产物 result.json 与 Kavor 脚本同 schema(label 默认 vllm),归档目录
benchs/results/decode_scaling_{时间戳}_{label}_{模型}/。
"""

import argparse
import json
import logging
import random
import statistics
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.version import __version__ as VLLM_VERSION

DEFAULT_CONCURRENCY = "1,2,4,8,16,32,64,128,256"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="模型路径(跑 sweep 时必填)")
    parser.add_argument("--label", default="vllm", help="结果标签,进 JSON 与图例")
    parser.add_argument("--tp", type=int, default=1,
                        help="tensor_parallel_size,同时是 per-GPU 归一化分母")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--concurrency", default=DEFAULT_CONCURRENCY,
                        help=f"逗号分隔的并发数列表,默认 {DEFAULT_CONCURRENCY}")
    parser.add_argument("--repeats", type=int, default=3, help="每个并发点重复次数")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--disable-chunked-prefill", action="store_true",
                        help="关闭 chunked prefill,消除 prefill/decode 混步"
                             "(Kavor 无混步;默认保持 vLLM 原生行为)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--no-plot", action="store_true", help="只出 JSON,不画图")
    parser.add_argument("--annotate", action="store_true", help="图上标注并发数 N")
    parser.add_argument("--x-stat", choices=["median", "mean"], default="median",
                        help="x 轴 per-user 速度的聚合方式")
    parser.add_argument("--plot-only", nargs="+", metavar="JSON",
                        help="不跑 bench,从已有结果 JSON 重画/叠加")
    parser.add_argument("--output",
                        help="图表输出目录(默认:run 目录 / 单 JSON 同目录 / 多文件 merged 目录)")
    return parser.parse_args()


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[round(q) - 1]


class StepRecorder:
    """挂到 output_processor.update_scheduler_stats 上,抓每步 iteration details。

    LLMEngine.step() 每步都会调用一次该函数(LLMEngine.py:326),stats 为 None
    或 iteration_details 未填充(0-token step)时跳过。
    """

    def __init__(self, llm: LLM):
        self.steps: list[dict] = []
        proc = llm.llm_engine.output_processor
        self._orig = proc.update_scheduler_stats
        proc.update_scheduler_stats = self._on_stats

    def _on_stats(self, stats):
        if stats is not None:
            d = stats.iteration_details
            if d is not None and not d.is_dummy:
                prefill = d.num_ctx_tokens > 0
                self.steps.append({
                    "prefill": prefill,
                    # prefill 步记 prompt token 数,decode 步记 generation token 数
                    "tokens": d.num_ctx_tokens if prefill else d.num_generation_tokens,
                    "step_time": d.elapsed_ms / 1000.0,
                    # 混步(prefill chunk + decode)中 decode 侧产出,不计入 y 轴
                    "mixed_gen_tokens": d.num_generation_tokens if prefill else 0,
                })
        return self._orig(stats)

    def reset(self):
        self.steps.clear()


def summarize_run(request_stats: list[dict], step_stats: list[dict], tp: int) -> dict:
    """聚合一次 generate 的统计,键名与 Kavor 的 summarize_run 对齐。"""
    decode_steps = [s for s in step_stats if not s["prefill"]]
    prefill_steps = [s for s in step_stats if s["prefill"]]
    decode_tokens = sum(s["tokens"] for s in decode_steps)
    decode_time = sum(s["step_time"] for s in decode_steps)
    prefill_tokens = sum(s["tokens"] for s in prefill_steps)
    prefill_time = sum(s["step_time"] for s in prefill_steps)
    mixed_gen_tokens = sum(s["mixed_gen_tokens"] for s in prefill_steps)

    per_user = []
    tpots_ms = []
    ttfts_s = []
    for r in request_stats:
        n = r["num_completion_tokens"] - 1  # 首个 token 出自 prefill
        t = r["finish_time"] - r["first_token_time"]
        if n > 0 and t > 0:
            per_user.append(n / t)
            tpots_ms.append(1000.0 * t / n)
        if r["first_token_time"] is not None:
            ttfts_s.append(r["first_token_time"] - r["arrival_time"])

    summary = {
        "num_requests": len(request_stats),
        "decode_tokens": decode_tokens,
        "decode_time": decode_time,
        "decode_throughput": decode_tokens / decode_time if decode_time else 0.0,
        "decode_throughput_per_gpu": decode_tokens / decode_time / tp if decode_time else 0.0,
        "per_user_tokens_per_s": {
            "median": statistics.median(per_user),
            "mean": statistics.mean(per_user),
            "p10": _percentile(per_user, 10),
            "p90": _percentile(per_user, 90),
        } if per_user else None,
        "tpot_ms_median": statistics.median(tpots_ms) if tpots_ms else None,
        "ttft_s_median": statistics.median(ttfts_s) if ttfts_s else None,
        "prefill_tokens": prefill_tokens,
        "prefill_time": prefill_time,
        "prefill_throughput": prefill_tokens / prefill_time if prefill_time else 0.0,
        "mixed_gen_tokens": mixed_gen_tokens,
        # 一致性校验:纯 decode step + 混步产出的 token 总数应等于各请求 completion-1 之和
        "consistent": decode_tokens + mixed_gen_tokens == sum(
            r["num_completion_tokens"] - 1 for r in request_stats),
    }
    return summary


def run_sweep(args) -> Path:
    import torch

    llm = LLM(
        args.model,
        tensor_parallel_size=args.tp,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        # iteration details 依赖 log_stats;LLM 默认强制 disable_log_stats=True
        disable_log_stats=False,
        enable_logging_iteration_details=True,
    )
    # 逐步 INFO 日志刷屏,压到 WARNING(钩子不走日志,不受影响)
    logging.getLogger("vllm").setLevel(logging.WARNING)
    recorder = StepRecorder(llm)

    cfg = llm.llm_engine.vllm_config
    cache_cfg, sched_cfg = cfg.cache_config, cfg.scheduler_config
    # KV 容量感知的 sweep 上限(对齐 Kavor 的做法,防超出后抢占轮转破坏稳态);
    # TP>1 时 engine core 在子进程,前端拿不到 num_gpu_blocks 则跳过该截断
    block = cache_cfg.block_size
    num_gpu_blocks = getattr(cache_cfg, "num_gpu_blocks", None) or 0
    effective_cap = sched_cfg.max_num_seqs
    kv_capacity = None
    if num_gpu_blocks:
        blocks_per_seq = -(-(args.input_len + args.output_len) // block)
        kv_capacity = num_gpu_blocks // blocks_per_seq
        effective_cap = min(sched_cfg.max_num_seqs, kv_capacity)
        print(f"[info] sweep 上限: max_num_seqs={sched_cfg.max_num_seqs},"
              f" KV 容量={kv_capacity}({num_gpu_blocks} 块,"
              f"每条序列 {blocks_per_seq} 块),取 {effective_cap}")
    else:
        print("[info] 前端拿不到 num_gpu_blocks,跳过 KV 容量截断,"
              f"sweep 上限取 max_num_seqs={sched_cfg.max_num_seqs}")
    concurrencies = sorted({int(n) for n in args.concurrency.split(",")})
    dropped = [n for n in concurrencies if n > effective_cap]
    if dropped:
        print(f"[warn] 并发 {dropped} 超出上述上限,已从 sweep 中剔除"
              "(超出后 decode 会被时间片轮转/抢占,不再是单批稳态)")
        concurrencies = [n for n in concurrencies if n <= effective_cap]
    if not concurrencies:
        raise SystemExit("没有可跑的并发点")

    # warmup: init 已做图捕获,这里再热一遍采样与 decode 路径
    llm.generate(["Warmup."] * 8,
                 SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=16),
                 use_tqdm=False)

    runs = []
    for n in concurrencies:
        for rep in range(args.repeats):
            # 与 Kavor 相同的种子方案 => 相同的 prompt 集,避免跨 run prefix cache 命中
            rng = random.Random(f"{args.seed}-{n}-{rep}")
            prompts = [[rng.randint(0, 10000) for _ in range(args.input_len)]
                       for _ in range(n)]
            token_prompts = [{"prompt_token_ids": p} for p in prompts]
            sampling_params = SamplingParams(
                temperature=args.temperature, ignore_eos=True, max_tokens=args.output_len)

            recorder.reset()
            outputs = llm.generate(token_prompts, sampling_params, use_tqdm=False)
            assert len(outputs) == n
            assert all(len(o.outputs[0].token_ids) == args.output_len for o in outputs)

            request_stats = [{
                "num_completion_tokens": len(o.outputs[0].token_ids),
                "first_token_time": o.metrics.first_token_ts,
                "finish_time": o.metrics.last_token_ts,
                "arrival_time": o.metrics.queued_ts,
            } for o in outputs]
            step_stats = [dict(s) for s in recorder.steps]
            summary = summarize_run(request_stats, step_stats, args.tp)
            runs.append({
                "concurrency": n,
                "repeat": rep,
                "summary": summary,
                "request_stats": request_stats,
                "step_stats": step_stats,
            })
            per_user = summary["per_user_tokens_per_s"]
            mixed = (f", 混步 decode tokens {summary['mixed_gen_tokens']}"
                     if summary["mixed_gen_tokens"] else "")
            print(f"N={n:>4} rep={rep}: "
                  f"per-user {per_user['median']:7.1f} tok/s (median, "
                  f"TPOT {summary['tpot_ms_median']:.1f} ms), "
                  f"per-GPU {summary['decode_throughput_per_gpu']:8.1f} tok/s, "
                  f"prefill {summary['prefill_throughput']:.0f} tok/s{mixed}")
            if not summary["consistent"]:
                print(f"[warn] N={n} rep={rep}: decode step tokens 与请求侧计数不一致")

    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    result = {
        "benchmark": "decode_scaling",
        "label": args.label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "model_name": Path(args.model).name,
        "tensor_parallel_size": args.tp,
        "gpus": gpu_names,
        "torch": torch.__version__,
        "vllm": VLLM_VERSION,
        "config": {
            "max_num_seqs_requested": args.max_num_seqs,
            "max_num_seqs_effective": sched_cfg.max_num_seqs,
            "kv_capacity_for_workload": kv_capacity,
            "sweep_cap": effective_cap,
            "max_num_batched_tokens": sched_cfg.max_num_batched_tokens,
            "max_model_len": args.max_model_len,
            "kvcache_block_size": block,
            "num_kvcache_blocks": num_gpu_blocks or None,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "chunked_prefill": not args.disable_chunked_prefill,
            "prefix_caching": bool(getattr(cache_cfg, "enable_prefix_caching", False)),
        },
        "workload": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "temperature": args.temperature,
            "seed": args.seed,
            "repeats": args.repeats,
            "concurrency": concurrencies,
        },
        "runs": runs,
    }

    out_dir = Path(args.out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"decode_scaling_{ts}_{args.label}_{result['model_name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "result.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved results: {json_path}")
    return run_dir


def plot_scaling(json_paths: list[str], out_dir: Path,
                 x_stat: str = "median", annotate: bool = False):
    """从结果 JSON 画两张图到 out_dir,每个 JSON 一条曲线(与 Kavor 脚本同实现)。
    - decode_scaling_curve.png: per-user tok/s vs per-GPU tok/s(吞吐-时延权衡)
    - decode_vs_batch.png:      batch size vs per-GPU tok/s(吞吐扩展曲线)
    点为 repeats 均值,误差棒为 std。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = []
    for path in json_paths:
        data = json.loads(Path(path).read_text())
        # 按 concurrency 聚合 repeats:点 = 均值,误差棒 = std
        by_n = {}
        for run in data["runs"]:
            x = run["summary"]["per_user_tokens_per_s"][x_stat]
            y = run["summary"]["decode_throughput_per_gpu"]
            by_n.setdefault(run["concurrency"], ([], []))
            by_n[run["concurrency"]][0].append(x)
            by_n[run["concurrency"]][1].append(y)
        ns = sorted(by_n)
        series.append({
            "data": data,
            "label": (f"{data['label']} ({data['model_name']}, "
                      f"TP{data['tensor_parallel_size']})"),
            "ns": ns,
            "xs": [statistics.mean(by_n[n][0]) for n in ns],
            "ys": [statistics.mean(by_n[n][1]) for n in ns],
            "xerr": [statistics.stdev(by_n[n][0]) if len(by_n[n][0]) > 1 else 0.0
                     for n in ns],
            "yerr": [statistics.stdev(by_n[n][1]) if len(by_n[n][1]) > 1 else 0.0
                     for n in ns],
        })

    if len(series) == 1:
        d = series[0]["data"]
        wl = d["workload"]
        title = (f"Decode Scaling — {d['model_name']} "
                 f"(TP{d['tensor_parallel_size']}, "
                 f"in {wl['input_len']} / out {wl['output_len']})")
    else:
        title = "Decode Scaling — comparison"

    # 图 1: per-user vs per-GPU(吞吐-时延权衡曲线)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for s in series:
        ax.errorbar(s["xs"], s["ys"], xerr=s["xerr"], yerr=s["yerr"],
                    marker="o", capsize=3, label=s["label"])
        if annotate:
            for x, y, n in zip(s["xs"], s["ys"], s["ns"]):
                ax.annotate(f"N={n}", (x, y), textcoords="offset points",
                            xytext=(5, 5), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Per-user decode throughput (tokens/s)")
    ax.set_ylabel("Decode throughput per GPU (tokens/s)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "decode_scaling_curve.png")
    plt.close(fig)

    # 图 2: batch size vs per-GPU(吞吐扩展曲线)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for s in series:
        ax.errorbar(s["ns"], s["ys"], yerr=s["yerr"],
                    marker="o", capsize=3, label=s["label"])
    ax.set_title(title)
    ax.set_xlabel("Batch size (concurrency)")
    ax.set_ylabel("Decode throughput per GPU (tokens/s)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "decode_vs_batch.png")
    plt.close(fig)

    print(f"Saved plots:\n  {out_dir / 'decode_scaling_curve.png'}"
          f"\n  {out_dir / 'decode_vs_batch.png'}")


def main():
    args = parse_args()
    if args.plot_only:
        if args.output:
            out_dir = Path(args.output)
        elif len(args.plot_only) == 1:
            out_dir = Path(args.plot_only[0]).parent
        else:
            out_dir = Path(args.out_dir) / (f"decode_scaling_merged_"
                                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_scaling(args.plot_only, out_dir, x_stat=args.x_stat, annotate=args.annotate)
        return
    if not args.model:
        raise SystemExit("跑 sweep 需要 --model(或用 --plot-only 重画已有结果)")

    run_dir = run_sweep(args)
    if not args.no_plot:
        out_dir = Path(args.output) if args.output else run_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            plot_scaling([str(run_dir / "result.json")], out_dir,
                         x_stat=args.x_stat, annotate=args.annotate)
        except ImportError:
            print("[warn] 本环境无 matplotlib,跳过画图;result.json 已落盘,"
                  "可在 kavor 环境用 decode_scaling.py --plot-only 重画")


if __name__ == "__main__":
    main()
