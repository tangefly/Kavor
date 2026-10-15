import argparse
import json
import traceback


def add_server_args(parser: argparse.ArgumentParser):
    """serve 子命令的参数定义(vLLM 风格:模型路径为位置参数)。"""
    parser.add_argument("model", type=str, help="本地模型目录(只接受本地路径)")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_false")
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="允许加载 tokenizer 时执行模型目录里的自定义代码。模型 config 由 Kavor 自己解析,不受此开关影响")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--mm-max-pixels", type=int, default=1280 * 28 * 28,
        help="多模态请求里每张图(视频为每帧)的像素上限,超了会等比缩小;"
             "不设上限时模型自己会用 16777216(打平正好 16384 个视觉 token,顶满 max-num-batched-tokens)")
    parser.add_argument(
        "--default-chat-template-kwargs", type=json.loads, default=None,
        help='模板的默认变量(JSON),比如 \'{"enable_thinking": true}\'。'
             '请求里的 chat_template_kwargs 优先级更高')
    

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="kavor", description="Kavor CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="启动 OpenAI 兼容 API 服务")
    add_server_args(serve_parser)

    args = parser.parse_args(argv)
    if args.command == "serve":
        try:
            from kavor.interface.openai import api_server
        except ImportError as e:
            if e.name in ("fastapi", "uvicorn"):
                raise SystemExit(
                    f"serve 需要 {e.name},请执行: pip install kavor[serve]") from e
            traceback.print_exc()
            raise SystemExit(f"serve 代码导入失败: {type(e).__name__}: {e}") from e
        api_server.run(args)
    

if __name__ == "__main__":
    main()
