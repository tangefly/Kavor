import argparse


def add_server_args(parser: argparse.ArgumentParser):
    """serve 子命令的参数定义(vLLM 风格:模型路径为位置参数)。"""
    parser.add_argument("model", type=str, help="本地模型目录(只接受本地路径)")
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="kavor", description="Kavor CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="启动 OpenAI 兼容 API 服务")
    add_server_args(serve_parser)

    args = parser.parse_args(argv)
    if args.command == "serve":
        try:
            from kavor.entrypoints.openai import api_server
        except ImportError as e:
            raise SystemExit(
                "serve 需要 fastapi/uvicorn,请执行: pip install kavor[serve]") from e
        api_server.run(args)
    

if __name__ == "__main__":
    main()
