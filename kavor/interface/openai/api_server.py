import argparse
import asyncio
import json
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import time

from kavor.engine.async_engine import AsyncEngine
from kavor.interface.cli import add_server_args
from kavor.sampling_params import SamplingParams
from kavor.interface.parser.factory import *

from .request import *
from .response import *


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="api_server")
    add_server_args(parser)  # 参数定义统一放在 cli.py,kavor serve 子命令复用
    return parser.parse_args(argv)

def to_sampling_params(req: ChatCompletionRequest, args: argparse.Namespace) -> SamplingParams:
    # OpenAI 习惯 temperature=0 表示 greedy,但 SamplingParams 禁止 <= 1e-10,映射成 epsilon
    if args.temperature:
        req.temperature = args.temperature
    if args.max_model_len:
        req.max_tokens = args.max_model_len
    temperature = req.temperature if req.temperature > 1e-10 else 1e-6
    return SamplingParams(
        temperature=temperature,
        max_tokens=req.max_tokens
    )

async def consume_all(queue: asyncio.Queue) -> tuple[list[int], str]:
    token_ids: list[int] = []
    finish_reason = "stop"
    while True:
        item = await queue.get()
        if item is None:
            return token_ids, finish_reason
        kind, payload = item
        if kind == "delta":
            token_ids.extend(payload)
        elif kind == "finish":
            finish_reason = payload


async def stream_sse(queue: asyncio.Queue, tokenizer):
    sent_text = ""
    token_ids: list[int] = []
    while True:
        item = await queue.get()
        if item is None:
            yield "data: [DONE]\n\n"
            return
        kind, payload = item
        if kind == "delta":
            token_ids.extend(payload)
            new_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            delta = new_text[len(sent_text):]
            if delta:
                sent_text = new_text
                yield f"data: {json.dumps({'text': delta}, ensure_ascii=False)}\n\n"
        elif kind == "finish":
            yield f"data: {json.dumps({'finish_reason': payload}, ensure_ascii=False)}\n\n"

def build_app(args: argparse.Namespace) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = AsyncEngine(
            args.model,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
        )
        engine.loop = asyncio.get_running_loop()
        app.state.engine = engine
        app.state.args = args
        app.state.paser = get_paser(args.model)
        yield
        engine.exit()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        if app.state.engine.is_alive():
            return {"status": "ok"}
        return JSONResponse(status_code=503, content={"status": "engine thread dead"})

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": args.model,
                "object": "model",
                "owned_by": "kavor",
            }],
        }

    @app.post("/generate")
    async def generate(req: ChatCompletionRequest):
        if not req.prompt:
            raise HTTPException(status_code=400, detail="prompt 不能为空")
        request_id = req.request_id or uuid.uuid4().hex
        queue = await app.state.engine.add_request(
            request_id, req.prompt, to_sampling_params(req, app.state.args))

        if req.stream:
            return StreamingResponse(
                stream_sse(queue, app.state.engine.tokenizer),
                media_type="text/event-stream",
            )

        token_ids, finish_reason = await consume_all(queue)
        return {
            "request_id": request_id,
            "text": app.state.engine.tokenizer.decode(token_ids, skip_special_tokens=True),
            "token_ids": token_ids,
            "num_completion_tokens": len(token_ids),
            "finish_reason": finish_reason,
        }
        
    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if req.model is not app.state.engine.model_name:
            pass
        
        request_id = uuid.uuid4().hex
        
        messages = []
        for item in req.messages:
            msg = {"role": item.role, "content": item.content}
            if item.tool_calls:
                msg["tool_calls"] = item.tool_calls
            messages.append(msg)
        
        prompt = app.state.engine.tokenizer.apply_chat_template(
            messages,
            tools=req.tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        queue = await app.state.engine.add_request(
            request_id, prompt, to_sampling_params(req, app.state.args))

        token_ids, finish_reason = await consume_all(queue)
        raw_content = app.state.engine.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=req.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        content=app.state.paser.extract_content(raw_content),
                        tool_calls=app.state.paser.parse_tool_calls(raw_content),
                        reasoning_content=app.state.paser.parse_thinking(raw_content)
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=len(token_ids),
                total_tokens=0,
            ),
        )
        
        return response

    return app

def run(args: argparse.Namespace):
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port)

def main(argv: list[str] | None = None):
    run(parse_args(argv))

if __name__ == "__main__":
    main()
