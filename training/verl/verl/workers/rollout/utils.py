# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import inspect
import logging
import os

import uvicorn
from fastapi import FastAPI

from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__file__)


def get_max_position_embeddings(hf_config) -> int:
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            max_len = getattr(text_config, "max_position_embeddings", None)

    if max_len is None:
        raise ValueError("max_position_embeddings not found in HFModelConfig!")
    return int(max_len)


async def run_unvicorn(
    app: FastAPI,
    server_args,
    server_address,
    max_retries=5,
    start_timeout: float = 30.0,
) -> tuple[int, asyncio.Task]:
    server_port, server_task = None, None

    for i in range(max_retries):
        sock = None
        try:
            server_port, sock = get_free_port(server_address)
            app.server_args = server_args
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server = uvicorn.Server(config)

            serve_kwargs = {}
            if "sockets" in inspect.signature(server.serve).parameters:
                serve_kwargs["sockets"] = [sock]
            else:
                sock.close()
                sock = None

            server_task = asyncio.create_task(server.serve(**serve_kwargs))
            deadline = asyncio.get_running_loop().time() + start_timeout
            while not getattr(server, "started", False):
                if server_task.done():
                    exc = server_task.exception()
                    raise RuntimeError(f"uvicorn server exited during startup: {exc!r}")
                if asyncio.get_running_loop().time() > deadline:
                    server.should_exit = True
                    raise TimeoutError(f"HTTP server did not start within {start_timeout:.1f}s")
                await asyncio.sleep(0.05)
            break
        except (OSError, SystemExit) as e:
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
            if server_task is not None and not server_task.done():
                server_task.cancel()
            if sock is not None:
                sock.close()
        except Exception as e:
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
            if server_task is not None and not server_task.done():
                server_task.cancel()
            if sock is not None:
                sock.close()
    else:
        logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
        os._exit(-1)

    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task


async def ensure_async_iterator(iterable):
    """Convert an iterable to an async iterator."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item
