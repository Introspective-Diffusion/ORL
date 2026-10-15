#!/usr/bin/env python3
"""HTTP client example; first start a local server with your explicit model/config.

From the repository root:
  python -m examples.generate --api sglang --prompt 'Hello' \
      --sampling-params '{"max_new_tokens": 128, "temperature": 0}'

This is a generic request example, not a paper decoder or benchmark protocol.
The server must already be running on loopback. No weights are downloaded here.
"""
from inference.generate import main


if __name__ == '__main__':
    main()
