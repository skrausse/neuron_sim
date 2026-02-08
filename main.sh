#!/bin/bash

# && ensures we only proceed if the previous command succeeded
cargo run --release && \
cd analysis/ && \
uv run python main.py