#!/bin/bash
set -e

pip install uv
"$HOME/.local/bin/uv" sync --frozen
