#!/usr/bin/env bash
set -euo pipefail

# Install huggingface_cli if it is not already installed.
if ! command -v hf >/dev/null 2>&1; then
	# Install uv if it is not already installed.
	if ! command -v uv >/dev/null 2>&1; then
		sudo apt update
		sudo apt install -y curl
		curl -LsSf https://astral.sh/uv/install.sh | sh
		source ~/.bashrc
		uv tool update-shell
	fi
	
	uv tool install huggingface_hub[cli]
fi

#Install diffusers if not present
if ! command -v diffusers >/dev/null 2>&1; then
	# Install uv if it is not already installed.
	if ! command -v uv >/dev/null 2>&1; then
		sudo apt update
		sudo apt install -y curl
		curl -LsSf https://astral.sh/uv/install.sh | sh
		source ~/.bashrc
	fi
	uv tool install git+https://github.com/huggingface/diffusers
fi

uv tool install git+https://github.com/CoLorenzo/qwen-ie
mv ./qwen_ie ~/.config/

