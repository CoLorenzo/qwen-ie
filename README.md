# qwen-ie

CLI for image editing from textual instructions based on Qwen-Image-Edit.

## Installation

```bash
uv tool install git+https://github.com/CoLorenzo/qwen-ie
```

Requires an NVIDIA GPU with CUDA drivers installed (the package downloads
`torch` from PyPI; for CUDA-specific builds see the
[official PyTorch guide](https://pytorch.org/get-started/locally/)).

Alternatively, [`install.sh`](install.sh) bootstraps `uv` (if missing),
installs the `hf` command (`huggingface_hub[cli]`, useful for `hf auth login`
on gated models) and finally `qwen-ie`:

```bash
curl -LsSf https://raw.githubusercontent.com/CoLorenzo/qwen-ie/main/install.sh | sh
```

Or via `git clone`:

```bash
# Install qwen-ie if not present
if ! command -v qwen-ie >/dev/null 2>&1; then
	sudo apt update
	sudo apt install -y git
	git clone https://github.com/CoLorenzo/qwen-ie
	cd qwen-ie
	./install.sh
	cd ..
	rm -drf qwen-ie
fi
```

To update:

```bash
uv tool upgrade qwen-ie
```

To uninstall:

```bash
uv tool uninstall qwen-ie
```

## Usage

```bash
qwen-ie --image foto.png --prompt "change the background to a sunset"
qwen-ie --image a.png --image b.png --prompt "merge them" -o combo.png --seed 42
```

Main options:

- `--image` (repeatable): input image, multiple allowed
- `--prompt`: editing instruction
- `-o/--output`: output file (default `output.png`)
- `--model`: HuggingFace model repo (default `Qwen/Qwen-Image-Edit-2511`)
- `--steps`: denoising steps (default 20, 4 with `--lightning`)
- `--cfg`: true_cfg_scale (default 4.0, 1.0 with `--lightning`)
- `--seed`: seed for reproducibility
- `--negative`: negative prompt
- `--quant`: `fp8` (default, pre-quantized weights, ~20GB) or `nf4` (bitsandbytes 4-bit, ~11GB)
- `--lightning`: fast mode (4 steps, cfg 1.0)

## Configuration

Defaults can be overridden in `~/.config/qwen_ie/config`
(`key=value` format, see [`qwen_ie/config.example`](qwen_ie/config.example)).
Precedence: CLI flags > `QWEN_IE_MODEL` environment variable > config file > defaults.

The config file path can be changed with the `QWEN_IE_CONFIG` environment
variable.