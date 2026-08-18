import argparse
import os
import re
import sys

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/qwen_ie/config")
DEFAULT_MODEL = "Qwen/Qwen-Image-Edit-2511"
FP8_REPO = "1038lab/Qwen-Image-Edit-2511-FP8"
FP8_FILE = "Qwen-Image-Edit-2511-FP8_e4m3fn.safetensors"


def load_config(path):
    cfg = {}
    if not path:
        return cfg
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return cfg
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            cfg[k.strip().lower()] = v.strip()
    return cfg


def to_bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def parse_args():
    p = argparse.ArgumentParser(
        description="CLI per Qwen-Image-Edit: editing di immagini da istruzioni testuali",
        epilog=(
            "Configurazione: ~/.config/qwen_ie/config (formato chiave=valore).\n"
            "Precedenza: flag CLI > env QWEN_IE_MODEL > file di config > default.\n"
            "Esempi:\n"
            "  qwen-ie --image foto.png --prompt \"cambia lo sfondo con un tramonto\"\n"
            "  qwen-ie --image a.png --image b.png --prompt \"uniscili\" -o combo.png --seed 42"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image", action="append", required=True,
                   help="immagine in input (ripetibile, multi-immagine)")
    p.add_argument("--prompt", required=True, help="istruzione di editing")
    p.add_argument("-o", "--output", default="output.png", help="file di output (default: output.png)")
    p.add_argument("--model", default=None,
                   help="repo HuggingFace del modello (default: dal config, poi " + DEFAULT_MODEL + ")")
    p.add_argument("--steps", type=int, default=None,
                   help="passi di denoising (default: dal config, poi 20)")
    p.add_argument("--cfg", type=float, default=None,
                   help="true_cfg_scale (default: dal config, poi 4.0)")
    p.add_argument("--seed", type=int, default=None, help="seed per riproducibilita")
    p.add_argument("--negative", default=None,
                   help="negative prompt (default: dal config, poi ' ')")
    p.add_argument("--quant", choices=["fp8", "nf4"], default=None,
                   help="quantizzazione: fp8 (pesi FP8 pre-quantizzati, default, ~20GB) o nf4 (bitsandbytes 4-bit, ~11GB)")
    p.add_argument("--lightning", action="store_true", default=None,
                   help="modalita veloce: 4 step e cfg 1.0")
    return p.parse_args()


def resolve(args, cfg):
    model = args.model or os.environ.get("QWEN_IE_MODEL") or cfg.get("model") or DEFAULT_MODEL
    quant = args.quant or cfg.get("quantization", "fp8")
    lightning = to_bool(cfg.get("lightning", "false")) if args.lightning is None else args.lightning

    if lightning:
        steps = args.steps if args.steps is not None else 4
        cfg_scale = args.cfg if args.cfg is not None else 1.0
    else:
        steps = args.steps if args.steps is not None else int(cfg.get("steps", 20))
        cfg_scale = args.cfg if args.cfg is not None else float(cfg.get("cfg", 4.0))

    negative = args.negative if args.negative is not None else (cfg.get("negative") or " ")
    return {
        "model": model, "quant": quant, "lightning": lightning,
        "steps": steps, "cfg": cfg_scale, "negative": negative,
    }


def resolve_diffusers_bnb():
    import diffusers
    for name in ("BitsAndBytesConfig",):
        cls = getattr(diffusers, name, None)
        if cls is not None:
            return cls
    q = getattr(diffusers, "quantizers", None)
    if q is not None:
        for name in ("DiffusersBitsAndBytesConfig", "BitsAndBytesConfig"):
            cls = getattr(q, name, None)
            if cls is not None:
                return cls
    return None


def materialize_meta_tensors(model):
    """Materializza su CPU tutti i tensori meta rimasti nel modello.

    I pos_freqs/neg_freqs del rotary embed sono attributi plain (non buffer registrati,
    per non perdere la parte immaginaria complex64) e non sono nel file FP8: su
    costruzione meta device restano meta. Li ricomputiamo con la stessa formula
    dell'__init__ (rope_params + theta/axes_dim del modulo). Per eventuali altri
    tensori meta ricrea tensori con forma e dtype corretti e stampa un warning.
    """
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenEmbedLayer3DRope,
        QwenEmbedRope,
    )

    for name, module in model.named_modules():
        if isinstance(module, (QwenEmbedRope, QwenEmbedLayer3DRope)):
            if getattr(module.pos_freqs, "is_meta", False) or getattr(module.neg_freqs, "is_meta", False):
                print(f"Ricreazione freqs rope per {name} (theta={module.theta}, axes_dim={module.axes_dim})")
                pos_index = torch.arange(4096)
                neg_index = torch.arange(4096).flip(0) * -1 - 1
                module.pos_freqs = torch.cat(
                    [module.rope_params(pos_index, dim, module.theta) for dim in module.axes_dim], dim=1
                )
                module.neg_freqs = torch.cat(
                    [module.rope_params(neg_index, dim, module.theta) for dim in module.axes_dim], dim=1
                )
                continue
        for attr, value in list(vars(module).items()):
            if isinstance(value, torch.Tensor) and value.is_meta:
                print(
                    f"Warning: tensore meta {name}.{attr} shape={tuple(value.shape)} dtype={value.dtype} "
                    f"ricreato su CPU"
                )
                setattr(module, attr, torch.empty(value.shape, dtype=value.dtype))

    meta_params = [(n, tuple(p.shape)) for n, p in model.named_parameters() if p.is_meta]
    meta_buffers = [(n, tuple(b.shape)) for n, b in model.named_buffers() if b.is_meta]
    meta_attrs = []
    for name, module in model.named_modules():
        for attr, value in vars(module).items():
            if isinstance(value, torch.Tensor) and value.is_meta:
                meta_attrs.append(f"{name}.{attr}")
    if meta_params or meta_buffers or meta_attrs:
        sys.exit(
            f"Errore: tensori meta rimasti nel transformer: "
            f"parametri={meta_params} buffer={meta_buffers} attributi={meta_attrs}"
        )
    print("OK: nessun tensore meta nel transformer (parametri, buffer e attributi)")


def load_pipeline(model_id, quant):
    if quant == "nf4":
        bnb_cls = resolve_diffusers_bnb()
        if bnb_cls is None:
            sys.exit("Errore: --quant nf4 richiede BitsAndBytesConfig di diffusers, non trovato su questa versione")
        from transformers import Qwen2_5_VLForConditionalGeneration
        from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=bnb_cls(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
            ),
            dtype=torch.bfloat16,
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            subfolder="text_encoder",
            quantization_config=TransformersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            dtype=torch.bfloat16,
        )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_id, transformer=transformer, text_encoder=text_encoder, dtype=torch.bfloat16)
    else:  # fp8 (default): pesi FP8 pre-quantizzati caricati direttamente
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        print(f"Scaricamento pesi FP8 ({FP8_REPO}/{FP8_FILE})...")
        fp8_path = hf_hub_download(FP8_REPO, FP8_FILE)
        print("Caricamento pesi FP8...")
        config = QwenImageTransformer2DModel.load_config(model_id, subfolder="transformer")
        state_dict = load_file(fp8_path, device="cpu")
        with torch.device("meta"):
            transformer = QwenImageTransformer2DModel.from_config(config)
        result = transformer.load_state_dict(state_dict, strict=False, assign=True)
        del state_dict
        if result.missing_keys:
            print(f"Attenzione: {len(result.missing_keys)} chiavi mancanti")
        if result.unexpected_keys:
            print(f"Attenzione: {len(result.unexpected_keys)} chiavi inattese")
        materialize_meta_tensors(transformer)
        if not hasattr(transformer, "enable_layerwise_casting"):
            sys.exit("Errore: enable_layerwise_casting non disponibile su questa versione di diffusers")
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        skip_patterns = ("pos_embed", "patch_embed", "norm", "^proj_in$", "^proj_out$")
        skip_re = [re.compile(p) for p in skip_patterns]
        with torch.no_grad():
            for name, param in transformer.named_parameters():
                if param.data.dtype != torch.float8_e4m3fn:
                    continue
                if name.startswith("transformer_blocks.") and not any(
                    r.search(name) for r in skip_re
                ):
                    continue
                param.data = param.data.to(dtype=torch.bfloat16)
            for name, buf in transformer.named_buffers():
                if buf.data.dtype != torch.float8_e4m3fn:
                    continue
                if name.startswith("transformer_blocks.") and not any(
                    r.search(name) for r in skip_re
                ):
                    continue
                buf.data = buf.data.to(dtype=torch.bfloat16)
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_id, transformer=transformer, dtype=torch.bfloat16)

    pipe.enable_model_cpu_offload()
    return pipe


def main():
    args = parse_args()
    cfg = load_config(os.environ.get("QWEN_IE_CONFIG", DEFAULT_CONFIG_PATH))
    s = resolve(args, cfg)

    images = []
    for path in args.image:
        if not os.path.isfile(path):
            sys.exit(f"Errore: immagine non trovata: {path}")
        images.append(Image.open(path).convert("RGB"))

    print(f"Modello: {s['model']}  | quant: {s['quant']}  | steps: {s['steps']}  | cfg: {s['cfg']}")
    pipe = load_pipeline(s["model"], s["quant"])

    inputs = {
        "image": images,
        "prompt": args.prompt,
        "num_inference_steps": s["steps"],
        "true_cfg_scale": s["cfg"],
        "negative_prompt": s["negative"],
    }
    if args.seed is not None:
        inputs["generator"] = torch.Generator("cpu").manual_seed(args.seed)

    print(f"Editing in corso ({s['steps']} step)...")
    with torch.inference_mode():
        out = pipe(**inputs).images[0]

    out.save(args.output)
    print(f"OK -> {args.output}")


if __name__ == "__main__":
    main()

