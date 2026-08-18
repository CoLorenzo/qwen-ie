import argparse
import base64
import contextlib
import io
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/qwen_ie/config")
DEFAULT_MODEL = "Qwen/Qwen-Image-Edit-2511"
FP8_REPO = "1038lab/Qwen-Image-Edit-2511-FP8"
FP8_FILE = "Qwen-Image-Edit-2511-FP8_e4m3fn.safetensors"
DEFAULT_PORT = 17070
DEFAULT_HOST = "127.0.0.1"


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
            "  qwen-ie --image a.png --image b.png --prompt \"uniscili\" -o combo.png --seed 42\n"
            "\n"
            "Modalita server (modello caricato una sola volta, riusato per piu' richieste):\n"
            "  qwen-ie --serve --port 17070\n"
            "  qwen-ie --client --image foto.png --prompt \"...\" -o output.png\n"
            "  pkill -f \"[q]wen-ie --serve\"   # per fermare il server"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--serve", action="store_true",
                       help="avvia un server HTTP che carica il modello una volta sola e resta in ascolto")
    mode.add_argument("--client", action="store_true",
                       help="invia una richiesta a un server avviato con --serve invece di caricare il modello")
    p.add_argument("--image", action="append", default=None,
                   help="immagine in input (ripetibile, multi-immagine; richiesto tranne con --serve)")
    p.add_argument("--prompt", default=None, help="istruzione di editing (richiesto tranne con --serve)")
    p.add_argument("-o", "--output", default="output.png", help="file di output (default: output.png)")
    p.add_argument("--model", default=None,
                   help="repo HuggingFace del modello (default: dal config, poi " + DEFAULT_MODEL + "; ignorato con --client)")
    p.add_argument("--steps", type=int, default=None,
                   help="passi di denoising (default: dal config, poi 20)")
    p.add_argument("--cfg", type=float, default=None,
                   help="true_cfg_scale (default: dal config, poi 4.0)")
    p.add_argument("--seed", type=int, default=None, help="seed per riproducibilita")
    p.add_argument("--negative", default=None,
                   help="negative prompt (default: dal config, poi ' ')")
    p.add_argument("--quant", choices=["fp8", "nf4"], default=None,
                   help="quantizzazione: fp8 (pesi FP8 pre-quantizzati, default, ~20GB) o nf4 (bitsandbytes 4-bit, ~11GB; ignorato con --client)")
    p.add_argument("--lightning", action="store_true", default=None,
                   help="modalita veloce: 4 step e cfg 1.0")
    p.add_argument("--port", type=int, default=None,
                   help=f"porta del server, per --serve/--client (default: dal config, poi {DEFAULT_PORT})")
    p.add_argument("--host", default=None,
                   help=f"host del server, per --serve/--client (default: dal config, poi {DEFAULT_HOST})")
    args = p.parse_args()

    if not args.serve:
        if not args.image:
            p.error("--image e' richiesto (tranne con --serve)")
        if not args.prompt:
            p.error("--prompt e' richiesto (tranne con --serve)")

    return args


def resolve_model(args, cfg):
    model = args.model or os.environ.get("QWEN_IE_MODEL") or cfg.get("model") or DEFAULT_MODEL
    quant = args.quant or cfg.get("quantization", "fp8")
    return {"model": model, "quant": quant}


def resolve_gen(args, cfg):
    lightning = to_bool(cfg.get("lightning", "false")) if args.lightning is None else args.lightning

    if lightning:
        steps = args.steps if args.steps is not None else 4
        cfg_scale = args.cfg if args.cfg is not None else 1.0
    else:
        steps = args.steps if args.steps is not None else int(cfg.get("steps", 20))
        cfg_scale = args.cfg if args.cfg is not None else float(cfg.get("cfg", 4.0))

    negative = args.negative if args.negative is not None else (cfg.get("negative") or " ")
    return {"lightning": lightning, "steps": steps, "cfg": cfg_scale, "negative": negative}


def resolve(args, cfg):
    return {**resolve_model(args, cfg), **resolve_gen(args, cfg)}


def resolve_host_port(args, cfg):
    port = args.port if args.port is not None else int(cfg.get("port", DEFAULT_PORT))
    host = args.host if args.host is not None else (cfg.get("host") or DEFAULT_HOST)
    return host, port


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
        from huggingface_hub import hf_hub_download, list_repo_files
        from safetensors.torch import load_file

        config_repo = model_id
        weights_repo, weights_file = FP8_REPO, FP8_FILE
        try:
            config = QwenImageTransformer2DModel.load_config(config_repo, subfolder="transformer")
        except OSError:
            # model_id e' un repo che contiene solo i pesi FP8 (es. quello di FP8_REPO), senza
            # la struttura completa della pipeline: usalo come sorgente dei pesi e ricadi sul
            # repo base ufficiale per config/pipeline.
            print(f"'{model_id}' non ha una config di pipeline completa: uso i suoi pesi FP8 "
                  f"e la config da {DEFAULT_MODEL}")
            weights_repo = model_id
            weights_file = next(
                (f for f in list_repo_files(model_id) if f.endswith(".safetensors")),
                FP8_FILE,
            )
            config_repo = DEFAULT_MODEL
            config = QwenImageTransformer2DModel.load_config(config_repo, subfolder="transformer")

        print(f"Scaricamento pesi FP8 ({weights_repo}/{weights_file})...")
        fp8_path = hf_hub_download(weights_repo, weights_file)
        print("Caricamento pesi FP8...")
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
            config_repo, transformer=transformer, dtype=torch.bfloat16)

    pipe.enable_model_cpu_offload()
    return pipe


def run_edit(pipe, images, prompt, steps, cfg_scale, negative, seed, lock=None):
    inputs = {
        "image": images,
        "prompt": prompt,
        "num_inference_steps": steps,
        "true_cfg_scale": cfg_scale,
        "negative_prompt": negative,
    }
    if seed is not None:
        inputs["generator"] = torch.Generator("cpu").manual_seed(seed)

    ctx = lock if lock is not None else contextlib.nullcontext()
    with ctx, torch.inference_mode():
        return pipe(**inputs).images[0]


def run_oneshot(args, cfg):
    s = resolve(args, cfg)

    images = []
    for path in args.image:
        if not os.path.isfile(path):
            sys.exit(f"Errore: immagine non trovata: {path}")
        images.append(Image.open(path).convert("RGB"))

    print(f"Modello: {s['model']}  | quant: {s['quant']}  | steps: {s['steps']}  | cfg: {s['cfg']}")
    pipe = load_pipeline(s["model"], s["quant"])

    print(f"Editing in corso ({s['steps']} step)...")
    out = run_edit(pipe, images, args.prompt, s["steps"], s["cfg"], s["negative"], args.seed)

    out.save(args.output)
    print(f"OK -> {args.output}")


def run_server(args, cfg):
    m = resolve_model(args, cfg)
    host, port = resolve_host_port(args, cfg)

    print(f"Modello: {m['model']}  | quant: {m['quant']}")
    pipe = load_pipeline(m["model"], m["quant"])
    lock = threading.Lock()
    print(f"Modello caricato. In ascolto su http://{host}:{port} (POST /edit, GET /health)")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/edit":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                images = [
                    Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
                    for b in payload["images"]
                ]
                prompt = payload["prompt"]
                steps = int(payload.get("steps", 20))
                cfg_scale = float(payload.get("cfg", 4.0))
                negative = payload.get("negative") or " "
                seed = payload.get("seed")

                print(f"[server] editing: {prompt!r} ({len(images)} immagini, {steps} step)")
                out = run_edit(pipe, images, prompt, steps, cfg_scale, negative, seed, lock=lock)

                buf = io.BytesIO()
                out.save(buf, format="PNG")
                data = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                print("[server] OK")
            except Exception as e:
                print(f"[server] errore: {e}")
                msg = str(e).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

        def log_message(self, fmt, *a):
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def run_client(args, cfg):
    g = resolve_gen(args, cfg)
    host, port = resolve_host_port(args, cfg)

    images_b64 = []
    for path in args.image:
        if not os.path.isfile(path):
            sys.exit(f"Errore: immagine non trovata: {path}")
        with open(path, "rb") as f:
            images_b64.append(base64.b64encode(f.read()).decode("ascii"))

    payload = {
        "prompt": args.prompt,
        "images": images_b64,
        "steps": g["steps"],
        "cfg": g["cfg"],
        "negative": g["negative"],
    }
    if args.seed is not None:
        payload["seed"] = args.seed

    url = f"http://{host}:{port}/edit"
    print(f"Invio richiesta a {url} ...")
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            out_bytes = resp.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"Errore server: {e.read().decode('utf-8', errors='replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"Errore: impossibile contattare il server su {host}:{port} ({e.reason}). "
                  f"E' in esecuzione 'qwen-ie --serve --port {port}'?")

    with open(args.output, "wb") as f:
        f.write(out_bytes)
    print(f"OK -> {args.output}")


def main():
    args = parse_args()
    cfg = load_config(os.environ.get("QWEN_IE_CONFIG", DEFAULT_CONFIG_PATH))

    if args.serve:
        run_server(args, cfg)
    elif args.client:
        run_client(args, cfg)
    else:
        run_oneshot(args, cfg)


if __name__ == "__main__":
    main()

