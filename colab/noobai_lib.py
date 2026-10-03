# -*- coding: utf-8 -*-
"""
NoobAI-XL EPS 1.1 – bộ cài tự động cho Google Colab (nền Stable Diffusion WebUI Forge).

File này là "nguồn duy nhất" của toàn bộ logic cài đặt. Notebook Colab chỉ:
  1. Ghi cấu hình (form) ra JSON
  2. Ghi file này ra /content/noobai_lib.py
  3. Gọi lần lượt: install() -> download() -> self_test() -> launch()

Có thể chạy độc lập ngoài Colab (dùng để test):
  NOOBAI_ROOT=/tmp/noob python noobai_lib.py all
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------------------
# Hằng số / tài nguyên
# --------------------------------------------------------------------------------------
LIB_VERSION = "1.5.0"

ROOT = Path(os.environ.get("NOOBAI_ROOT", "/content")).resolve()
CONFIG_PATH = Path(os.environ.get("NOOBAI_CONFIG", str(ROOT / "noobai_config.json")))

FORGE_REPO = "https://github.com/lllyasviel/stable-diffusion-webui-forge"
# Commit đã được kiểm thử với bộ cài này (26/06/2025). Đặt FORGE_COMMIT="" trong config để lấy bản mới nhất.
FORGE_COMMIT_TESTED = "dfdcbab685e57677014f05a3309b48cc87383167"

PYTHON_VERSION = "3.10"  # Forge chỉ hỗ trợ 3.7–3.11; Colab hiện là 3.12/3.13 nên phải tạo env riêng.
TORCH_SPEC_GPU = "torch==2.3.1 torchvision==0.18.1"
TORCH_INDEX_GPU = "https://download.pytorch.org/whl/cu121"
XFORMERS_SPEC = "xformers==0.0.27"
CLIP_PKG = "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip"
OPENCLIP_PKG = "https://github.com/mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip"

TAG_CSV_NAME = "danbooru_e621_merged_2026-10-01_pt20-ia-dd-ed-spc.csv"
TAG_CSV_URLS = [
    f"https://github.com/manhlee1196-boop/anime2/raw/main/{TAG_CSV_NAME}",
    f"https://raw.githubusercontent.com/manhlee1196-boop/anime2/main/{TAG_CSV_NAME}",
    # GitHub API trả nội dung raw (hoạt động cả khi raw.githubusercontent.com bị chặn)
    f"https://api.github.com/repos/manhlee1196-boop/anime2/contents/{TAG_CSV_NAME}?ref=main",
]

# Bản dịch tiếng Việt cho giao diện Forge (file nằm cạnh noobai_lib.py trong Colab, hoặc tải từ repo)
VI_LOCALE_NAME = "vi_VN"
VI_LOCALE_URLS = [
    f"https://github.com/manhlee1196-boop/anime2/raw/main/colab/{VI_LOCALE_NAME}.json",
    f"https://raw.githubusercontent.com/manhlee1196-boop/anime2/main/colab/{VI_LOCALE_NAME}.json",
    f"https://api.github.com/repos/manhlee1196-boop/anime2/contents/colab/{VI_LOCALE_NAME}.json?ref=main",
]

# build_notebook.py sẽ chèn nội dung colab/vi_VN.json vào đây để notebook tự chứa bản dịch (không cần tải thêm)
VI_LOCALE_EMBEDDED: dict | None = None
# build_notebook.py sẽ chèn mã nguồn colab/noob_tools.py (script Forge: độ phân giải chuẩn + tự fix) vào đây
NOOB_TOOLS_EMBEDDED: str | None = None
NOOB_TOOLS_NAME = "noob_tools.py"
NOOB_TOOLS_URLS = [
    f"https://github.com/manhlee1196-boop/anime2/raw/main/colab/{NOOB_TOOLS_NAME}",
    f"https://raw.githubusercontent.com/manhlee1196-boop/anime2/main/colab/{NOOB_TOOLS_NAME}",
    f"https://api.github.com/repos/manhlee1196-boop/anime2/contents/colab/{NOOB_TOOLS_NAME}?ref=main",
]

# Độ phân giải chuẩn (bucket ~1 MP của SDXL / NoobAI-XL) – dùng cho form notebook và script noob_tools
STANDARD_RESOLUTIONS = {
    "832x1216 (dọc 2:3 – chuẩn NoobAI)": (832, 1216),
    "1216x832 (ngang 3:2)": (1216, 832),
    "1024x1024 (vuông)": (1024, 1024),
    "896x1152 (dọc 7:9)": (896, 1152),
    "1152x896 (ngang 9:7)": (1152, 896),
    "768x1344 (dọc 9:16)": (768, 1344),
    "1344x768 (ngang 16:9)": (1344, 768),
    "640x1536 (dọc 5:12)": (640, 1536),
    "1536x640 (ngang 12:5)": (1536, 640),
}

HF = "https://huggingface.co"

# (tên file đích, url, kích thước byte, sha256) – lấy từ HuggingFace API, dùng để kiểm tra sau khi tải.
MODEL_FILES = {
    "noobai_xl_eps_1_1": dict(
        kind="Stable-diffusion",
        name="NoobAI-XL-v1.1.safetensors",
        url=f"{HF}/Laxhar/noobai-XL-1.1/resolve/main/NoobAI-XL-v1.1.safetensors",
        size=7105349958,
        sha256="6681e8e4b134c81f16533acedb0d406d7e5e366e1624b4105178c64d00b05d51",
    ),
}
VAE_FILES = {
    "sdxl_vae_fp16_fix": dict(
        kind="VAE",
        name="sdxl_vae_fp16_fix.safetensors",
        url=f"{HF}/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors",
        size=334641162,
        sha256="235745af8d86bf4a4c1b5b4f529868b37019a10f7c0b2e79ad0abca3a22bc6e1",
    ),
}
UPSCALER_FILES = {
    "4x_ultrasharp": dict(
        kind="ESRGAN", name="4x-UltraSharp.pth",
        url=f"{HF}/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth",
        size=66961958, sha256="a5812231fc936b42af08a5edba784195495d303d5b3248c24489ef0c4021fe01",
    ),
    "4x_animesharp": dict(
        kind="ESRGAN", name="4x-AnimeSharp.pth",
        url=f"{HF}/Kim2091/AnimeSharp/resolve/main/4x-AnimeSharp.pth",
        size=67010245, sha256="e7a7de2dafd7331c1992862bbbcd9e9712a9f9f8e6303f0aaa59b4341d359bab",
    ),
}
# ControlNet: bản fp16 (2.5 GB/model). Forge đọc trực tiếp định dạng diffusers.
_EU = f"{HF}/Eugeoter"
CONTROLNET_FILES = {
    "union_promax": dict(
        kind="ControlNet", name="xinsir_controlnet_union_sdxl_promax.safetensors",
        url=f"{HF}/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors",
        size=2513342408, sha256="9fae2e50cb431bfcbe05822b59ec2228df545ef27f711dea8949e9f4ed9f7cdc",
    ),
    "noob_canny": dict(
        kind="ControlNet", name="noob_sdxl_controlnet_canny_fp16.safetensors",
        url=f"{_EU}/noob-sdxl-controlnet-canny/resolve/main/diffusion_pytorch_model.fp16.safetensors",
        size=2502139136, sha256="68d4115d0f97ff59f3444aa06bd81a6c1e96c8a3750d8f6b442aa0495cc89508",
    ),
    "noob_openpose": dict(
        kind="ControlNet", name="noob_sdxl_controlnet_openpose.safetensors",
        url=f"{HF}/Laxhar/noob_openpose/resolve/main/openpose_pre.safetensors",
        size=2502140008, sha256="918cfc4d7def165ea7a06bee10841fe525332515f8a0486c5cb2a171ec40b873",
    ),
    "noob_depth_midas": dict(
        kind="ControlNet", name="noob_sdxl_controlnet_depth_midas_v1-1_fp16.safetensors",
        url=f"{_EU}/noob-sdxl-controlnet-depth_midas-v1-1/resolve/main/diffusion_pytorch_model.fp16.safetensors",
        size=2502139136, sha256="678b94ef3260744a1e9e50932ef8895e7f7fabf25eec7fd5cc96298f7220fe50",
    ),
    "noob_lineart_anime": dict(
        kind="ControlNet", name="noob_sdxl_controlnet_lineart_anime_fp16.safetensors",
        url=f"{_EU}/noob-sdxl-controlnet-lineart_anime/resolve/main/diffusion_pytorch_model.fp16.safetensors",
        size=2502139136, sha256="44eae6a514a60ae426ff966ecdbb9e170ad63d8889456982bb29037df42b86a8",
    ),
    "noob_tile": dict(
        kind="ControlNet", name="noob_sdxl_controlnet_tile_fp16.safetensors",
        url=f"{_EU}/noob-sdxl-controlnet-tile/resolve/main/diffusion_pytorch_model.fp16.safetensors",
        size=2502139136, sha256="d6d676f29153357ac8d57727e35b8edc090aa4de57e861b878872ff45c432615",
    ),
    "noob_scribble_pidinet": dict(
        kind="ControlNet", name="noob_sdxl_controlnet_scribble_pidinet_fp16.safetensors",
        url=f"{_EU}/noob-sdxl-controlnet-scribble_pidinet/resolve/main/diffusion_pytorch_model.fp16.safetensors",
        size=2502139136, sha256="6ba55d75e34f63d0538c2dc1006415c495f254624c1c826df7dfaf4bc172ee47",
    ),
    "noob_softedge_hed": dict(
        kind="ControlNet", name="noob_sdxl_controlnet_softedge_hed_fp16.safetensors",
        url=f"{_EU}/noob-sdxl-controlnet-softedge_hed/resolve/main/diffusion_pytorch_model.fp16.safetensors",
        size=2502139136, sha256="c540e9bb474d7b090cee5f45017ceba2d58775c71bc0f6eafa410b435b1dfd92",
    ),
    "noob_manga_line": dict(
        kind="ControlNet", name="noob_sdxl_controlnet_manga_line_fp16.safetensors",
        url=f"{_EU}/noob-sdxl-controlnet-manga_line/resolve/main/diffusion_pytorch_model.fp16.safetensors",
        size=2502139136, sha256="6f2283231fc00f389b20fd46287185b8e9d8c3e647e056ba48268c45b931d213",
    ),
}
# ADetailer (tải sẵn để dùng được ngay, không chờ HF lúc gen)
ADETAILER_FILES = {
    "face_yolov8n": dict(kind="adetailer", name="face_yolov8n.pt",
                         url=f"{HF}/Bingsu/adetailer/resolve/main/face_yolov8n.pt", size=6230011,
                         sha256="70b640f8f60b1cf0dcc72f30caf3da9495eb2fb6509da48c53374ad6806e6a9c"),
    "face_yolov8s": dict(kind="adetailer", name="face_yolov8s.pt",
                         url=f"{HF}/Bingsu/adetailer/resolve/main/face_yolov8s.pt", size=22507707,
                         sha256="c7237eff25787377de196961140ceaed324d859ee8de5a775d93d33a0e3fab78"),
    "hand_yolov8n": dict(kind="adetailer", name="hand_yolov8n.pt",
                         url=f"{HF}/Bingsu/adetailer/resolve/main/hand_yolov8n.pt", size=6237883,
                         sha256="3991202eb69e9ddcb3b9ba80cdeb41e734ffaf844403d6c9f47d515cd88c6f29"),
    "person_yolov8n_seg": dict(kind="adetailer", name="person_yolov8n-seg.pt",
                               url=f"{HF}/Bingsu/adetailer/resolve/main/person_yolov8n-seg.pt", size=6777003,
                               sha256="38fc8aaae97cb6e70be4ec44770005b26ed473471362afcda62a0037d7ccf432"),
}
CONTROLNET_SETS = {
    "none": [],
    "union_promax": ["union_promax"],
    "noob_core": ["noob_canny", "noob_openpose", "noob_depth_midas", "noob_lineart_anime", "noob_tile"],
    "noob_all": list(CONTROLNET_FILES.keys()),
}

EXTENSIONS = {
    "tagcomplete": ("a1111-sd-webui-tagcomplete", "https://github.com/DominikDoom/a1111-sd-webui-tagcomplete"),
    "adetailer": ("adetailer", "https://github.com/Bing-su/adetailer"),
    "image_browser": ("sd-webui-infinite-image-browsing", "https://github.com/zanllp/sd-webui-infinite-image-browsing"),
    "ultimate_upscale": ("ultimate-upscale-for-automatic1111", "https://github.com/Coyote-A/ultimate-upscale-for-automatic1111"),
    "dynamic_prompts": ("sd-dynamic-prompts", "https://github.com/adieyal/sd-dynamic-prompts"),
    "civitai_helper": ("Stable-Diffusion-Webui-Civitai-Helper", "https://github.com/zixaphir/Stable-Diffusion-Webui-Civitai-Helper"),
    "wd14_tagger": ("stable-diffusion-webui-wd14-tagger", "https://github.com/picobyte/stable-diffusion-webui-wd14-tagger"),
    "regional_prompter": ("sd-webui-regional-prompter", "https://github.com/hako-mikan/sd-webui-regional-prompter"),
}

# Prompt khuyến nghị chính thức của NoobAI-XL (EPS)
NOOB_POSITIVE_PREFIX = "masterpiece, best quality, newest, absurdres, highres, "
NOOB_NEGATIVE = (
    "worst quality, old, early, low quality, lowres, signature, username, logo, bad hands, "
    "mutated hands, mammal, anthro, furry, ambiguous form, feral, semi-anthro"
)


# --------------------------------------------------------------------------------------
# Cấu hình
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    # Tài nguyên
    download_model: bool = True
    download_vae: bool = True
    download_upscalers: bool = True
    controlnet: str = "union_promax"          # none | union_promax | noob_core | noob_all
    extra_model_urls: str = ""                # cách nhau bằng dấu phẩy / xuống dòng
    extra_lora_urls: str = ""
    extra_embedding_urls: str = ""
    civitai_token: str = ""
    hf_token: str = ""
    verify_sha256: bool = False
    # Extensions
    extensions: list = field(default_factory=lambda: [
        "tagcomplete", "adetailer", "image_browser", "ultimate_upscale", "dynamic_prompts", "civitai_helper",
    ])
    use_repo_tag_csv: bool = True
    # Google Drive
    mount_drive: bool = True
    save_outputs_to_drive: bool = True
    link_drive_models: bool = True
    drive_folder: str = "NoobAI"
    # Giao diện / tunnel
    tunnel: str = "gradio.live"               # gradio.live | ngrok | both | none
    ngrok_token: str = ""
    gradio_auth: str = ""                     # "user:pass"
    theme: str = "dark"
    ui_language: str = "vi"                   # vi (tiếng Việt) | en (English)
    port: int = 7860
    use_xformers: bool = True
    gpu_mode: str = "max"                     # max (dùng tối đa GPU) | balanced (mặc định Forge) | lowvram (ảnh/batch lớn)
    extra_args: str = ""
    forge_commit: str = FORGE_COMMIT_TESTED
    # Mặc định UI – param_mode: recommended (dùng bộ NoobAI chuẩn, bỏ qua các default_* bên dưới) | custom
    param_mode: str = "recommended"
    default_width: int = 832
    default_height: int = 1216
    default_steps: int = 28
    default_cfg: float = 6.0
    default_sampler: str = "Euler a"
    default_scheduler: str = "Automatic"
    clip_skip: int = 2
    # ✨ Tự fix: mở sẵn Hires. fix + ADetailer khi vào UI
    auto_hires_fix: bool = True
    hires_upscale_by: float = 1.5
    hires_denoise: float = 0.4
    auto_adetailer: str = "face+hand"          # face+hand | face | off
    # Chế độ test (CPU, không GPU) – chỉ dùng khi kiểm thử bộ cài
    test_mode: bool = False

    @staticmethod
    def load(path: Path = CONFIG_PATH) -> "Config":
        cfg = Config()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Đường dẫn
# --------------------------------------------------------------------------------------
class Paths:
    root = ROOT
    forge = ROOT / "noobai-forge"
    venv = ROOT / "venv-noobai"
    uv_bin = ROOT / "bin" / "uv"
    drive = Path("/content/drive/MyDrive")

    @classmethod
    def python(cls) -> Path:
        return cls.venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    @classmethod
    def models(cls, kind: str) -> Path:
        return cls.forge / "models" / kind

    @classmethod
    def extensions(cls) -> Path:
        return cls.forge / "extensions"


# --------------------------------------------------------------------------------------
# Tiện ích
# --------------------------------------------------------------------------------------
def banner(msg: str) -> None:
    line = "─" * max(8, min(78, len(msg) + 6))
    print(f"\n┌{line}┐\n│   {msg}\n└{line}┘", flush=True)


def log(msg: str) -> None:
    print(f"  • {msg}", flush=True)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def run(cmd: list[str] | str, *, cwd: Path | None = None, env: dict | None = None,
        check: bool = True, quiet: bool = False) -> int:
    """Chạy lệnh, stream output trực tiếp ra cell."""
    shell = isinstance(cmd, str)
    if not quiet:
        log("$ " + (cmd if shell else " ".join(str(c) for c in cmd)))
    full_env = {**os.environ, **(env or {})}
    full_env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=full_env, shell=shell,
                          stdout=subprocess.DEVNULL if quiet else None,
                          stderr=subprocess.STDOUT if quiet else None)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Lệnh thất bại (mã {proc.returncode}): {cmd}")
    return proc.returncode


def capture(cmd: list[str], **kw) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, **kw).stdout.strip()


def is_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ or "google.colab" in sys.modules


def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True).returncode == 0


def gpu_name() -> str:
    if not has_nvidia_gpu():
        return "không có GPU"
    return capture(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]) or "GPU không rõ"


def gpu_vram_mb() -> int:
    """Tổng VRAM (MB) của GPU đầu tiên; 0 nếu không có GPU."""
    if not has_nvidia_gpu():
        return 0
    out = capture(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]) or "0"
    try:
        return int(float(out.splitlines()[0].strip()))
    except Exception:  # noqa: BLE001
        return 0


def gpu_compute_capability() -> float:
    """Compute capability (7.5 = T4/Turing, 8.x = A100/L4/Ampere…); 0 nếu không rõ."""
    if not has_nvidia_gpu():
        return 0.0
    out = capture(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]) or ""
    try:
        return float(out.splitlines()[0].strip())
    except Exception:  # noqa: BLE001
        return 0.0


def gpu_plan(cfg: "Config") -> dict:
    """Quyết định cờ Forge + cài đặt bộ nhớ theo GPU_MODE và GPU thực tế.

    max      : giữ toàn bộ model trên GPU, không hoán đổi ra RAM (nhanh nhất).
               VRAM ≥ 20 GB (L4/A100): --always-gpu ; 12–20 GB (T4 15 GB): --always-high-vram ; < 12 GB: như balanced.
    balanced : mặc định Forge (tự hoán đổi theo VRAM trống).
    lowvram  : ưu tiên ảnh lớn / batch lớn: --always-normal-vram --always-offload-from-vram, chừa 4 GB cho inference.
    """
    mode = str(cfg.gpu_mode or "max").lower()
    vram = gpu_vram_mb()
    cc = gpu_compute_capability()
    flags: list[str] = []
    settings: dict = {"forge_inference_memory": 1024, "forge_async_loading": "Queue", "forge_pin_shared_memory": "CPU"}
    why = []
    if vram <= 0 or cfg.test_mode:
        return {"mode": mode, "flags": flags, "settings": settings, "vram": vram, "cc": cc, "why": ["không có GPU"]}
    if mode.startswith("max"):
        if vram >= 20000:
            flags.append("--always-gpu")
            why.append(f"VRAM {vram/1024:.0f} GB ≥ 20 GB → toàn bộ model + VAE + text encoder nằm hẳn trên GPU")
        elif vram >= 12000:
            flags.append("--always-high-vram")
            why.append(f"VRAM {vram/1024:.0f} GB → UNet nằm hẳn trên GPU (high-vram), không hoán đổi ra RAM")
        else:
            why.append(f"VRAM {vram/1024:.0f} GB < 12 GB → giữ chế độ tự hoán đổi của Forge")
        flags.append("--cuda-malloc")
        why.append("--cuda-malloc: cấp phát bộ nhớ CUDA nhanh hơn")
        if cfg.download_vae:
            flags.append("--vae-in-fp16")
            why.append("--vae-in-fp16: giải mã ảnh bằng fp16 (an toàn vì dùng sdxl-vae-fp16-fix) → nhanh gấp ~2 so với fp32"
                       + (" trên T4/Turing" if cc and cc < 8 else ""))
        settings["forge_inference_memory"] = 1024
        settings["show_progress_every_n_steps"] = 8
    elif mode.startswith("low"):
        flags += ["--always-normal-vram", "--always-offload-from-vram"]
        settings["forge_inference_memory"] = 4096
        why.append("chừa 4 GB VRAM cho inference, giải phóng model sau mỗi bước → ảnh/batch lớn hơn, chậm hơn")
    else:
        why.append("mặc định Forge")
    return {"mode": mode, "flags": flags, "settings": settings, "vram": vram, "cc": cc, "why": why}


def sha256sum(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def split_urls(text: str) -> list[str]:
    return [u.strip() for u in re.split(r"[,\n\s]+", text or "") if u.strip()]


# --------------------------------------------------------------------------------------
# Tải file (aria2c nếu có, fallback Python thuần có resume + progress)
# --------------------------------------------------------------------------------------
def _auth_headers(url: str, cfg: Config) -> list[str]:
    headers: list[str] = []
    if "api.github.com" in url:
        headers.append("Accept: application/vnd.github.raw")
    if "civitai.com" in url and cfg.civitai_token:
        headers.append(f"Authorization: Bearer {cfg.civitai_token}")
    if "huggingface.co" in url and cfg.hf_token:
        headers.append(f"Authorization: Bearer {cfg.hf_token}")
    return headers


def _py_download(url: str, dest: Path, headers: list[str]) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url)
    for h in headers:
        k, v = h.split(":", 1)
        req.add_header(k.strip(), v.strip())
    req.add_header("User-Agent", "noobai-colab-installer/" + LIB_VERSION)
    if existing:
        req.add_header("Range", f"bytes={existing}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        if existing and resp.status != 206:
            existing = 0  # server không hỗ trợ resume → tải lại từ đầu
        total = resp.headers.get("Content-Length")
        total = int(total) + existing if total else None
        mode = "ab" if existing else "wb"
        done, last, t0, shown = existing, time.time(), time.time(), False
        with tmp.open(mode) as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 2:
                    last, shown = time.time(), True
                    speed = (done - existing) / max(1e-6, last - t0)
                    pct = f"{done / total * 100:5.1f}%" if total else ""
                    print(f"\r    {dest.name}: {human(done)} {pct}  {human(speed)}/s   ", end="", flush=True)
        if shown:
            print()
    tmp.rename(dest)


def download_file(url: str, dest: Path, cfg: Config, *, size: int | None = None,
                  sha256: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if size is None or dest.stat().st_size == size:
            log(f"✔ đã có: {dest.name} ({human(dest.stat().st_size)})")
            return dest
        log(f"⚠ {dest.name} sai kích thước ({dest.stat().st_size} ≠ {size}) → tải lại")
        dest.unlink()

    headers = _auth_headers(url, cfg)
    log(f"⬇ tải {dest.name}  ({human(size) if size else '?'})")
    aria = shutil.which("aria2c")
    ok = False
    if aria:
        cmd = [aria, "--console-log-level=error", "--summary-interval=10", "-c", "-x", "16", "-s", "16",
               "-k", "1M", "--file-allocation=none", "--auto-file-renaming=false", "--allow-overwrite=true",
               "-d", str(dest.parent), "-o", dest.name, url]
        for h in headers:
            cmd.insert(1, f"--header={h}")
        ok = run(cmd, check=False, quiet=False) == 0 and dest.exists()
        if not ok:
            log("aria2c thất bại → dùng trình tải Python")
    if not ok:
        for attempt in range(1, 4):
            try:
                _py_download(url, dest, headers)
                ok = True
                break
            except Exception as e:  # noqa: BLE001
                log(f"lỗi tải (lần {attempt}/3): {e}")
                time.sleep(3)
    if not ok or not dest.exists():
        raise RuntimeError(f"Không tải được {url}")

    actual = dest.stat().st_size
    if size is not None and actual != size:
        raise RuntimeError(f"{dest.name}: kích thước {actual} ≠ {size} (file hỏng / link đổi / cần token)")
    if sha256 and cfg.verify_sha256:
        log(f"kiểm tra SHA256 {dest.name} …")
        got = sha256sum(dest)
        if got != sha256:
            raise RuntimeError(f"{dest.name}: SHA256 không khớp ({got})")
        log("✔ SHA256 khớp")
    log(f"✔ xong: {dest.name} ({human(actual)})")
    return dest


def normalize_url(url: str) -> str:
    """Chuyển link trang Civitai (…/models/123?modelVersionId=456) thành link tải API."""
    m = re.search(r"civitai\.com/models/\d+.*?modelVersionId=(\d+)", url)
    if m:
        return f"https://civitai.com/api/download/models/{m.group(1)}"
    return url


def _guess_filename(url: str) -> str:
    """Đoán tên file từ URL; với link Civitai thì hỏi server (Content-Disposition)."""
    name = url.split("?")[0].rstrip("/").split("/")[-1]
    if "civitai.com" in url or "." not in name:
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as r:
                cd = r.headers.get("Content-Disposition", "")
                m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', cd)
                if m:
                    name = urllib.parse.unquote(m.group(1))
        except Exception:  # noqa: BLE001
            pass
    if not name:
        name = f"download_{int(time.time())}"
    if "." not in name:
        name += ".safetensors"
    return name


# --------------------------------------------------------------------------------------
# Bước 1: môi trường Python 3.10 + torch + Forge
# --------------------------------------------------------------------------------------
def ensure_system_tools() -> None:
    if shutil.which("aria2c") is None and shutil.which("apt-get"):
        log("cài aria2 (tải song song nhanh hơn)")
        sudo = [] if os.geteuid() == 0 else (["sudo", "-n"] if shutil.which("sudo") else [])
        run(sudo + ["apt-get", "install", "-y", "-qq", "aria2"], check=False, quiet=True)
    if shutil.which("aria2c") is None:
        log("⚠ không có aria2c → sẽ dùng trình tải Python (chậm hơn một chút)")
    if shutil.which("git") is None:
        raise RuntimeError("Thiếu git")


def ensure_uv() -> Path:
    if Paths.uv_bin.exists():
        return Paths.uv_bin
    found = shutil.which("uv")
    if found:
        return Path(found)
    log("cài uv (trình quản lý Python/pip siêu nhanh)")
    Paths.uv_bin.parent.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "--target", str(Paths.uv_bin.parent / "_uvpkg"), "uv"])
    cand = list((Paths.uv_bin.parent / "_uvpkg").rglob("uv"))
    cand = [c for c in cand if c.is_file() and os.access(c, os.X_OK)]
    if not cand:
        raise RuntimeError("Không tìm thấy binary uv sau khi cài")
    shutil.copy2(cand[0], Paths.uv_bin)
    Paths.uv_bin.chmod(0o755)
    return Paths.uv_bin


def uv_env() -> dict:
    return {
        "UV_PYTHON_INSTALL_DIR": str(ROOT / "uv-python"),
        "UV_CACHE_DIR": str(ROOT / "uv-cache"),
        "UV_SYSTEM_CERTS": "1",  # dùng CA của hệ thống (cần khi mạng đi qua proxy TLS)
        "UV_HTTP_TIMEOUT": "600",
    }


def find_system_python() -> Path | None:
    """Tìm Python 3.10 (ưu tiên) hoặc 3.11 có sẵn trên máy – Forge hỗ trợ 3.7–3.11."""
    candidates = ["python3.10", "python3.11"]
    for name in candidates:
        exe = shutil.which(name)
        if exe:
            return Path(exe)
    if sys.version_info[:2] in ((3, 10), (3, 11)):
        return Path(sys.executable)
    return None


def ensure_python_env(cfg: Config) -> Path:
    py = Paths.python()
    uv = str(ensure_uv())
    env = uv_env()
    if not py.exists():
        log(f"tạo môi trường Python {PYTHON_VERSION} tại {Paths.venv}")
        base = find_system_python()
        if base is None:
            # Colab hiện là Python 3.12/3.13 → tải Python 3.10 độc lập bằng uv
            if run([uv, "python", "install", PYTHON_VERSION], env=env, check=False) == 0:
                base = PYTHON_VERSION
            elif shutil.which("apt-get"):
                log("uv không tải được Python → thử cài python3.10 bằng apt")
                sudo = [] if os.geteuid() == 0 else (["sudo", "-n"] if shutil.which("sudo") else [])
                run(sudo + ["apt-get", "update", "-qq"], check=False, quiet=True)
                run(sudo + ["apt-get", "install", "-y", "-qq", "python3.10", "python3.10-venv", "python3.10-distutils"],
                    check=False, quiet=True)
                base = find_system_python()
        if base is None:
            raise RuntimeError("Không chuẩn bị được Python 3.10/3.11 cho Forge")
        log(f"dùng Python: {base}")
        run([uv, "venv", "--python", str(base), "--seed", str(Paths.venv)], env=env)
    ver = capture([str(py), "-c", "import sys;print('%d.%d.%d'%sys.version_info[:3])"])
    log(f"Python trong venv: {ver}")

    def uv_pip(*args: str) -> None:
        run([uv, "pip", "install", "--python", str(py), *args], env=env)

    have_torch = subprocess.run([str(py), "-c", "import torch, torchvision"], capture_output=True).returncode == 0
    if not have_torch:
        if cfg.test_mode or not has_nvidia_gpu():
            log("không có GPU NVIDIA → cài torch bản mặc định PyPI (CPU) – chỉ để kiểm thử")
            uv_pip(*TORCH_SPEC_GPU.split())
        else:
            log(f"cài {TORCH_SPEC_GPU} (CUDA 12.1)")
            uv_pip(*TORCH_SPEC_GPU.split(), "--index-url", TORCH_INDEX_GPU)
    if cfg.use_xformers and has_nvidia_gpu() and not cfg.test_mode:
        uv_pip(XFORMERS_SPEC, "--no-deps", "--extra-index-url", TORCH_INDEX_GPU)
    return py


def install_forge(cfg: Config) -> None:
    if not (Paths.forge / "launch.py").exists():
        log(f"clone Forge → {Paths.forge}")
        run(["git", "clone", "--filter=blob:none", FORGE_REPO, str(Paths.forge)])
    commit = (cfg.forge_commit or "").strip()
    if commit and commit.lower() != "latest":
        cur = capture(["git", "-C", str(Paths.forge), "rev-parse", "HEAD"])
        if cur != commit:
            log(f"checkout Forge commit {commit[:10]} (đã kiểm thử)")
            run(["git", "-C", str(Paths.forge), "fetch", "--depth", "1", "origin", commit], check=False, quiet=True)
            run(["git", "-C", str(Paths.forge), "checkout", "-q", commit])
    else:
        log("cập nhật Forge lên bản mới nhất")
        run(["git", "-C", str(Paths.forge), "pull", "--ff-only"], check=False)

    py = Paths.python()
    uv = str(ensure_uv())
    env = uv_env()
    log("cài requirements của Forge bằng uv (nhanh hơn pip nhiều lần)")
    run([uv, "pip", "install", "--python", str(py), "-r", str(Paths.forge / "requirements_versions.txt")], env=env)
    # CLIP dùng pkg_resources trong setup.py → build bằng setuptools 69.5.1 đã có trong venv (không isolation)
    run([uv, "pip", "install", "--python", str(py), "wheel", "joblib"], env=env)  # joblib: script soft-inpainting của Forge
    run([uv, "pip", "install", "--python", str(py), "--no-build-isolation", CLIP_PKG], env=env)
    if subprocess.run([str(py), "-c", "import open_clip"], capture_output=True).returncode != 0:
        run([uv, "pip", "install", "--python", str(py), "--no-build-isolation", OPENCLIP_PKG], env=env)
    if cfg.tunnel in ("ngrok", "both"):
        run([uv, "pip", "install", "--python", str(py), "ngrok"], env=env)
    for d in ("Stable-diffusion", "VAE", "Lora", "ControlNet", "ESRGAN", "embeddings"):
        (Paths.forge / "models" / d).mkdir(parents=True, exist_ok=True)
    (Paths.forge / "embeddings").mkdir(exist_ok=True)
    (Paths.forge / "outputs").mkdir(exist_ok=True)


def write_constraints() -> Path:
    """Tạo file constraints từ requirements_versions.txt của Forge.

    Đặt PIP_CONSTRAINT khi chạy install.py của các extension để chúng KHÔNG nâng cấp
    numpy / httpx / pillow / torch … lên bản phá vỡ Forge (ví dụ ADetailer → ultralytics → numpy 2.x).
    """
    req = Paths.forge / "requirements_versions.txt"
    pins = [ln.split("#")[0].strip() for ln in req.read_text().splitlines()]
    pins = [p for p in pins if "==" in p and not p.lower().startswith("torch")]
    pins += TORCH_SPEC_GPU.split()
    out = ROOT / "noobai_constraints.txt"
    out.write_text("\n".join(pins) + "\n", encoding="utf-8")
    return out


def forge_env() -> dict:
    """Biến môi trường an toàn khi chạy Forge từ trong notebook Colab/Jupyter."""
    return {
        # Colab đặt MPLBACKEND=module://matplotlib_inline.backend_inline → venv của Forge không có module này
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "GRADIO_ANALYTICS_ENABLED": "False",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONWARNINGS": "ignore::DeprecationWarning",
    }


def pip_env() -> dict:
    return {
        "PIP_CONSTRAINT": str(write_constraints()),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "IIB_SKIP_OPTIONAL_DEPS": "1",  # image browser: bỏ qua hnswlib (tùy chọn, cần compiler)
    }


def prepare_forge_environment(cfg: Config) -> None:
    """Chạy launch.py --exit: clone repo phụ + chạy install.py của extension (có ràng buộc phiên bản)."""
    log("chuẩn bị môi trường Forge (repos phụ + installer của extension)…")
    args = ["--exit", "--skip-torch-cuda-test", "--skip-python-version-check"]
    if cfg.use_xformers and has_nvidia_gpu() and not cfg.test_mode:
        args.append("--xformers")
    env = {**pip_env(), **forge_env(), "COMMANDLINE_ARGS": " ".join(args)}
    rc = run([str(Paths.python()), "launch.py"], cwd=Paths.forge, env=env, check=False)
    if rc != 0:
        log(f"⚠ launch.py --exit trả về mã {rc} – xem log phía trên")
    # Ghim lại các gói cốt lõi phòng khi installer nào đó vẫn đổi phiên bản
    uv = str(ensure_uv())
    run([uv, "pip", "install", "--python", str(Paths.python()), "-r", str(Paths.forge / "requirements_versions.txt")],
        env=uv_env(), quiet=True)
    fix_opencv_if_no_libgl()


def fix_opencv_if_no_libgl() -> None:
    """Máy không có libGL (hiếm – Colab có sẵn) → chuyển opencv sang bản headless để import cv2 không lỗi."""
    import ctypes.util
    if ctypes.util.find_library("GL"):
        return
    py = str(Paths.python())
    if subprocess.run([py, "-c", "import cv2"], capture_output=True).returncode == 0:
        return
    log("không có libGL → chuyển opencv sang bản headless")
    uv = str(ensure_uv())
    installed = capture([uv, "pip", "freeze", "--python", py], env={**os.environ, **uv_env()})
    pkgs = [ln.split("==")[0] for ln in installed.splitlines() if ln.lower().startswith("opencv")]
    if pkgs:
        run([uv, "pip", "uninstall", "--python", py, *pkgs], env=uv_env(), quiet=True)
    want = ["opencv-python-headless"] + (["opencv-contrib-python-headless"] if any("contrib" in p for p in pkgs) else [])
    run([uv, "pip", "install", "--python", py, "--constraint", str(write_constraints()), *want], env=uv_env())


def install_extensions(cfg: Config) -> None:
    ext_dir = Paths.extensions()
    ext_dir.mkdir(parents=True, exist_ok=True)
    for key in cfg.extensions:
        if key not in EXTENSIONS:
            log(f"⚠ bỏ qua extension lạ: {key}")
            continue
        folder, url = EXTENSIONS[key]
        target = ext_dir / folder
        if (target / ".git").exists():
            log(f"✔ extension đã có: {folder}")
            continue
        log(f"cài extension {folder}")
        run(["git", "clone", "--depth", "1", "-q", url, str(target)])


def install_tag_csv(cfg: Config) -> Path | None:
    """Đưa file tag Danbooru+e621 (từ repo anime2) vào Tag Autocomplete."""
    if "tagcomplete" not in cfg.extensions:
        return None
    tags_dir = Paths.extensions() / EXTENSIONS["tagcomplete"][0] / "tags"
    tags_dir.mkdir(parents=True, exist_ok=True)
    dest = tags_dir / TAG_CSV_NAME
    if not cfg.use_repo_tag_csv:
        return None
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log(f"✔ file tag đã có: {dest.name}")
        return dest
    local = Path(__file__).resolve().parent.parent / TAG_CSV_NAME  # khi chạy từ repo
    if local.exists():
        shutil.copy2(local, dest)
        log(f"✔ copy file tag từ repo local ({human(dest.stat().st_size)})")
        return dest
    for url in TAG_CSV_URLS:
        try:
            download_file(url, dest, cfg)
            if dest.stat().st_size > 1_000_000:
                return dest
            dest.unlink()
        except Exception as e:  # noqa: BLE001
            log(f"không tải được từ {url}: {e}")
    log("⚠ không lấy được file tag của repo → dùng danbooru_e621_merged.csv có sẵn của extension")
    return None


def install_localization(cfg: Config) -> Path | None:
    """Cài bản dịch tiếng Việt vào <forge>/localizations/vi_VN.json (Forge tự nạp qua cài đặt `localization`)."""
    dest_dir = Paths.forge / "localizations"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{VI_LOCALE_NAME}.json"
    here = Path(__file__).resolve().parent
    candidates = [here / f"{VI_LOCALE_NAME}.json",                 # Colab: /content/vi_VN.json
                  here.parent / "colab" / f"{VI_LOCALE_NAME}.json",  # chạy từ repo
                  Path("/content") / f"{VI_LOCALE_NAME}.json"]
    for local in candidates:
        if local.exists() and local.stat().st_size > 1000:
            try:
                json.loads(local.read_text(encoding="utf-8"))
                shutil.copy2(local, dest)
                log(f"✔ bản dịch tiếng Việt: {dest} ({human(dest.stat().st_size)})")
                return dest
            except Exception as e:  # noqa: BLE001
                log(f"⚠ file dịch {local} lỗi: {e}")
    if VI_LOCALE_EMBEDDED:
        dest.write_text(json.dumps(VI_LOCALE_EMBEDDED, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"✔ bản dịch tiếng Việt (nhúng sẵn, {len(VI_LOCALE_EMBEDDED)} chuỗi) → {dest}")
        return dest
    if dest.exists() and dest.stat().st_size > 1000:
        log(f"✔ bản dịch tiếng Việt đã có: {dest.name}")
        return dest
    for url in VI_LOCALE_URLS:
        try:
            download_file(url, dest, cfg)
            json.loads(dest.read_text(encoding="utf-8"))
            log(f"✔ tải bản dịch tiếng Việt từ {url}")
            return dest
        except Exception as e:  # noqa: BLE001
            log(f"không tải được từ {url}: {e}")
            if dest.exists():
                dest.unlink()
    log("⚠ không có bản dịch tiếng Việt → giao diện sẽ hiển thị tiếng Anh")
    return None


def install_noob_tools(cfg: Config) -> Path | None:
    """Chép script noob_tools.py (dropdown độ phân giải chuẩn + tự bật Hires fix/ADetailer) vào <forge>/scripts/."""
    dest = Paths.forge / "scripts" / NOOB_TOOLS_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    for local in (here / NOOB_TOOLS_NAME, here.parent / "colab" / NOOB_TOOLS_NAME, Path("/content") / NOOB_TOOLS_NAME):
        if local.exists() and local.stat().st_size > 500:
            shutil.copy2(local, dest)
            log(f"✔ NoobAI Tools script: {dest}")
            return dest
    if NOOB_TOOLS_EMBEDDED:
        dest.write_text(NOOB_TOOLS_EMBEDDED, encoding="utf-8")
        log(f"✔ NoobAI Tools script (nhúng sẵn) → {dest}")
        return dest
    if dest.exists():
        log(f"✔ NoobAI Tools script đã có: {dest.name}")
        return dest
    for url in NOOB_TOOLS_URLS:
        try:
            download_file(url, dest, cfg)
            compile(dest.read_text(encoding="utf-8"), str(dest), "exec")
            log(f"✔ tải NoobAI Tools script từ {url}")
            return dest
        except Exception as e:  # noqa: BLE001
            log(f"không tải được từ {url}: {e}")
            if dest.exists():
                dest.unlink()
    log("⚠ không có noob_tools.py → không có dropdown độ phân giải chuẩn / tự fix")
    return None


def setup_drive(cfg: Config) -> dict:
    """Mount Google Drive (chỉ trong Colab) và liên kết thư mục outputs / models."""
    info = {"mounted": False, "outputs": str(Paths.forge / "outputs")}
    if not cfg.mount_drive:
        return info
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        log("không phải Colab → bỏ qua Google Drive")
        return info
    if not Paths.drive.exists():
        drive.mount("/content/drive")
    if not Paths.drive.exists():
        log("⚠ mount Drive thất bại")
        return info
    info["mounted"] = True
    base = Paths.drive / cfg.drive_folder
    if cfg.save_outputs_to_drive:
        out = base / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        info["outputs"] = str(out)
        log(f"✔ ảnh sẽ lưu vào Drive: {out}")
    if cfg.link_drive_models:
        for sub, kind in (("models", "Stable-diffusion"), ("Lora", "Lora"), ("embeddings", "embeddings"),
                          ("VAE", "VAE"), ("ControlNet", "ControlNet")):
            src = base / sub
            src.mkdir(parents=True, exist_ok=True)
            link = (Paths.forge / "embeddings" / "drive") if kind == "embeddings" else (Paths.models(kind) / "drive")
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                continue
            link.symlink_to(src, target_is_directory=True)
            log(f"✔ liên kết Drive/{cfg.drive_folder}/{sub} → {link.relative_to(Paths.forge)}")
    return info


# --------------------------------------------------------------------------------------
# Bước 2: tải tài nguyên
# --------------------------------------------------------------------------------------
def planned_downloads(cfg: Config) -> list[dict]:
    items: list[dict] = []
    if cfg.download_model:
        items += MODEL_FILES.values()
    if cfg.download_vae:
        items += VAE_FILES.values()
    if cfg.download_upscalers:
        items += UPSCALER_FILES.values()
    for key in CONTROLNET_SETS.get(cfg.controlnet, []):
        items.append(CONTROLNET_FILES[key])
    if "adetailer" in cfg.extensions:
        items += ADETAILER_FILES.values()
    for url in map(normalize_url, split_urls(cfg.extra_model_urls)):
        items.append(dict(kind="Stable-diffusion", name=_guess_filename(url), url=url, size=None, sha256=None))
    for url in map(normalize_url, split_urls(cfg.extra_lora_urls)):
        items.append(dict(kind="Lora", name=_guess_filename(url), url=url, size=None, sha256=None))
    for url in map(normalize_url, split_urls(cfg.extra_embedding_urls)):
        items.append(dict(kind="embeddings", name=_guess_filename(url), url=url, size=None, sha256=None))
    return items


def dest_for(item: dict) -> Path:
    if item["kind"] == "embeddings":
        return Paths.forge / "embeddings" / item["name"]
    return Paths.models(item["kind"]) / item["name"]


def download_resources(cfg: Config) -> None:
    items = planned_downloads(cfg)
    total = sum(i["size"] or 0 for i in items)
    banner(f"Tải tài nguyên: {len(items)} file (~{human(total)})")
    free = shutil.disk_usage(str(ROOT)).free
    log(f"dung lượng trống: {human(free)}")
    if total and free < total * 1.05:
        raise RuntimeError("Không đủ dung lượng đĩa cho các file đã chọn – bớt ControlNet hoặc dùng runtime khác")
    failed: list[tuple[dict, str]] = []
    for item in items:
        try:
            download_file(item["url"], dest_for(item), cfg, size=item["size"], sha256=item["sha256"])
        except Exception as e:  # noqa: BLE001
            log(f"❌ {item['name']}: {e}")
            failed.append((item, str(e)))
    if failed:
        print()
        log(f"⚠ {len(failed)}/{len(items)} file tải thất bại (chạy lại cell này để thử lại – tải tiếp từ chỗ dở):")
        for item, err in failed:
            log(f"   - {item['kind']}/{item['name']}")
        if any(f[0]["kind"] == "Stable-diffusion" and f[0]["name"] == MODEL_FILES["noobai_xl_eps_1_1"]["name"]
               for f in failed):
            raise RuntimeError("Model chính NoobAI-XL-v1.1 chưa tải được – kiểm tra mạng/dung lượng rồi chạy lại.")
    else:
        log("✔ đã tải đủ tất cả tài nguyên")


# --------------------------------------------------------------------------------------
# Bước 3: cấu hình mặc định cho NoobAI (config.json / ui-config.json / styles.csv)
# --------------------------------------------------------------------------------------
def _merge_json(path: Path, updates: dict) -> None:
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
    data.update(updates)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


RECOMMENDED_PARAMS = dict(default_width=832, default_height=1216, default_steps=28, default_cfg=6.0,
                          default_sampler="Euler a", default_scheduler="Automatic", clip_skip=2,
                          auto_hires_fix=True, hires_upscale_by=1.5, hires_denoise=0.4, auto_adetailer="face+hand")


def write_settings(cfg: Config, drive_info: dict | None = None) -> None:
    banner("Ghi cấu hình mặc định cho NoobAI-XL")
    if str(cfg.param_mode).lower().startswith("rec"):
        for k, v in RECOMMENDED_PARAMS.items():
            setattr(cfg, k, v)
        log("thông số: bộ KHUYÊN DÙNG NoobAI (Euler a · 28 steps · CFG 6 · 832×1216 · Clip skip 2 · Hires ×1.5/0.4 · ADetailer mặt+tay)")
    else:
        log(f"thông số: TỰ CHỈNH ({cfg.default_sampler} · {cfg.default_steps} steps · CFG {cfg.default_cfg} · "
            f"{cfg.default_width}×{cfg.default_height} · Hires {'on' if cfg.auto_hires_fix else 'off'} · ADetailer {cfg.auto_adetailer})")
    drive_info = drive_info or {}
    outputs = Path(drive_info.get("outputs") or (Paths.forge / "outputs"))
    vae_path = Paths.models("VAE") / VAE_FILES["sdxl_vae_fp16_fix"]["name"]
    model_name = MODEL_FILES["noobai_xl_eps_1_1"]["name"]

    settings = {
        "sd_model_checkpoint": model_name,
        "forge_preset": "xl",
        "forge_additional_modules": [str(vae_path)] if cfg.download_vae else [],
        "forge_unet_storage_dtype": "Automatic",
        "CLIP_stop_at_last_layers": cfg.clip_skip,
        "xl_t2i_width": cfg.default_width,
        "xl_t2i_height": cfg.default_height,
        "xl_t2i_cfg": cfg.default_cfg,
        "xl_t2i_hr_cfg": cfg.default_cfg,
        "xl_t2i_sampler": cfg.default_sampler,
        "xl_t2i_scheduler": cfg.default_scheduler,
        "xl_i2i_width": 1024,
        "xl_i2i_height": 1024,
        "xl_i2i_cfg": cfg.default_cfg,
        "xl_i2i_sampler": cfg.default_sampler,
        "xl_i2i_scheduler": cfg.default_scheduler,
        "samples_format": "png",
        "save_images_add_number": True,
        "samples_filename_pattern": "[seed]-[prompt_spaces]",
        "outdir_txt2img_samples": str(outputs / "txt2img-images"),
        "outdir_img2img_samples": str(outputs / "img2img-images"),
        "outdir_extras_samples": str(outputs / "extras-images"),
        "outdir_txt2img_grids": str(outputs / "txt2img-grids"),
        "outdir_img2img_grids": str(outputs / "img2img-grids"),
        "outdir_save": str(outputs / "saved"),
        "live_previews_enable": True,
        "show_progress_every_n_steps": 5,
        "show_progressbar": True,
        "quicksettings_list": ["sd_model_checkpoint", "sd_vae", "CLIP_stop_at_last_layers"],
        "upscaler_for_img2img": "4x-AnimeSharp" if cfg.download_upscalers else "None",
        "hires_fix_show_sampler": True,
        "hires_fix_show_prompts": True,
        "token_merging_ratio": 0.0,
        "emphasis": "Original",
        "always_discard_next_to_last_sigma": False,
        "sdxl_crop_top": 0,
        "sdxl_crop_left": 0,
        "sdxl_refiner_low_aesthetic_score": 2.5,
        "sdxl_refiner_high_aesthetic_score": 6.0,
        "disable_all_extensions": "none",
        # Bộ nhớ GPU (theo GPU_MODE) – xem gpu_plan()
        **gpu_plan(cfg)["settings"],
        # Ngôn ngữ giao diện (Settings ▸ User interface ▸ Localization)
        "localization": VI_LOCALE_NAME if (str(cfg.ui_language).lower().startswith("vi") and
                                           (Paths.forge / "localizations" / f"{VI_LOCALE_NAME}.json").exists())
                        else "None",
        # Tag Autocomplete
        "tac_tagFile": TAG_CSV_NAME if (cfg.use_repo_tag_csv and
                                         (Paths.extensions() / EXTENSIONS["tagcomplete"][0] / "tags" / TAG_CSV_NAME).exists())
                       else "danbooru_e621_merged.csv",
        "tac_active": True,
        "tac_maxResults": 8,
        "tac_showAllResults": False,
        "tac_replaceUnderscores": False,
        "tac_escapeParentheses": True,
        "tac_appendComma": True,
        "tac_appendSpace": True,
        "tac_alias.searchByAlias": True,
        "tac_useWildcards": True,
        "tac_useEmbeddings": True,
        "tac_useLoras": True,
        "tac_useLycos": True,
        "tac_extra.extraFile": "extra-quality-tags.csv",
        "tac_chantFile": "noob_characters-chants.json",
        # ADetailer mặc định
        "ad_max_models": 2,
        "ad_save_previews": False,
    }
    _merge_json(Paths.forge / "config.json", settings)
    log(f"✔ config.json (ngôn ngữ giao diện: {settings['localization']})")

    ui = {
        "txt2img/Prompt/value": NOOB_POSITIVE_PREFIX,
        "txt2img/Negative prompt/value": NOOB_NEGATIVE,
        "img2img/Prompt/value": NOOB_POSITIVE_PREFIX,
        "img2img/Negative prompt/value": NOOB_NEGATIVE,
        "customscript/sampler.py/txt2img/Sampling steps/value": cfg.default_steps,
        "customscript/sampler.py/img2img/Sampling steps/value": cfg.default_steps,
        "txt2img/Upscaler/value": "4x-AnimeSharp" if cfg.download_upscalers else "Latent",
        "txt2img/Upscale by/value": float(cfg.hires_upscale_by),
        "txt2img/Hires steps/value": 14,
        "txt2img/Denoising strength/value": float(cfg.hires_denoise),
        "txt2img/Batch count/value": 1,
        "txt2img/Batch size/value": 1,
        "img2img/Denoising strength/value": 0.5,
    }
    # ✨ Tự fix: ADetailer – bộ 1 sửa mặt, bộ 2 sửa tay (model YOLO đã nằm trong gói tải mặc định)
    ad = str(cfg.auto_adetailer or "off").lower()
    if ad != "off" and "adetailer" in cfg.extensions:
        for tab in ("txt2img", "img2img"):
            ui[f"{tab}/ADetailer detector/value"] = "face_yolov8n.pt"
            ui[f"{tab}/Detection model confidence threshold/value"] = 0.3
            ui[f"{tab}/Inpaint denoising strength/value"] = 0.4
            ui[f"{tab}/Enable this tab (2nd)/value"] = True
            ui[f"{tab}/ADetailer detector 2nd/value"] = "hand_yolov8n.pt" if ad == "face+hand" else "None"
            ui[f"{tab}/Detection model confidence threshold 2nd/value"] = 0.3
            ui[f"{tab}/Inpaint denoising strength 2nd/value"] = 0.4
    _merge_json(Paths.forge / "ui-config.json", ui)
    log("✔ ui-config.json (prompt / steps / hires / ADetailer mặc định)")

    tools = {
        "auto_hires_fix": bool(cfg.auto_hires_fix),
        "auto_adetailer": ad if "adetailer" in cfg.extensions else "off",
        "param_mode": "recommended" if str(cfg.param_mode).lower().startswith("rec") else "custom",
    }
    (Paths.forge / "noob_tools.json").write_text(json.dumps(tools, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"✔ noob_tools.json (tự fix: hires={tools['auto_hires_fix']}, adetailer={tools['auto_adetailer']})")

    styles = [
        ("Noob ✦ Quality (chuẩn)", NOOB_POSITIVE_PREFIX + "{prompt}", NOOB_NEGATIVE),
        ("Noob ✦ Quality + Safe", NOOB_POSITIVE_PREFIX + "general, {prompt}", NOOB_NEGATIVE + ", nsfw, explicit, sensitive"),
        ("Noob ✦ Năm: newest", "{prompt}, newest", "old, early"),
        ("Noob ✦ Năm: recent", "{prompt}, recent", "old, early"),
        ("Noob ✦ Năm: mid", "{prompt}, mid", ""),
        ("Noob ✦ Năm: early", "{prompt}, early", "newest"),
        ("Noob ✦ Năm: old", "{prompt}, old", "newest"),
        ("Noob ✦ Phong cách: sketch", "{prompt}, sketch, lineart, monochrome, greyscale", "colored"),
        ("Noob ✦ Phong cách: flat color", "{prompt}, flat color, limited palette, clean lines", "gradient, photorealistic"),
        ("Noob ✦ Phong cách: watercolor", "{prompt}, watercolor \\(medium\\), traditional media, soft colors", ""),
        ("Noob ✦ Phong cách: cinematic", "{prompt}, cinematic lighting, depth of field, volumetric lighting, dramatic shadows", "flat lighting"),
        ("Noob ✦ Phong cách: 90s anime", "{prompt}, 1990s \\(style\\), retro artstyle, film grain", ""),
        ("Noob ✦ Negative: tay/anatomy", "{prompt}", NOOB_NEGATIVE + ", extra fingers, fewer fingers, extra digits, missing fingers, malformed limbs, extra arms, extra legs, deformed, disfigured"),
        ("Noob ✦ Negative: chữ/watermark", "{prompt}", "text, watermark, signature, artist name, logo, patreon username, web address, speech bubble"),
    ]
    styles_path = Paths.forge / "styles.csv"
    existing = styles_path.read_text(encoding="utf-8") if styles_path.exists() else ""
    import csv
    with styles_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not existing:
            w.writerow(["name", "prompt", "negative_prompt"])
        for name, p, n in styles:
            if name not in existing:
                w.writerow([name, p, n])
    log(f"✔ styles.csv ({len(styles)} preset NoobAI)")


# --------------------------------------------------------------------------------------
# Bước 4: tự kiểm tra
# --------------------------------------------------------------------------------------
def self_test(cfg: Config) -> bool:
    banner("Tự kiểm tra bộ cài")
    results: list[tuple[bool, str]] = []

    def check(ok: bool, msg: str) -> None:
        results.append((ok, msg))
        print(("  ✅ " if ok else "  ❌ ") + msg, flush=True)

    py = Paths.python()
    check(py.exists(), f"Python venv: {py}")
    if py.exists():
        ver = capture([str(py), "-c", "import sys;print('%d.%d'%sys.version_info[:2])"])
        check(ver in ("3.10", "3.11"), f"Phiên bản Python venv = {ver} (Forge cần 3.10/3.11)")
        probe = capture([str(py), "-c",
                         "import torch,json;print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),"
                         "'dev':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}))"])
        try:
            t = json.loads(probe)
            check(True, f"torch {t['torch']} – thiết bị: {t['dev']}")
            if has_nvidia_gpu() and not cfg.test_mode:
                check(t["cuda"], "torch nhìn thấy GPU CUDA")
        except Exception:  # noqa: BLE001
            check(False, f"import torch thất bại: {probe}")
        for mod in ("gradio", "transformers", "diffusers", "safetensors", "clip", "open_clip", "cv2"):
            rc = subprocess.run([str(py), "-c", f"import {mod}"], capture_output=True).returncode
            check(rc == 0, f"module {mod}")
        rc = subprocess.run([str(py), "-c", "import pytorch_lightning, matplotlib"], capture_output=True,
                            env={**os.environ, "MPLBACKEND": "module://matplotlib_inline.backend_inline", **forge_env()}).returncode
        check(rc == 0, "import pytorch_lightning/matplotlib với env kiểu Colab (MPLBACKEND)")
        npv = capture([str(py), "-c", "import numpy;print(numpy.__version__)"])
        check(npv.startswith("1."), f"numpy {npv} (Forge cần numpy 1.x – extension không được nâng lên 2.x)")
        hx = capture([str(py), "-c", "import httpx;print(httpx.__version__)"])
        check(hx == "0.24.1", f"httpx {hx} (Forge ghim 0.24.1)")
        if "adetailer" in cfg.extensions:
            rc = subprocess.run([str(py), "-c", "import ultralytics"], capture_output=True).returncode
            check(rc == 0, "ultralytics (ADetailer)")
        if cfg.use_xformers and has_nvidia_gpu() and not cfg.test_mode:
            rc = subprocess.run([str(py), "-c", "import xformers"], capture_output=True).returncode
            check(rc == 0, "xformers")

    check((Paths.forge / "launch.py").exists(), f"Forge tại {Paths.forge}")
    commit = capture(["git", "-C", str(Paths.forge), "rev-parse", "--short", "HEAD"]) if Paths.forge.exists() else "?"
    log(f"Forge commit: {commit}")
    for key in cfg.extensions:
        folder = EXTENSIONS.get(key, (key,))[0]
        check((Paths.extensions() / folder).exists(), f"extension {folder}")

    for item in planned_downloads(cfg):
        p = dest_for(item)
        if p.exists():
            size_ok = item["size"] is None or p.stat().st_size == item["size"]
            check(size_ok, f"{item['kind']}/{item['name']} ({human(p.stat().st_size)})")
        else:
            check(False, f"{item['kind']}/{item['name']} CHƯA tải")

    if "tagcomplete" in cfg.extensions and cfg.use_repo_tag_csv:
        csv_path = Paths.extensions() / EXTENSIONS["tagcomplete"][0] / "tags" / TAG_CSV_NAME
        if csv_path.exists():
            n_lines = sum(1 for _ in csv_path.open("rb"))
            bad = 0
            import csv as _csv
            with csv_path.open(encoding="utf-8", newline="") as f:
                for i, row in enumerate(_csv.reader(f)):
                    if len(row) < 3 or not row[1].isdigit() or not row[2].isdigit():
                        bad += 1
                    if i > 5000:
                        break
            check(bad == 0, f"file tag autocomplete hợp lệ ({n_lines:,} tag, định dạng tag,category,count,aliases)")
        else:
            check(False, "file tag của repo (sẽ dùng file mặc định của extension)")

    for name in ("config.json", "ui-config.json", "styles.csv"):
        p = Paths.forge / name
        ok = p.exists()
        if ok and name.endswith(".json"):
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                ok = False
        check(ok, f"{name}")

    plan = gpu_plan(cfg)
    check(True, f"chế độ GPU '{plan['mode']}': {' '.join(plan['flags']) or 'mặc định Forge'}"
                f" | GPU Weights ≈ {max(plan['vram'] - plan['settings']['forge_inference_memory'], 0)} MB")

    tools_py = Paths.forge / "scripts" / NOOB_TOOLS_NAME
    tools_ok = tools_py.exists()
    if tools_ok:
        try:
            compile(tools_py.read_text(encoding="utf-8"), str(tools_py), "exec")
        except Exception:  # noqa: BLE001
            tools_ok = False
    check(tools_ok and (Paths.forge / "noob_tools.json").exists(),
          f"NoobAI Tools (độ phân giải chuẩn + tự fix: hires={cfg.auto_hires_fix}, adetailer={cfg.auto_adetailer})")

    if str(cfg.ui_language).lower().startswith("vi"):
        loc = Paths.forge / "localizations" / f"{VI_LOCALE_NAME}.json"
        n_keys = 0
        try:
            n_keys = len(json.loads(loc.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
        try:
            applied = json.loads((Paths.forge / "config.json").read_text(encoding="utf-8")).get("localization")
        except Exception:  # noqa: BLE001
            applied = None
        check(n_keys > 100 and applied == VI_LOCALE_NAME,
              f"giao diện tiếng Việt ({n_keys} chuỗi dịch, localization={applied})")

    free = shutil.disk_usage(str(ROOT)).free
    check(free > 3 * 1024 ** 3, f"dung lượng trống còn {human(free)}")

    failed = [m for ok, m in results if not ok]
    print()
    if failed:
        print(f"⚠ {len(failed)} mục chưa đạt:", flush=True)
        for m in failed:
            print("   -", m)
    else:
        print("🎉 Tất cả kiểm tra đều đạt – sẵn sàng khởi chạy!", flush=True)
    return not failed


# --------------------------------------------------------------------------------------
# Bước 5: khởi chạy Forge + gradio.live
# --------------------------------------------------------------------------------------
def build_args(cfg: Config) -> list[str]:
    # --skip-install: installer của extension đã chạy ở bước cài (prepare_forge_environment) → khởi động nhanh,
    # không bị pip đổi phiên bản gói mỗi lần chạy. Thêm extension mới thì chạy lại cell "Cài đặt".
    args = ["--listen", "--port", str(cfg.port), "--enable-insecure-extension-access", "--api",
            "--no-download-sd-model", "--skip-version-check", "--skip-python-version-check", "--skip-install",
            "--theme", cfg.theme]
    if cfg.tunnel in ("gradio.live", "both"):
        args.append("--share")
    if cfg.tunnel in ("ngrok", "both") and cfg.ngrok_token:
        args += ["--ngrok", cfg.ngrok_token]
    if cfg.gradio_auth:
        args += ["--gradio-auth", cfg.gradio_auth]
    if cfg.use_xformers and has_nvidia_gpu() and not cfg.test_mode:
        args.append("--xformers")
    if cfg.test_mode or not has_nvidia_gpu():
        args += ["--skip-torch-cuda-test", "--always-cpu"]
    else:
        args += gpu_plan(cfg)["flags"]
    if cfg.extra_args:
        args += cfg.extra_args.split()
    return args


URL_PATTERNS = [
    re.compile(r"https://[a-z0-9-]+\.gradio\.live"),
    re.compile(r"https://[a-z0-9.-]+\.ngrok(?:-free)?\.(?:app|io|dev)"),
]


LAUNCH_LOG = ROOT / "noobai_launch.log"

KNOWN_ERRORS = [
    ("Your device does not support the current version of Torch/CUDA",
     "Torch không thấy GPU. Kiểm tra Runtime ▸ Change runtime type = GPU, rồi chạy lại cell 3 (cài) và cell 6."),
    ("CUDA out of memory", "Hết VRAM. Giảm kích thước ảnh / tắt ControlNet, hoặc thêm `--always-low-vram` vào EXTRA_ARGS."),
    ("Cannot find empty port", "Cổng 7860 đang bị một Forge cũ chiếm. Chạy lại cell 6 (bộ cài sẽ tự dọn tiến trình cũ)."),
    ("address already in use", "Cổng 7860 đang bị chiếm. Chạy lại cell 6 (bộ cài sẽ tự dọn tiến trình cũ)."),
    ("ModuleNotFoundError", "Thiếu thư viện Python. Chạy lại cell 3 (cài đặt) rồi cell 6."),
    ("Could not create share link", "gradio.live không tạo được link. Dùng LINK DỰ PHÒNG (Colab proxy) in bên dưới, hoặc đổi TUNNEL=ngrok."),
    ("No module named 'xformers'", "xformers lỗi. Đặt USE_XFORMERS=False ở cell 1, chạy lại cell 1 và 6."),
    ("Killed", "Tiến trình bị hệ thống kill (hết RAM). Dùng runtime High-RAM hoặc tắt bớt extension."),
    ("No checkpoints found", "Chưa có model. Chạy lại cell 4 (tải tài nguyên)."),
    ("matplotlib_inline.backend_inline", "Biến MPLBACKEND của Colab rò sang Forge – bản lib mới đã sửa; chạy lại cell 2 rồi cell 6."),
]


def kill_stale_forge() -> None:
    """Dọn tiến trình Forge cũ còn chạy ngầm (ví dụ cell trước bị văng) để không kẹt cổng."""
    pids = forge_pids()
    for pid in pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                pass
    if pids:
        log(f"đã dừng {len(pids)} tiến trình Forge cũ")
        time.sleep(3)
        for pid in forge_pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass


def colab_proxy_url(port: int) -> str | None:
    """Link dự phòng qua proxy của Colab (chỉ người đang đăng nhập Colab mở được)."""
    try:
        from google.colab.output import eval_js  # type: ignore
        return str(eval_js(f"google.colab.kernel.proxyPort({port})", timeout_sec=20)).rstrip("/")
    except Exception:  # noqa: BLE001
        return None


def diagnose(lines: list[str], returncode: int | None) -> None:
    print("\n" + "!" * 70, flush=True)
    print(f"!  Forge đã THOÁT (mã {returncode}). Log đầy đủ: {LAUNCH_LOG}", flush=True)
    text = "\n".join(lines)
    hints = [h for key, h in KNOWN_ERRORS if key in text]
    if hints:
        print("!  Nguyên nhân có thể:", flush=True)
        for h in dict.fromkeys(hints):
            print(f"!   • {h}", flush=True)
    err_lines = [ln for ln in lines if re.search(r"Traceback|Error|error:|Exception|Killed", ln)]
    if err_lines:
        print("!  Các dòng lỗi cuối:", flush=True)
        for ln in err_lines[-8:]:
            print("!    " + ln.rstrip()[:200], flush=True)
    print("!  → Chạy cell 🩺 Chẩn đoán để xem 80 dòng log cuối, hoặc gửi file log để được hỗ trợ.", flush=True)
    print("!" * 70 + "\n", flush=True)


URLS_FILE = ROOT / "noobai_urls.json"
PID_FILE = ROOT / "noobai_forge.pid"
# Sau khi UI đã lên, chỉ in các dòng log quan trọng (tránh làm nặng trình duyệt Colab → dễ mất kết nối)
IMPORTANT_LOG = re.compile(r"(error|exception|traceback|warning|out of memory|outofmemory|cuda|gradio\.live|ngrok|"
                           r"running on|total progress|model loaded|loading weights|interrupted|skipped|"
                           r"killed|stopped|restart)", re.I)


def forge_pids() -> list[int]:
    """PID các tiến trình Forge (launch.py / webui.py chạy trong thư mục Forge) đang chạy nền."""
    me = os.getpid()
    forge_dir = str(Paths.forge.resolve())
    pids = []
    for pid_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(pid_dir.name)
            if pid == me:
                continue
            cmd = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            if "launch.py" not in cmd and "webui.py" not in cmd:
                continue
            if forge_dir in cmd or os.path.realpath(str(pid_dir / "cwd")) == forge_dir:
                pids.append(pid)
        except Exception:  # noqa: BLE001
            pass
    return pids


def port_open(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0


def forge_alive(cfg: Config | None = None) -> bool:
    cfg = cfg or Config.load()
    return bool(forge_pids()) and port_open(cfg.port)


def _save_urls(urls: list[str], proxy: str | None) -> None:
    try:
        URLS_FILE.write_text(json.dumps({"public": urls, "proxy": proxy, "time": time.time()}), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def saved_urls() -> dict:
    try:
        return json.loads(URLS_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def urls_from_log() -> list[str]:
    if not LAUNCH_LOG.exists():
        return []
    text = LAUNCH_LOG.read_text(encoding="utf-8", errors="replace")
    found: list[str] = []
    for pat in URL_PATTERNS:
        for m in pat.findall(text):
            if m not in found:
                found.append(m)
    return found


def show_ready(url: str, label: str = "GIAO DIỆN ĐÃ SẴN SÀNG") -> None:
    print("\n" + "★" * 70 + f"\n★  {label} → {url}\n" + "★" * 70 + "\n", flush=True)


def stop_forge() -> int:
    """Dừng hẳn Forge đang chạy nền."""
    pids = forge_pids()
    for pid in pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                pass
    time.sleep(3)
    for pid in forge_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    if pids:
        log(f"⏹ đã dừng Forge (pid {pids})")
    else:
        log("Forge không chạy")
    return len(pids)


def launch(cfg: Config, *, wait: bool = True, extra_args: Iterable[str] = (), ready_timeout: int = 0,
           quiet_after_ready: bool = True):
    """Khởi chạy Forge dưới dạng tiến trình nền ĐỘC LẬP (session riêng, log ghi thẳng ra file).

    - Cell Colab chỉ "theo dõi" log: trình duyệt mất kết nối / cell bị ngắt → Forge vẫn chạy, link vẫn sống.
    - Dùng reconnect() (cell 🔗) để lấy lại link, stop_forge() để dừng hẳn.
    """
    banner("Khởi chạy NoobAI-XL 1.1 (Forge)")
    py = Paths.python()
    if not py.exists() or not (Paths.forge / "launch.py").exists():
        raise RuntimeError("Chưa cài đặt – hãy chạy cell 3 (Cài đặt) trước.")
    kill_stale_forge()
    fix_opencv_if_no_libgl()
    args = build_args(cfg) + list(extra_args)
    plan = gpu_plan(cfg)
    log(f"GPU: {gpu_name()} | chế độ GPU: {plan['mode']}" + (f" (compute {plan['cc']})" if plan['cc'] else ""))
    for w in plan["why"]:
        log("   · " + w)
    log("COMMANDLINE_ARGS = " + " ".join(args))
    log(f"log được ghi vào {LAUNCH_LOG}")
    env = {**os.environ, **pip_env(), **forge_env(), "COMMANDLINE_ARGS": " ".join(args)}
    for k in list(env):
        if k.startswith(("JPY_", "IPY", "COLAB_BACKEND", "KERNEL_")):
            env.pop(k, None)
    if cfg.hf_token:
        env["HF_TOKEN"] = cfg.hf_token
    if URLS_FILE.exists():
        URLS_FILE.unlink()
    logf = LAUNCH_LOG.open("wb")
    proc = subprocess.Popen([str(py), str(Paths.forge / "launch.py")], cwd=str(Paths.forge), env=env,
                            stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)  # session riêng → không chết theo cell/kernel
    logf.close()
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    log(f"Forge chạy nền, pid {proc.pid} (mất kết nối Colab cũng không tắt; dừng hẳn bằng cell 🔗 → Dừng)")
    public_urls = follow_log(cfg, proc, wait=wait, ready_timeout=ready_timeout, quiet_after_ready=quiet_after_ready)
    return proc, public_urls


def follow_log(cfg: Config, proc: "subprocess.Popen | None" = None, *, wait: bool = True, ready_timeout: int = 0,
               quiet_after_ready: bool = True, from_start: bool = True) -> list[str]:
    """Theo dõi noobai_launch.log, in link khi UI lên; trả về danh sách link công khai."""
    public_urls: list[str] = list(saved_urls().get("public", [])) if not from_start else []
    local_ready = not from_start and bool(saved_urls())
    share_failed = False
    proxy = None
    tail: list[str] = []
    t0 = time.time()
    buf = ""

    def alive() -> bool:
        if proc is not None:
            return proc.poll() is None
        return bool(forge_pids())

    def handle(line: str) -> bool:
        nonlocal local_ready, share_failed, proxy
        tail.append(line)
        if len(tail) > 400:
            del tail[:100]
        important = bool(IMPORTANT_LOG.search(line))
        if not local_ready or not quiet_after_ready or important:
            try:
                sys.stdout.write(line.rsplit("\r", 1)[-1] if "\r" in line else line)
                sys.stdout.flush()
            except Exception:  # noqa: BLE001
                pass
        for pat in URL_PATTERNS:
            for m in pat.findall(line):
                if m not in public_urls:
                    public_urls.append(m)
                    show_ready(m)
                    _save_urls(public_urls, proxy)
        if "Could not create share link" in line:
            share_failed = True
        if "Running on local URL" in line and not local_ready:
            local_ready = True
            proxy = colab_proxy_url(cfg.port)
            _save_urls(public_urls, proxy)
            if proxy:
                show_ready(proxy, "LINK DỰ PHÒNG (Colab proxy – mở bằng tài khoản đang chạy Colab)")
                print("   (link gradio.live sẽ hiện thêm bên dưới nếu tạo được – thường mất 10–60 s)\n", flush=True)
            if quiet_after_ready:
                print("   (từ đây chỉ in các dòng log quan trọng; log đầy đủ ở", LAUNCH_LOG, ")\n", flush=True)
        if share_failed and local_ready and cfg.tunnel in ("gradio.live", "both") and \
                "gradio.live" not in "".join(public_urls):
            print("\n⚠ gradio.live không tạo được link lúc này. Hãy dùng LINK DỰ PHÒNG ở trên, "
                  "hoặc đặt TUNNEL=ngrok (cell 1) rồi chạy lại cell 6.\n", flush=True)
            share_failed = False
        if not wait and (local_ready and (public_urls or cfg.tunnel == "none")):
            return False
        if ready_timeout and time.time() - t0 > ready_timeout:
            return False
        return True

    interrupted = False
    try:
        with LAUNCH_LOG.open("rb") as f:
            if not from_start:
                f.seek(0, os.SEEK_END)
            while True:
                chunk = f.read(65536)
                if chunk:
                    buf += chunk.decode("utf-8", errors="replace")
                    *lines, buf = buf.split("\n")
                    for ln in lines:
                        if not handle(ln + "\n"):
                            return public_urls
                    continue
                if not alive():
                    if buf:
                        handle(buf + "\n")
                    break
                if not wait and ready_timeout and time.time() - t0 > ready_timeout:
                    return public_urls
                time.sleep(0.5)
    except KeyboardInterrupt:
        interrupted = True
        print("\n⏸ Đã ngừng theo dõi log – Forge VẪN CHẠY NỀN, link vẫn dùng được.\n"
              "   Lấy lại link: cell 🔗 (Nối lại). Dừng hẳn Forge: cell 🔗 với ACTION = Dừng.", flush=True)
    if wait and not interrupted and not alive():
        rc = proc.returncode if proc is not None else None
        diagnose(tail, rc)
    return public_urls


def reconnect(cfg: Config | None = None, action: str = "link") -> None:
    """Cell 🔗: 'link' = in lại link (khởi chạy lại nếu Forge đã chết) | 'restart' | 'stop'."""
    cfg = cfg or Config.load()
    banner("🔗 Nối lại với Forge")
    if action == "stop":
        stop_forge()
        return
    if action == "restart":
        stop_forge()
        launch(cfg)
        return
    if forge_alive(cfg):
        info = saved_urls()
        urls = info.get("public") or urls_from_log()
        proxy = colab_proxy_url(cfg.port) or info.get("proxy")
        log(f"Forge đang chạy (pid {forge_pids()}, cổng {cfg.port} mở) – link hiện tại:")
        for u in urls:
            show_ready(u)
        if proxy:
            show_ready(proxy, "LINK DỰ PHÒNG (Colab proxy)")
        if not urls and cfg.tunnel != "none":
            log("⚠ chưa thấy link gradio.live trong log – gradio.live có thể đang lỗi; dùng link dự phòng, "
                "hoặc ACTION = Khởi động lại.")
        print("\nĐang tiếp tục theo dõi log (bấm ⏹ để ngừng theo dõi, Forge vẫn chạy)…\n", flush=True)
        follow_log(cfg, None, wait=True, from_start=False)
    else:
        if forge_pids():
            log(f"Forge đang khởi động (pid {forge_pids()}), cổng {cfg.port} chưa mở – theo dõi log:")
            follow_log(cfg, None, wait=True, from_start=True)
        else:
            log("Forge KHÔNG chạy (runtime có thể đã bị Colab thu hồi / khởi động lại) → khởi chạy lại…")
            if not (Paths.forge / "launch.py").exists() or not Paths.python().exists():
                log("⚠ môi trường không còn (Colab đã reset máy) → cần chạy lại cell 3 → 4 → 6.")
                return
            launch(cfg)


def doctor(cfg: Config | None = None, n: int = 80) -> None:
    """Cell 🩺: tóm tắt trạng thái + 80 dòng log cuối của lần khởi chạy gần nhất."""
    cfg = cfg or Config.load()
    banner("🩺 Chẩn đoán")
    log(f"Python hệ thống {platform.python_version()} | GPU: {gpu_name()} | Colab: {is_colab()}")
    log(f"Forge: {'có' if (Paths.forge / 'launch.py').exists() else 'CHƯA cài'} | venv: {'có' if Paths.python().exists() else 'CHƯA có'}")
    free = shutil.disk_usage(str(ROOT)).free
    log(f"đĩa trống: {human(free)}")
    try:
        mem = [ln for ln in Path("/proc/meminfo").read_text().splitlines() if ln.startswith(("MemTotal", "MemAvailable"))]
        log("RAM: " + " | ".join(" ".join(m.split()) for m in mem))
    except Exception:  # noqa: BLE001
        pass
    running = forge_pids()
    log(f"Forge đang chạy ngầm: {running or 'không'} | cổng {cfg.port}: {'mở' if port_open(cfg.port) else 'đóng'}")
    for u in saved_urls().get("public", []):
        log(f"link gần nhất: {u}")
    model = dest_for(MODEL_FILES["noobai_xl_eps_1_1"])
    log(f"model: {model} → {'OK ' + human(model.stat().st_size) if model.exists() else 'CHƯA có'}")
    if LAUNCH_LOG.exists():
        lines = LAUNCH_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"\n--- {n} dòng cuối của {LAUNCH_LOG} ({len(lines)} dòng) ---")
        for ln in lines[-n:]:
            print(ln[:300])
        text = "\n".join(lines)
        hints = [h for key, h in KNOWN_ERRORS if key in text]
        if hints:
            print("\nGợi ý:")
            for h in dict.fromkeys(hints):
                print(" •", h)
    else:
        print("\nChưa có log khởi chạy – hãy chạy cell 6 trước.")


# --------------------------------------------------------------------------------------
# Luồng chính
# --------------------------------------------------------------------------------------
def install(cfg: Config) -> dict:
    banner(f"NoobAI-XL EPS 1.1 Installer v{LIB_VERSION} – cài môi trường")
    log(f"Python hệ thống: {platform.python_version()}  |  GPU: {gpu_name()}  |  thư mục: {ROOT}")
    ensure_system_tools()
    ensure_python_env(cfg)
    install_forge(cfg)
    install_extensions(cfg)
    prepare_forge_environment(cfg)
    install_tag_csv(cfg)
    install_localization(cfg)
    install_noob_tools(cfg)
    drive_info = setup_drive(cfg)
    write_settings(cfg, drive_info)
    (ROOT / "noobai_drive.json").write_text(json.dumps(drive_info), encoding="utf-8")
    log("✔ cài đặt môi trường hoàn tất")
    return drive_info


def download(cfg: Config) -> None:
    download_resources(cfg)
    # ghi lại config.json để chắc chắn tên model/VAE trỏ đúng file đã tải
    drive_path = ROOT / "noobai_drive.json"
    drive_info = json.loads(drive_path.read_text()) if drive_path.exists() else {}
    write_settings(cfg, drive_info)


def set_ui_language(cfg: Config, language: str | None = None) -> str:
    """Đổi ngôn ngữ giao diện (vi/en) mà không cần cài lại; có hiệu lực sau khi khởi động lại / Reload UI."""
    if language:
        cfg.ui_language = language
    if str(cfg.ui_language).lower().startswith("vi"):
        install_localization(cfg)
    drive_path = ROOT / "noobai_drive.json"
    drive_info = json.loads(drive_path.read_text()) if drive_path.exists() else {}
    write_settings(cfg, drive_info)
    return json.loads((Paths.forge / "config.json").read_text(encoding="utf-8")).get("localization", "None")


def main(argv: list[str]) -> int:
    cfg = Config.load()
    step = argv[1] if len(argv) > 1 else "all"
    if step == "lang":
        print("localization =", set_ui_language(cfg, argv[2] if len(argv) > 2 else None))
        return 0
    if step == "settings":  # ghi lại config.json / ui-config.json / noob_tools.json theo noobai_config.json hiện tại
        install_localization(cfg)
        install_noob_tools(cfg)
        drive_path = ROOT / "noobai_drive.json"
        write_settings(cfg, json.loads(drive_path.read_text()) if drive_path.exists() else {})
        return 0
    if step in ("install", "all"):
        install(cfg)
    if step in ("download", "all"):
        download(cfg)
    if step in ("test", "all"):
        if not self_test(cfg) and step == "all":
            print("Có mục kiểm tra chưa đạt – vẫn thử khởi chạy, xem log để sửa.")
    if step in ("launch", "all"):
        launch(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
