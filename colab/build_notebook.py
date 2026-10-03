# -*- coding: utf-8 -*-
"""Sinh file NoobAI_XL_1.1_Colab.ipynb từ colab/noobai_lib.py.

Chạy:  python colab/build_notebook.py
Notebook là sản phẩm sinh ra – KHÔNG sửa tay, hãy sửa noobai_lib.py / file này rồi build lại.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LIB = HERE / "noobai_lib.py"
OUT = REPO / "NoobAI_XL_1.1_Colab.ipynb"

GITHUB_USER_REPO = "manhlee1196-boop/anime2"

# ----------------------------------------------------------------------------- cells
INTRO_MD = f"""# 🎨 NoobAI-XL EPS 1.1 – Bộ cài 1-click cho Google Colab

Cài **NoobAI-XL (NAI-XL) Epsilon-pred 1.1** trên nền **Stable Diffusion WebUI Forge** với đầy đủ tính năng,
giao diện Gradio chia sẻ qua **gradio.live** (hoặc ngrok).

| Có sẵn | Chi tiết |
|---|---|
| 🧠 Model | NoobAI-XL-v1.1.safetensors (7,1 GB, kiểm tra kích thước + SHA256) |
| 🎨 VAE | sdxl-vae-fp16-fix (chống ảnh đen/NaN khi chạy fp16) |
| 🔍 Upscaler | 4x-AnimeSharp, 4x-UltraSharp (Hires fix / Extras / Ultimate Upscale) |
| 🕹 ControlNet | ControlNet Union ProMax (1 file đủ canny/depth/openpose/lineart/tile…) hoặc bộ ControlNet riêng của NoobAI |
| 🧩 Extensions | Tag Autocomplete (dùng **file tag Danbooru+e621 của repo này**), ADetailer, Infinite Image Browsing, Ultimate SD Upscale, Dynamic Prompts, Civitai Helper (+ tuỳ chọn WD14 Tagger, Regional Prompter) |
| ⚙️ Mặc định NoobAI | Prompt chất lượng + negative chính thức, Euler a · 28 steps · CFG 6 · 832×1216 · Clip skip 2, 14 preset Styles |
| ☁️ Google Drive | Lưu ảnh ra Drive, tự liên kết thư mục `MyDrive/NoobAI/(models|Lora|embeddings|VAE|ControlNet)` |
| ✅ Tự kiểm tra | Cell kiểm tra môi trường, file model, phiên bản thư viện trước khi chạy |

**Yêu cầu:** Runtime → Change runtime type → **GPU** (T4 trở lên). Dung lượng đĩa cần ~15 GB (mặc định) – ~35 GB (tải hết ControlNet).

> ⚠️ Google Colab **miễn phí** có thể hạn chế/cảnh báo khi chạy WebUI (theo điều khoản của Colab). Khuyến nghị **Colab Pro**. Bộ cài này không vi phạm gì thêm – chỉ cần bạn tự chịu trách nhiệm với tài khoản của mình.

**Cách dùng nhanh:** chỉnh form ở cell 1 (hoặc để mặc định) → **Runtime ▸ Run all** → chờ đến khi hiện dòng `★ GIAO DIỆN ĐÃ SẴN SÀNG → https://xxxx.gradio.live` → bấm vào link.
"""

CONFIG_CODE = '''#@title 1️⃣ Cấu hình (chỉnh rồi chạy – có thể để mặc định) { display-mode: "form" }
#@markdown ### 🧠 Model & tài nguyên
DOWNLOAD_MODEL = True  #@param {type:"boolean"}
DOWNLOAD_VAE_FP16_FIX = True  #@param {type:"boolean"}
DOWNLOAD_UPSCALERS = True  #@param {type:"boolean"}
#@markdown `none` = không ControlNet · `union_promax` = 1 file 2,5 GB dùng được mọi loại · `noob_core` = 5 ControlNet riêng của NoobAI (12,5 GB) · `noob_all` = 9 file (22 GB)
CONTROLNET = "union_promax"  #@param ["none", "union_promax", "noob_core", "noob_all"]
#@markdown Link tải thêm (HuggingFace / Civitai…), nhiều link cách nhau bằng dấu phẩy. Link trang Civitai dạng `…/models/123?modelVersionId=456` cũng được.
EXTRA_MODEL_URLS = ""  #@param {type:"string"}
EXTRA_LORA_URLS = ""  #@param {type:"string"}
EXTRA_EMBEDDING_URLS = ""  #@param {type:"string"}
CIVITAI_TOKEN = ""  #@param {type:"string"}
HF_TOKEN = ""  #@param {type:"string"}
VERIFY_SHA256 = False  #@param {type:"boolean"}

#@markdown ### 🧩 Extensions
EXT_TAGCOMPLETE = True  #@param {type:"boolean"}
USE_REPO_TAG_CSV = True  #@param {type:"boolean"}
EXT_ADETAILER = True  #@param {type:"boolean"}
EXT_IMAGE_BROWSER = True  #@param {type:"boolean"}
EXT_ULTIMATE_UPSCALE = True  #@param {type:"boolean"}
EXT_DYNAMIC_PROMPTS = True  #@param {type:"boolean"}
EXT_CIVITAI_HELPER = True  #@param {type:"boolean"}
EXT_WD14_TAGGER = False  #@param {type:"boolean"}
EXT_REGIONAL_PROMPTER = False  #@param {type:"boolean"}

#@markdown ### ☁️ Google Drive
MOUNT_DRIVE = True  #@param {type:"boolean"}
SAVE_OUTPUTS_TO_DRIVE = True  #@param {type:"boolean"}
LINK_DRIVE_MODELS = True  #@param {type:"boolean"}
DRIVE_FOLDER = "NoobAI"  #@param {type:"string"}

#@markdown ### 🌐 Giao diện / đường hầm
TUNNEL = "gradio.live"  #@param ["gradio.live", "ngrok", "both"]
NGROK_TOKEN = ""  #@param {type:"string"}
#@markdown Đặt mật khẩu cho link công khai (dạng `user:pass`), để trống nếu không cần
GRADIO_AUTH = ""  #@param {type:"string"}
THEME = "dark"  #@param ["dark", "light"]
USE_XFORMERS = True  #@param {type:"boolean"}
EXTRA_ARGS = ""  #@param {type:"string"}
#@markdown Commit Forge đã kiểm thử; đặt `latest` để lấy bản mới nhất (có thể phát sinh lỗi mới)
FORGE_COMMIT = "dfdcbab685e57677014f05a3309b48cc87383167"  #@param {type:"string"}

#@markdown ### 🎛 Mặc định khi mở UI
DEFAULT_WIDTH = 832  #@param {type:"integer"}
DEFAULT_HEIGHT = 1216  #@param {type:"integer"}
DEFAULT_STEPS = 28  #@param {type:"integer"}
DEFAULT_CFG = 6.0  #@param {type:"number"}
DEFAULT_SAMPLER = "Euler a"  #@param ["Euler a", "Euler", "DPM++ 2M", "DPM++ 2M SDE", "DPM++ 3M SDE", "DPM++ SDE", "Restart"]
DEFAULT_SCHEDULER = "Automatic"  #@param ["Automatic", "Karras", "Exponential", "SGM Uniform", "Simple", "Normal"]
CLIP_SKIP = 2  #@param {type:"integer"}

import json, os
extensions = [k for k, on in {
    "tagcomplete": EXT_TAGCOMPLETE, "adetailer": EXT_ADETAILER, "image_browser": EXT_IMAGE_BROWSER,
    "ultimate_upscale": EXT_ULTIMATE_UPSCALE, "dynamic_prompts": EXT_DYNAMIC_PROMPTS,
    "civitai_helper": EXT_CIVITAI_HELPER, "wd14_tagger": EXT_WD14_TAGGER, "regional_prompter": EXT_REGIONAL_PROMPTER,
}.items() if on]
config = dict(
    download_model=DOWNLOAD_MODEL, download_vae=DOWNLOAD_VAE_FP16_FIX, download_upscalers=DOWNLOAD_UPSCALERS,
    controlnet=CONTROLNET, extra_model_urls=EXTRA_MODEL_URLS, extra_lora_urls=EXTRA_LORA_URLS,
    extra_embedding_urls=EXTRA_EMBEDDING_URLS, civitai_token=CIVITAI_TOKEN.strip(), hf_token=HF_TOKEN.strip(),
    verify_sha256=VERIFY_SHA256, extensions=extensions, use_repo_tag_csv=USE_REPO_TAG_CSV,
    mount_drive=MOUNT_DRIVE, save_outputs_to_drive=SAVE_OUTPUTS_TO_DRIVE, link_drive_models=LINK_DRIVE_MODELS,
    drive_folder=DRIVE_FOLDER.strip() or "NoobAI", tunnel=TUNNEL, ngrok_token=NGROK_TOKEN.strip(),
    gradio_auth=GRADIO_AUTH.strip(), theme=THEME, use_xformers=USE_XFORMERS, extra_args=EXTRA_ARGS.strip(),
    forge_commit=FORGE_COMMIT.strip(), default_width=DEFAULT_WIDTH, default_height=DEFAULT_HEIGHT,
    default_steps=DEFAULT_STEPS, default_cfg=DEFAULT_CFG, default_sampler=DEFAULT_SAMPLER,
    default_scheduler=DEFAULT_SCHEDULER, clip_skip=CLIP_SKIP,
)
os.makedirs("/content", exist_ok=True)
with open("/content/noobai_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print("✔ Đã lưu cấu hình vào /content/noobai_config.json")
print(json.dumps(config, indent=2, ensure_ascii=False))
'''

INSTALL_CODE = '''#@title 3️⃣ Cài đặt môi trường (Python 3.10 + Forge + extensions) – ~3–5 phút
import importlib, sys
sys.path.insert(0, "/content")
import noobai_lib as L
importlib.reload(L)
cfg = L.Config.load()
drive_info = L.install(cfg)
'''

DOWNLOAD_CODE = '''#@title 4️⃣ Tải model / VAE / upscaler / ControlNet – ~2–6 phút tuỳ lựa chọn
import importlib, sys
sys.path.insert(0, "/content")
import noobai_lib as L
importlib.reload(L)
cfg = L.Config.load()
L.download(cfg)
'''

TEST_CODE = '''#@title 5️⃣ Tự kiểm tra trước khi chạy
import importlib, sys
sys.path.insert(0, "/content")
import noobai_lib as L
importlib.reload(L)
cfg = L.Config.load()
ok = L.self_test(cfg)
if not ok:
    print("\\n⚠ Có mục chưa đạt. Thường chỉ cần chạy lại cell 3 hoặc 4. Nếu vẫn lỗi, copy log ở trên để được hỗ trợ.")
'''

LAUNCH_CODE = '''#@title 6️⃣ 🚀 Khởi chạy giao diện (link gradio.live sẽ hiện ở đây)
#@markdown Cell này chạy liên tục trong lúc bạn dùng WebUI. Muốn dừng: bấm ⏹ của cell.
import importlib, sys
sys.path.insert(0, "/content")
import noobai_lib as L
importlib.reload(L)
cfg = L.Config.load()
L.launch(cfg)
'''

EXTRA_DL_CODE = '''#@title 🧰 (Tuỳ chọn) Tải thêm model / LoRA / embedding bất kỳ lúc nào { display-mode: "form" }
#@markdown Dán link rồi chạy. Sau đó trong WebUI bấm nút 🔄 Refresh cạnh danh sách model/LoRA là thấy ngay (không cần khởi động lại).
URL = ""  #@param {type:"string"}
KIND = "Lora"  #@param ["Lora", "Stable-diffusion", "VAE", "ControlNet", "embeddings", "ESRGAN"]
FILENAME = ""  #@param {type:"string"}
import importlib, sys
sys.path.insert(0, "/content")
import noobai_lib as L
importlib.reload(L)
cfg = L.Config.load()
for u in L.split_urls(URL):
    u = L.normalize_url(u)
    name = FILENAME.strip() or L._guess_filename(u)
    dest = L.dest_for(dict(kind=KIND, name=name))
    L.download_file(u, dest, cfg)
    print("→", dest)
'''

BACKUP_CODE = '''#@title 💾 (Tuỳ chọn) Nén toàn bộ ảnh đã tạo thành ZIP (để tải về / lưu Drive)
import shutil, time, json, os
from pathlib import Path
info_path = Path("/content/noobai_drive.json")
outputs = Path(json.loads(info_path.read_text())["outputs"]) if info_path.exists() else Path("/content/noobai-forge/outputs")
zip_base = f"/content/noobai_outputs_{time.strftime('%Y%m%d_%H%M%S')}"
shutil.make_archive(zip_base, "zip", outputs)
print("✔ Đã nén:", zip_base + ".zip", f"({os.path.getsize(zip_base + '.zip')/1e6:.1f} MB)")
try:
    from google.colab import files
    files.download(zip_base + ".zip")
except Exception:
    pass
'''

GUIDE_MD = """## 📖 Mẹo dùng NoobAI-XL EPS 1.1

**Prompt chuẩn (đã điền sẵn trong UI):**
```
masterpiece, best quality, newest, absurdres, highres, <tag Danbooru của bạn…>
```
**Negative chính thức:**
```
worst quality, old, early, low quality, lowres, signature, username, logo, bad hands, mutated hands, mammal, anthro, furry, ambiguous form, feral, semi-anthro
```
- Sampler **Euler a**, 24–32 steps, **CFG 5–7**, Clip skip 2, kích thước ≈ 1 MP (832×1216, 1024×1024, 1216×832…).
- Nhân vật: gõ tên theo Danbooru, ví dụ `hatsune_miku, vocaloid` – Tag Autocomplete sẽ gợi ý từ **349 714 tag** của repo (gõ `_` hoặc chữ cái đầu, Tab để chọn).
- Phong cách hoạ sĩ: `artist:xxx` hoặc chỉ tên hoạ sĩ theo Danbooru. Năm: `newest / recent / mid / early / old`.
- Hires fix: 4x-AnimeSharp × 1.5, denoise 0.35–0.45 (đã đặt mặc định). ADetailer: bật `face_yolov8n.pt` để sửa mặt.
- ControlNet Union ProMax: chọn model `xinsir_controlnet_union_sdxl_promax`, chọn preprocessor tương ứng (canny/depth/openpose/lineart/tile…).

## 🛠 Xử lý sự cố
| Hiện tượng | Cách xử lý |
|---|---|
| Không thấy link gradio.live | Đợi thêm 30–60 s; hoặc đổi `TUNNEL = ngrok` (cần token miễn phí tại ngrok.com) |
| `CUDA out of memory` | Giảm kích thước ảnh / batch, tắt ControlNet, hoặc thêm `--always-low-vram` vào EXTRA_ARGS |
| Model không hiện trong danh sách | Chạy lại cell 4 (tải sẽ tiếp tục nếu bị đứt), bấm 🔄 Refresh trong UI |
| Colab báo "disallowed code" | Giới hạn của Colab miễn phí với WebUI – cân nhắc Colab Pro |
| Thêm extension mới qua UI | Sau khi cài trong tab Extensions, chạy lại cell 3 rồi cell 6 (để installer chạy với ràng buộc phiên bản) |

Mọi logic cài đặt nằm trong `colab/noobai_lib.py` của repo – notebook này được sinh tự động bởi `colab/build_notebook.py`.
"""


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str, *, cellview: str | None = None) -> dict:
    meta: dict = {}
    if cellview:
        meta["cellView"] = cellview
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": src}


def build() -> dict:
    lib_src = LIB.read_text(encoding="utf-8")
    lib_cell = "#@title 2️⃣ Ghi thư viện cài đặt (noobai_lib.py) – chỉ cần chạy, không cần sửa\n" \
               "%%writefile /content/noobai_lib.py\n" + lib_src
    cells = [
        md(INTRO_MD),
        code(CONFIG_CODE, cellview="form"),
        code(lib_cell, cellview="form"),
        code(INSTALL_CODE, cellview="form"),
        code(DOWNLOAD_CODE, cellview="form"),
        code(TEST_CODE, cellview="form"),
        code(LAUNCH_CODE, cellview="form"),
        md(GUIDE_MD),
        code(EXTRA_DL_CODE, cellview="form"),
        code(BACKUP_CODE, cellview="form"),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4", "name": "NoobAI_XL_1.1_Colab.ipynb", "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


if __name__ == "__main__":
    nb = build()
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"✔ wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size/1024:.0f} KB, {len(nb['cells'])} cells)")
