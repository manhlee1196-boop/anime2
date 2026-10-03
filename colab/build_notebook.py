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
VI_LOCALE = HERE / "vi_VN.json"
NOOB_TOOLS = HERE / "noob_tools.py"
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
#@markdown Ngôn ngữ giao diện: `vi` = tiếng Việt (song ngữ, giữ thuật ngữ Prompt/Seed/CFG…), `en` = tiếng Anh gốc
UI_LANGUAGE = "vi"  #@param ["vi", "en"]
USE_XFORMERS = True  #@param {type:"boolean"}
EXTRA_ARGS = ""  #@param {type:"string"}
#@markdown Commit Forge đã kiểm thử; đặt `latest` để lấy bản mới nhất (có thể phát sinh lỗi mới)
FORGE_COMMIT = "dfdcbab685e57677014f05a3309b48cc87383167"  #@param {type:"string"}

#@markdown ### 🎛 Thông số tạo ảnh
#@markdown **Khuyên dùng** = bộ chuẩn NoobAI (Euler a · 28 steps · CFG 6 · 832×1216 · Clip skip 2 · Hires fix ×1.5/0.4 · ADetailer mặt+tay) – các ô bên dưới sẽ bị bỏ qua. **Tự chỉnh** = dùng các ô bên dưới. Trong UI vẫn đổi được bằng dropdown ⚙️ phía trên Sampling method.
PARAM_MODE = "Khuyên dùng (NoobAI chuẩn)"  #@param ["Khuyên dùng (NoobAI chuẩn)", "Tự chỉnh (dùng các ô bên dưới)"]
#@markdown Độ phân giải chuẩn (bucket ~1 MP mà NoobAI-XL được huấn luyện). Trong UI vẫn đổi được bằng dropdown 📐 dưới Width/Height.
DEFAULT_RESOLUTION = "832x1216 (dọc 2:3 – chuẩn NoobAI)"  #@param ["832x1216 (dọc 2:3 – chuẩn NoobAI)", "1216x832 (ngang 3:2)", "1024x1024 (vuông)", "896x1152 (dọc 7:9)", "1152x896 (ngang 9:7)", "768x1344 (dọc 9:16)", "1344x768 (ngang 16:9)", "640x1536 (dọc 5:12)", "1536x640 (ngang 12:5)"]
#@markdown Hoặc tự nhập (ví dụ `1024x1536`), để trống để dùng mục trên
CUSTOM_RESOLUTION = ""  #@param {type:"string"}
DEFAULT_STEPS = 28  #@param {type:"integer"}
DEFAULT_CFG = 6.0  #@param {type:"number"}
DEFAULT_SAMPLER = "Euler a"  #@param ["Euler a", "Euler", "DPM++ 2M", "DPM++ 2M SDE", "DPM++ 3M SDE", "DPM++ SDE", "Restart"]
DEFAULT_SCHEDULER = "Automatic"  #@param ["Automatic", "Karras", "Exponential", "SGM Uniform", "Simple", "Normal"]
CLIP_SKIP = 2  #@param {type:"integer"}

#@markdown ### ✨ Tự fix (mở sẵn khi vào UI – tắt được từng ảnh bằng cách bỏ tích)
#@markdown Hires. fix: vẽ ảnh ở độ phân giải chuẩn rồi tự phóng to bằng 4x-AnimeSharp và vẽ thêm chi tiết
AUTO_HIRES_FIX = True  #@param {type:"boolean"}
HIRES_UPSCALE_BY = 1.5  #@param [1.25, 1.5, 1.75, 2.0] {type:"raw"}
HIRES_DENOISE = 0.4  #@param {type:"number"}
#@markdown ADetailer: tự phát hiện và vẽ lại mặt (bộ 1) và tay (bộ 2) sau khi tạo ảnh
AUTO_ADETAILER = "face+hand"  #@param ["face+hand", "face", "off"]

import json, os, re
_res = CUSTOM_RESOLUTION.strip() or DEFAULT_RESOLUTION
_m = re.match(r"\s*(\d+)\s*[xX×]\s*(\d+)", _res)
assert _m, f"Độ phân giải không hợp lệ: {_res!r} (đúng dạng 832x1216)"
DEFAULT_WIDTH, DEFAULT_HEIGHT = (int(_m.group(1)) // 8 * 8, int(_m.group(2)) // 8 * 8)
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
    gradio_auth=GRADIO_AUTH.strip(), theme=THEME, ui_language=UI_LANGUAGE, use_xformers=USE_XFORMERS, extra_args=EXTRA_ARGS.strip(),
    forge_commit=FORGE_COMMIT.strip(), default_width=DEFAULT_WIDTH, default_height=DEFAULT_HEIGHT,
    default_steps=DEFAULT_STEPS, default_cfg=DEFAULT_CFG, default_sampler=DEFAULT_SAMPLER,
    default_scheduler=DEFAULT_SCHEDULER, clip_skip=CLIP_SKIP,
    auto_hires_fix=AUTO_HIRES_FIX, hires_upscale_by=float(HIRES_UPSCALE_BY), hires_denoise=float(HIRES_DENOISE),
    auto_adetailer=AUTO_ADETAILER,
    param_mode="recommended" if PARAM_MODE.startswith("Khuyên") else "custom",
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
#@markdown Forge chạy **nền độc lập** với cell này: trình duyệt mất kết nối hay cell bị ngắt thì Forge vẫn sống, link vẫn dùng được – chạy cell 🔗 để lấy lại link. Cell này chỉ theo dõi log (sau khi UI lên chỉ in dòng quan trọng). Muốn dừng hẳn Forge: cell 🔗 với ACTION = Dừng.
import importlib, sys
sys.path.insert(0, "/content")
import noobai_lib as L
importlib.reload(L)
cfg = L.Config.load()
if L.forge_alive(cfg):
    print("Forge đang chạy sẵn → chỉ lấy lại link (muốn khởi động lại: cell 🔗, ACTION = Khởi động lại)")
    L.reconnect(cfg, "link")
else:
    L.launch(cfg)
'''

RECONNECT_CODE = '''#@title 🔗 Nối lại sau khi Colab mất kết nối: lấy lại link / khởi động lại / dừng
#@markdown - **Lấy lại link**: Forge còn sống → in lại link; đã chết → tự khởi chạy lại (môi trường còn thì ~1 phút).
#@markdown - Nếu Colab đã **reset máy** (mất /content) cell sẽ báo → chạy lại cell 3 → 4 → 6 (model trên Drive không phải tải lại nếu đã bật LINK_DRIVE_MODELS và để model trong Drive/NoobAI/models).
ACTION = "Lấy lại link"  #@param ["Lấy lại link", "Khởi động lại", "Dừng"]
import importlib, sys
sys.path.insert(0, "/content")
try:
    import noobai_lib as L
except ModuleNotFoundError:
    raise SystemExit("⚠ /content/noobai_lib.py không còn → Colab đã reset máy. Chạy lại từ cell 1 (cell 2 → 3 → 4 → 6).")
importlib.reload(L)
L.reconnect(L.Config.load(), {"Lấy lại link": "link", "Khởi động lại": "restart", "Dừng": "stop"}[ACTION])
'''

KEEPALIVE_CODE = '''#@title ⏱ Chống Colab tự ngắt kết nối (đọc kỹ rồi làm theo)
#@markdown Colab miễn phí ngắt phiên khi **tab Colab không có tương tác ~90 phút** (dù bạn đang dùng gradio.live ở tab khác), tối đa 12 h/phiên, và có thể thu hồi GPU bất kỳ lúc nào. Cell này in **trạng thái** và **đoạn JS giữ phiên** để bạn dán vào Console của trình duyệt (F12 → Console) **trên tab Colab**. Lưu ý: tự động hoá tương tác có thể vi phạm điều khoản Colab – dùng có chừng mực, tự chịu trách nhiệm; cách an toàn nhất là thỉnh thoảng quay lại tab Colab bấm vào đâu đó, hoặc dùng Colab Pro (background execution).
import importlib, sys, time
sys.path.insert(0, "/content")
import noobai_lib as L
importlib.reload(L)
cfg = L.Config.load()
print("Forge:", "đang chạy ✅" if L.forge_alive(cfg) else "KHÔNG chạy ❌ (chạy cell 🔗)")
info = L.saved_urls()
for u in info.get("public", []):
    print("Link:", u)
try:
    import subprocess
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=10).stdout.strip())
except Exception:
    pass
print("""
──── Dán vào Console (F12) của TAB COLAB rồi Enter – cứ 60 s sẽ "chạm" vào trang để Colab không coi là bỏ không ────
(function(){
  if (window._noobKeep) clearInterval(window._noobKeep);
  window._noobKeep = setInterval(function(){
    try {
      var btn = document.querySelector("colab-connect-button");
      if (btn && btn.shadowRoot) { var b = btn.shadowRoot.querySelector("#connect"); if (b) b.click(); }
      document.dispatchEvent(new MouseEvent("mousemove", {bubbles:true}));
      console.log("NoobAI keep-alive", new Date().toLocaleTimeString());
    } catch(e) { console.log("keep-alive lỗi", e); }
  }, 60000);
  console.log("NoobAI keep-alive: BẬT (tắt bằng clearInterval(window._noobKeep))");
})();
────────────────────────────────────────────────────────────────────────────────────────────────────────
Mẹo thêm: • Giữ tab Colab mở (không thu nhỏ trình duyệt). • Ảnh đã lưu thẳng vào Drive/NoobAI/outputs nên mất phiên cũng không mất ảnh.
• Mất kết nối rồi nối lại được (Runtime ▸ Reconnect) → chạy cell 🔗. • Báo "Your session crashed after using all available RAM":
  giảm batch size, đóng tab/ứng dụng khác của Colab, hoặc thêm `--always-low-vram` vào EXTRA_ARGS.""")
'''

DOCTOR_CODE = '''#@title 🩺 Chẩn đoán khi không thấy link / Forge thoát sớm
import importlib, sys
sys.path.insert(0, "/content")
import noobai_lib as L
importlib.reload(L)
L.doctor(L.Config.load())
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

## ⚙️ Thông số khuyên dùng hay tự chỉnh?
- Cell 1 `PARAM_MODE`: **Khuyên dùng** (bộ NoobAI chuẩn, bỏ qua các ô chi tiết) hoặc **Tự chỉnh** (dùng các ô Width/Height/Steps/CFG/Sampler/Hires/ADetailer).
- Trong UI, dropdown **⚙️ Thông số tạo ảnh** nằm ngay trên *Sampling method* (txt2img và img2img). Mỗi bộ đặt **toàn bộ** sampler · scheduler · steps · CFG · kích thước · batch · Hires fix (bật/tắt, upscaler, ×, steps, denoise, CFG):

| Bộ (txt2img) | Khi nào dùng | Thông số |
|---|---|---|
| ✅ Khuyên dùng – NoobAI chuẩn | mặc định, đa số trường hợp | Euler a · Automatic · 28 · CFG 6 · 832×1216 · Hires 4x-AnimeSharp ×1.5, 14 steps, denoise 0.4 |
| ⚡ Nháp nhanh | thử prompt / tìm seed | Euler a · 20 · CFG 5.5 · Hires tắt (nhanh gấp ~3) |
| 💎 Chất lượng cao | ảnh cuối, in ấn | DPM++ 2M SDE · Karras · 32 · CFG 6.5 · Hires ×2, 16 steps, denoise 0.35 |
| 🎨 Sáng tạo | muốn đa dạng, ít bám prompt | Euler a · 30 · CFG 4.5 · Hires ×1.5, denoise 0.45 |
| 🎯 Chính xác | bám prompt chặt, ít ngẫu nhiên | DPM++ 2M · Karras · 30 · CFG 7.5 · Hires ×1.5, denoise 0.3 |
| ✏️ Tuỳ chỉnh | tự đặt | tự động chuyển sang mục này khi bạn sửa tay bất kỳ thông số nào |

img2img có bộ riêng: *Khuyên dùng* (denoise 0.5) · *Nhẹ* (0.3, giữ ảnh) · *Mạnh* (0.7, vẽ lại nhiều) · *Phóng to* (DPM++ 2M Karras, 0.3 – dùng với Ultimate SD Upscale).

Ý nghĩa từng thông số: **Steps** – số bước vẽ (20 nháp, 28 chuẩn, 32–40 kỹ; hơn 40 gần như không đẹp thêm). **CFG** – độ bám prompt (4–5 sáng tạo, 6 chuẩn, 7–8 chặt; >9 cháy màu). **Sampler** – Euler a mềm/anime, DPM++ 2M (SDE) Karras sắc nét hơn. **Denoise** (Hires/img2img) – mức được phép thay đổi ảnh (0.3 giữ, 0.5 vừa, 0.7 vẽ lại). **Clip skip** = 2 cho NoobAI (thanh trên cùng). **Batch count/size** – số ảnh mỗi lần bấm (T4: batch size 1–2).

## 📐 Độ phân giải chuẩn
NoobAI-XL (SDXL) được huấn luyện ở ~1 megapixel – vẽ đúng các cỡ này sẽ ít lỗi tay/chân/bố cục nhất:

| Dọc | Ngang | Vuông |
|---|---|---|
| **832×1216** (2:3, khuyên dùng) | **1216×832** (3:2) | **1024×1024** |
| 896×1152 (7:9) | 1152×896 (9:7) | |
| 768×1344 (9:16 điện thoại) | 1344×768 (16:9 màn hình) | |
| 640×1536 (5:12) | 1536×640 (12:5 banner) | |

Trong UI: dropdown **📐 Độ phân giải chuẩn** nằm ngay dưới Width/Height (txt2img và img2img) – chọn là tự điền; kéo tay Width/Height thì dropdown chuyển sang *Tuỳ chỉnh*. Mặc định đặt ở cell 1 (`DEFAULT_RESOLUTION`).
> Đừng vẽ thẳng 2048×2048 – model sẽ sinh thừa người/thừa chi. Muốn ảnh to hãy vẽ ở cỡ chuẩn rồi **tăng độ phân giải** như dưới.

## 🔍 Tăng độ phân giải cho ảnh (3 cách, từ nhanh đến kỹ)
1. **Hires. fix – ngay khi tạo ảnh (mặc định đã bật ✨)**: ảnh được vẽ ở cỡ chuẩn, phóng to bằng *4x-AnimeSharp* × **1.5** (832×1216 → 1248×1824) rồi vẽ thêm chi tiết với *Denoising* **0.4**.
   - Muốn to hơn: *Phóng to gấp* 2.0 (→ 1664×2432, T4 mất ~2 phút/ảnh). Denoise 0.3 giữ sát ảnh gốc, 0.5 thêm chi tiết nhưng có thể đổi nét mặt.
   - Tắt cho một ảnh: bỏ tích ở tiêu đề mục *Hires. fix*. Mẹo: vẽ nháp tắt Hires fix, tìm được seed ưng ý thì ♻️ dùng lại seed và bật Hires fix.
2. **Extras – phóng to ảnh có sẵn, không vẽ lại (vài giây)**: tab *Công cụ thêm (Extras)* → kéo ảnh vào → *Upscaler 1* = `4x-AnimeSharp` (anime) hoặc `4x-UltraSharp` (chi tiết/thực), *Phóng to gấp* 2–4 → *Tạo ảnh*. Nhanh, giữ nguyên nội dung, nhưng không thêm chi tiết mới.
3. **Ultimate SD Upscale – phóng to + vẽ lại chi tiết theo ô (kỹ nhất, 4K được)**: ở gallery bấm *Gửi sang img2img* → trong img2img: *Mức khử nhiễu* **0.25–0.35**, kéo xuống *Script* chọn **Ultimate SD upscale** → *Target size type* = *Scale from image size*, *Scale* = 2, *Upscaler* = 4x-AnimeSharp, *Tile width/height* 1024, *Padding* 32, *Seams fix* = *Half tile offset pass* → *Tạo ảnh*. Có thể bật thêm ControlNet **Tile** (model `noob_sdxl_controlnet_tile` nếu đã tải) để bám sát ảnh gốc hơn.

## ✨ Tự fix (đã bật sẵn, chỉnh ở cell 1)
- **ADetailer** chạy sau mỗi ảnh: bộ 1 `face_yolov8n.pt` tìm và vẽ lại **mặt**, bộ 2 `hand_yolov8n.pt` vẽ lại **tay** (denoise 0.4). Mặt vẫn lỗi → tăng *Inpaint denoising strength* lên 0.5; ảnh nhiều người → giảm *Detection confidence* 0.3 → 0.25; không muốn sửa tay → chọn `face`.
- **Hires. fix** như mục trên. Cả hai đều tắt được từng lần bằng cách bỏ tích trong UI; muốn tắt hẳn: `AUTO_HIRES_FIX = False` / `AUTO_ADETAILER = off` ở cell 1 → chạy lại cell 1 → `!python /content/noobai_lib.py settings` (hoặc chạy lại cell 3) → cell 6.
- Ảnh vẫn lỗi tay: thêm style *Noob ✦ Negative: tay/anatomy*, hoặc ở gallery → *Gửi sang inpaint* → tô vùng tay → *Chỉ vùng tô*, denoise 0.5 → tạo lại vài lần.

## 🛠 Xử lý sự cố
| Hiện tượng | Cách xử lý |
|---|---|
| **Colab tự ngắt kết nối** | Phân biệt 2 trường hợp: (a) chỉ mất kết nối trình duyệt (góc phải báo *Reconnect*, RAM/Disk vẫn hiện) → bấm Reconnect, chạy cell **🔗** là có lại link, Forge chưa hề tắt; (b) runtime bị thu hồi (máy mới, /content trống) → chạy lại cell 2 → 3 → 4 → 6 (ảnh đã nằm trên Drive). Nguyên nhân thường gặp: tab Colab bỏ không ~90 phút trong khi bạn dùng gradio.live ở tab khác → xem cell **⏱** |
| Link gradio.live cũ không mở được | Forge đã tắt theo runtime → cell 🔗; nếu Forge còn sống mà gradio.live lỗi → cell 🔗 ACTION = Khởi động lại, hoặc dùng link dự phòng |
| Không thấy link gradio.live | Cell 6 luôn in thêm **LINK DỰ PHÒNG (Colab proxy)** ngay khi UI lên – dùng link đó. Nếu cell 6 *kết thúc* (không chạy mãi) tức Forge đã thoát: chạy cell 🩺 để xem lỗi, hoặc đổi `TUNNEL = ngrok` |
| `CUDA out of memory` | Giảm kích thước ảnh / batch, tắt ControlNet, hoặc thêm `--always-low-vram` vào EXTRA_ARGS |
| Model không hiện trong danh sách | Chạy lại cell 4 (tải sẽ tiếp tục nếu bị đứt), bấm 🔄 Refresh trong UI |
| Colab báo "disallowed code" | Giới hạn của Colab miễn phí với WebUI – cân nhắc Colab Pro |
| Muốn đổi UI sang tiếng Anh / Việt | Trong UI: **Cài đặt ▸ Giao diện người dùng ▸ Ngôn ngữ (Localization)** chọn `None` (Anh) hoặc `vi_VN` (Việt) → *Áp dụng cài đặt* → *Tải lại UI*. Hoặc đặt `UI_LANGUAGE` ở cell 1 rồi chạy lại cell 1 → 3 → 6 |
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


def embed_locale(lib_src: str) -> str:
    """Chèn colab/vi_VN.json vào hằng VI_LOCALE_EMBEDDED để notebook tự chứa bản dịch tiếng Việt."""
    data = json.loads(VI_LOCALE.read_text(encoding="utf-8"))
    data.pop("__comment__", None)
    marker = "VI_LOCALE_EMBEDDED: dict | None = None"
    assert marker in lib_src, "không tìm thấy VI_LOCALE_EMBEDDED trong noobai_lib.py"
    literal = json.dumps(data, ensure_ascii=False, indent=1)  # JSON object = dict literal hợp lệ trong Python
    lib_src = lib_src.replace(marker, f"VI_LOCALE_EMBEDDED: dict | None = {literal}", 1)
    # script noob_tools.py (độ phân giải chuẩn + tự fix)
    src = NOOB_TOOLS.read_text(encoding="utf-8")
    compile(src, str(NOOB_TOOLS), "exec")
    marker2 = "NOOB_TOOLS_EMBEDDED: str | None = None"
    assert marker2 in lib_src, "không tìm thấy NOOB_TOOLS_EMBEDDED trong noobai_lib.py"
    return lib_src.replace(marker2, f"NOOB_TOOLS_EMBEDDED: str | None = {src!r}", 1)


def build() -> dict:
    lib_src = embed_locale(LIB.read_text(encoding="utf-8"))
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
        code(RECONNECT_CODE, cellview="form"),
        code(KEEPALIVE_CODE, cellview="form"),
        code(DOCTOR_CODE, cellview="form"),
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
