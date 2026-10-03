# anime2 – NoobAI-XL 1.1 Colab + Danbooru/e621 tag list

## 🚀 NoobAI-XL EPS 1.1 trên Google Colab (1-click)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manhlee1196-boop/anime2/blob/main/NoobAI_XL_1.1_Colab.ipynb)

Notebook **`NoobAI_XL_1.1_Colab.ipynb`** cài [NoobAI-XL (NAI-XL) Epsilon-pred 1.1](https://civitai.com/models/833294?modelVersionId=1116447)
trên nền **Stable Diffusion WebUI Forge**, chia sẻ giao diện qua **gradio.live** (hoặc ngrok).

| Thành phần | Chi tiết |
|---|---|
| Model | `NoobAI-XL-v1.1.safetensors` (7,1 GB) – tải từ HuggingFace, kiểm tra kích thước/SHA256 |
| VAE | `sdxl-vae-fp16-fix` |
| Upscaler | 4x-AnimeSharp, 4x-UltraSharp |
| ControlNet | Union ProMax (mặc định) / bộ ControlNet riêng của NoobAI (canny, openpose, depth, lineart_anime, tile, scribble, softedge, manga_line) |
| Extensions | Tag Autocomplete (dùng file tag trong repo này), ADetailer (+ model YOLO tải sẵn), Infinite Image Browsing, Ultimate SD Upscale, Dynamic Prompts, Civitai Helper, WD14 Tagger*, Regional Prompter* |
| Mặc định | Prompt/negative chính thức của NoobAI, Euler a · 28 steps · CFG 6 · 832×1216 · Clip skip 2, 14 preset Styles |
| Drive | Lưu ảnh ra Drive; liên kết `MyDrive/NoobAI/{models,Lora,embeddings,VAE,ControlNet}` |
| Thông số | Dropdown **⚙️ Thông số tạo ảnh** trên Sampling method: 5 bộ khuyên dùng (chuẩn / nháp nhanh / chất lượng cao / sáng tạo / chính xác) đặt toàn bộ sampler·steps·CFG·kích thước·Hires fix, hoặc *Tuỳ chỉnh* (tự chuyển khi sửa tay); cell 1 `PARAM_MODE` |
| Độ phân giải | Dropdown **📐 Độ phân giải chuẩn** (9 bucket ~1 MP của SDXL/NoobAI) ngay dưới Width/Height; mặc định 832×1216 |
| Tự fix | **Hires. fix** (4x-AnimeSharp ×1.5, denoise 0.4) và **ADetailer** (mặt `face_yolov8n` + tay `hand_yolov8n`) mở sẵn khi vào UI – `AUTO_HIRES_FIX` / `AUTO_ADETAILER` ở cell 1 |
| Giao diện | **Tiếng Việt** (song ngữ, giữ thuật ngữ Prompt/Seed/CFG/LoRA…) – `colab/vi_VN.json`, ~830 chuỗi; đổi `UI_LANGUAGE = en` để dùng tiếng Anh |
| GPU | `GPU_MODE = max` (mặc định): `--always-high-vram` (T4) / `--always-gpu` (≥ 20 GB) + `--cuda-malloc` + `--vae-in-fp16` → model nằm hẳn trên GPU, không hoán đổi; `balanced` / `lowvram` khi cần ảnh rất lớn |
| Chống mất kết nối | Forge chạy **nền độc lập** với cell (session riêng, log ra file); mất kết nối/ngắt cell không làm tắt Forge; cell 🔗 lấy lại link / khởi động lại / dừng; cell ⏱ hướng dẫn giữ phiên |
| Kiểm tra | Cell tự kiểm tra môi trường/model/phiên bản thư viện/bản dịch trước khi chạy |

\* tuỳ chọn, tắt mặc định.

### Cách dùng
1. Mở notebook bằng nút trên → Runtime ▸ Change runtime type ▸ **GPU (T4+)**.
2. Chỉnh form ở cell **1️⃣** (hoặc để mặc định) → **Runtime ▸ Run all**.
3. Chờ dòng `★ GIAO DIỆN ĐÃ SẴN SÀNG → https://xxxx.gradio.live` ở cell **6️⃣** rồi bấm vào link.

Thời gian lần đầu trên T4: ~3–5 phút cài + ~2–4 phút tải (~9,4 GB mặc định).

### Cấu trúc
```
NoobAI_XL_1.1_Colab.ipynb   # notebook (SINH TỰ ĐỘNG – đừng sửa tay)
colab/noobai_lib.py         # toàn bộ logic cài đặt / tải / cấu hình / tự kiểm tra / khởi chạy
colab/build_notebook.py     # sinh notebook từ noobai_lib.py (+ nhúng vi_VN.json):  python colab/build_notebook.py
colab/vi_VN.json            # bản dịch tiếng Việt cho giao diện Forge (cài vào <forge>/localizations/vi_VN.json)
colab/noob_tools.py         # script Forge: dropdown độ phân giải chuẩn + tự bật Hires fix/ADetailer (cài vào <forge>/scripts/)
danbooru_e621_merged_*.csv  # 349 714 tag Danbooru + e621 (định dạng Tag Autocomplete: tag,category,count,aliases)
```

Vì Colab hiện dùng Python 3.12/3.13 còn Forge chỉ hỗ trợ ≤ 3.11, bộ cài tự tạo môi trường **Python 3.10** riêng bằng `uv`
(fallback: `python3.10/3.11` có sẵn → `apt`), cài torch 2.3.1 + cu121, và **ghim phiên bản** (numpy 1.26, httpx 0.24…)
trong lúc chạy installer của các extension để chúng không phá Forge.

Chạy thử ngoài Colab (CPU, chỉ để kiểm tra bộ cài):
```bash
echo '{"test_mode": true, "download_model": false, "mount_drive": false, "tunnel": "none"}' > /tmp/noob/noobai_config.json
NOOBAI_ROOT=/tmp/noob python colab/noobai_lib.py all
```

Đổi cấu hình (ngôn ngữ, tự fix, độ phân giải mặc định…) sau khi đã cài: sửa `noobai_config.json` (cell 1) rồi
`python colab/noobai_lib.py settings` → khởi động lại UI. Hướng dẫn tăng độ phân giải (Hires fix / Extras / Ultimate SD Upscale) nằm trong notebook.

Đổi ngôn ngữ giao diện sau khi đã cài (không cần cài lại):
`python colab/noobai_lib.py lang vi` hoặc `lang en`, hoặc trong UI: *Cài đặt ▸ Giao diện người dùng ▸ Ngôn ngữ (Localization)* → `vi_VN` / `None` → Áp dụng → Tải lại UI.

> ⚠️ Colab miễn phí có thể hạn chế việc chạy WebUI; khuyến nghị Colab Pro.

## 📄 File tag `danbooru_e621_merged_2026-10-01_pt20-ia-dd-ed-spc.csv`
Danh sách tag Danbooru + e621 gộp (349 714 dòng) dùng cho extension
[a1111-sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete). Notebook sẽ tự nạp file này làm nguồn gợi ý mặc định.
