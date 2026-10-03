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
| Kiểm tra | Cell tự kiểm tra môi trường/model/phiên bản thư viện trước khi chạy |

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
colab/build_notebook.py     # sinh notebook từ noobai_lib.py:  python colab/build_notebook.py
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

> ⚠️ Colab miễn phí có thể hạn chế việc chạy WebUI; khuyến nghị Colab Pro.

## 📄 File tag `danbooru_e621_merged_2026-10-01_pt20-ia-dd-ed-spc.csv`
Danh sách tag Danbooru + e621 gộp (349 714 dòng) dùng cho extension
[a1111-sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete). Notebook sẽ tự nạp file này làm nguồn gợi ý mặc định.
