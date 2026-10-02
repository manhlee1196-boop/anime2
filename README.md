# Anime Studio — Colab + Gradio

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manhlee1196-boop/anime2/blob/arena%2F01a0fc9c-anime2/Animagine_XL_4_LPW_Colab.ipynb)

## Chạy thử

1. Bấm **Open In Colab** ở trên; bản thử nghiệm nằm trên nhánh `arena/01a0fc9c-anime2`, chưa nằm trên `main`.
2. Chọn **Runtime → Change runtime type → GPU** (T4 nếu có).
3. Chạy lần lượt ô 1–5. Nếu ô cài thư viện yêu cầu restart, restart rồi chạy từ ô 2.
4. Ô cuối hỏi mật khẩu rồi in link `https://…gradio.live`; đăng nhập bằng **anime** và mật khẩu vừa đặt.
5. Thử tạo ảnh → sửa vùng nếu cần → phóng to bằng AI hoặc Lanczos → tải PNG về máy trước khi runtime kết thúc.

Nếu link Colab không mở được nhánh có dấu `/`, tải `Animagine_XL_4_LPW_Colab.ipynb` từ nhánh này, vào Colab → **File → Upload notebook**.

- Animagine XL 4.0 + LPW, prompt >100 token (giới hạn giao diện: 300 token).
- Chỉnh kích thước, steps, CFG, seed; tải PNG và JSON thông số.
- FP16 + CPU offload; mỗi lần 1 ảnh, hàng đợi tuần tự.
- Link tạm, ngừng khi runtime dừng. Không phải hosting 24/7.
- Colab miễn phí hạn chế dùng chủ yếu qua web UI: dùng gói trả phí còn compute units hoặc GPU riêng. Xem https://research.google.com/colaboratory/faq.html#disallowed-activities.

Chưa kiểm thử inference trên GPU Colab. Kiểm tra cấu trúc/cú pháp notebook: `python tests/test_notebook.py`.

## CSV gợi ý tag
Trong giao diện mở **Tìm và thêm tag từ CSV**, bấm **Nạp CSV** để tải bản dữ liệu GitHub được ghim tại commit `b65bea25a2dabf46859b70db3f78258b68de47a4`, hoặc upload CSV riêng. CSV không header, 4 cột: tag, category ID, post count, aliases. Tìm tên/alias rồi chọn tag và thêm vào prompt/negative. Không tự huấn luyện model. Danh sách merged có thể chứa tag người lớn. Các category ID được giữ nguyên, không suy đoán mapping.

## Sửa vùng và phóng to
- Tab **Tạo ảnh** có nút chuyển ảnh sang tab **Sửa vùng** hoặc **Phóng to**.
- Trong **Sửa vùng**, upload ảnh hoặc chuyển ảnh vừa tạo, tô mask bằng cọ, nhập prompt mô tả kết quả mong muốn, chọn strength (bắt đầu 0.55), bấm **Sửa vùng đã tô**. Tô cả cổ tay/khớp liên quan; ảnh cần được sửa lỗi cấu trúc trước khi upscale.
- Crop vùng sửa bật mặc định, bổ sung ngữ cảnh rồi resize vùng đó để model xử lý rõ hơn. Sau đó ghép kết quả về kích thước gốc; chỉ mask đã làm mềm được thay đổi.
- Có ảnh trước/sau, PNG/JSON, dùng kết quả để sửa tiếp, hoàn tác bước gần nhất (không phải lịch sử nhiều bước).
- Inpainting dùng model Animagine và LPW sẵn có, không tải checkpoint mới. Hai tác vụ tạo/sửa dùng cùng concurrency ID để không chạy GPU song song. Chưa kiểm thử chất lượng inference trên GPU Colab.
- **Phóng to**: chọn **Real-ESRGAN Anime6B (AI)** hoặc **Lanczos (nhẹ, CPU)**, hệ số 1.5×/2×/4×. Anime6B luôn chạy native x4 trước rồi resize về hệ số yêu cầu. AI có thể thay đổi nét vẽ/texture, không sửa giải phẫu.
- Model AI tải khi dùng lần đầu (~18 MB) từ release chính thức Real-ESRGAN v0.2.2.4. Nạp bằng `torch.load(weights_only=True)`, kiểm tra size và strict state dict. Không cài BasicSR/torchvision; kiến trúc x4 được tích hợp trong notebook, kèm giấy phép Apache-2.0.
- Tile 128 mặc định, 64 nếu thiếu VRAM, 256 nếu có dư VRAM. Tạo/sửa/upscale dùng chung hàng đợi GPU. Khi upscale, diffusion chuyển về CPU; sau upscale model Anime6B được đưa về CPU.
- Giới hạn ảnh AI đầu vào 4 megapixel, ảnh đầu ra 16 megapixel. Có làm nét nhẹ tùy chọn (mặc định tắt).
- Nếu tải model hoặc GPU lỗi: dùng Lanczos. Chưa xác minh checkpoint chính thức hoặc chất lượng ảnh AI trên GPU Colab tại đây do sandbox không truy cập được máy chủ tải weights.

Kiểm thử xử lý mask, ghép vùng, crop, upscale và callbacks với pipeline giả lập trên CPU:
`python -m unittest discover -s tests -v` (cần Pillow và numpy, có sẵn khi cài Gradio).

Kiểm thử AI trên CPU (cần PyTorch): kiến trúc với weights ngẫu nhiên, tải checkpoint giả lập an toàn, chia tile/ghép biên ảnh, resize/metadata, lỗi download và giải phóng GPU giả lập. Không phải kiểm thử chất lượng ảnh của weights pretrained.

## Lỗi `demo is not defined`

Biến `demo` chỉ tồn tại sau khi ô **4. Giao diện** chạy thành công trong cùng runtime. Nếu chỉ chạy ô mở link, ô giao diện gặp lỗi, hoặc runtime restart, hãy chạy lại ô **2 → 3 → 4 → 5** (không cần nạp lại model nếu ô 2 vẫn còn trong phiên). Đợi ô 4 in **Giao diện đã sẵn sàng** trước khi chạy ô 5. Nếu ô 4 báo lỗi, gửi traceback đầu tiên của ô 4 để chẩn đoán; không chỉ chạy lại ô nhập mật khẩu.

Notebook kiểm tra prerequisites và trạng thái giao diện trước khi hỏi mật khẩu. Test không cần GPU: `python -m unittest discover -s tests -p test_launch.py -v`.

## Cảnh báo dependency khi cài trên Colab
Gradio 5.49.1 cần `pydantic<2.12`, `starlette<1.0` và client 1.13.3. Một số gói Colab hiện có (google-adk, google-genai, python-fasthtml, hf-gradio) cần các phiên bản mới hơn: chúng không tương thích hoàn toàn trong cùng phiên. Notebook không sử dụng các gói đó và không tự gỡ chúng. Nếu cần dùng Google ADK/GenAI, hãy dùng một runtime khác.

Ô 1 ghim FastAPI 0.115.12, Starlette 0.46.2, Pydantic 2.11.10 và kiểm tra import + HTTP `/config` của Gradio ở process mới. Nếu in **UI kiểm tra OK**, có thể tiếp tục; nếu có traceback thì gửi lỗi, không chạy tiếp. Không dùng `pip install -U pydantic starlette gradio-client` để chữa cảnh báo vì có thể phá Gradio 5. Sau khi cài, nếu kernel đã import phiên bản khác, **Runtime → Restart session**, chạy ô 2–5; ô 2 phát hiện module cũ và yêu cầu restart.

Bộ Gradio/FastAPI/Starlette/Pydantic này đã qua `pip check` trong venv sạch và HTTP config smoke test trên CPU; không phải xác nhận toàn bộ môi trường Colab hay inference GPU.

## Lỗi `custom_revision: v0.35.1 does not exist` và quy ước revision
`custom_revision` của community pipeline phải là bản trần `"0.35.1"` (không có tiền tố `v`). Diffusers 0.35.1 tự đổi thành thư mục `v0.35.1` trên mirror `diffusers/community-pipelines-mirror` sau khi kiểm tra. Danh sách version trong thông báo lỗi được lấy trực tiếp từ PyPI nên có thể dài hơn bản đã cài — điều đó không có nghĩa runtime đang dùng bản khác.

Runtime Colab mới bắt buộc chạy **ô 1** trước; nếu bỏ qua, Colab có sẵn diffusers khác bản ghim và ô 2 sẽ chặn lại với thông báo yêu cầu chạy ô 1. Notebook có canary `version("diffusers") != "0.35.1"` ngay trước khi nạp model.

Đã kiểm chứng bằng cách chạy lại mã thật của diffusers 0.35.1 (môi trường sạch, không cần mạng): `"0.35.1"` đi qua guard và ghi `v0.35.1/lpw_stable_diffusion_xl.py` vào cache; `"v0.35.1"` gây đúng lỗi ValueError đã gặp trên Colab. Xem `tests/test_diffusers_revision.py`. Chưa chạy được bước tải model thật trên GPU Colab từ môi trường phát triển.

## Chế độ dùng tối đa GPU
Từ bản này pipeline được đặt nguyên trên GPU bằng `pipe.to("cuda")` (FP16, ~7 GB — vừa T4 16 GB), bỏ CPU offload và VAE tiling/slicing nên mỗi bước diffusion không phải chuyển model qua lại giữa CPU/GPU: tạo ảnh nhanh hơn đáng kể. Ô tải model in thêm dòng VRAM đang dùng và `torch.backends.cudnn.benchmark = True` được bật cho các kích thước cố định.

Upscale AI vẫn tạm dời 4 thành phần diffusion sang CPU để lấy VRAM, rồi **trả về đúng thiết bị/độ chính xác gốc** (GPU/FP16) sau khi xong — đã có test kiểm tra cả trường hợp gốc là GPU và gốc là CPU.

Nếu GPU nhỏ hoặc gặp "CUDA out of memory": chạy một ô mới `pipe.enable_model_cpu_offload()` và `pipe.enable_vae_tiling()` để quay lại chế độ tiết kiệm VRAM; thông báo lỗi trong UI cũng gợi ý lệnh này. Vẫn chưa xác minh tốc độ/VRAM thực tế trên GPU Colab từ môi trường phát triển.
