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
