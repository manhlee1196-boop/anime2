# 🩹 Bộ thẻ (tag) sửa MẮT – TAY – CHÂN cho NoobAI-XL / Illustrious

Tổng hợp từ Danbooru tag groups (eyes / hands / feet / legs), negative chính thức của NoobAI-XL và kinh nghiệm cộng đồng
(Civitai, SeaArt, Hugging Face). NoobAI học trên tag Danbooru nên **tag có thật trên Danbooru tác dụng mạnh nhất**
(in đậm); các cụm tiếng Anh tự nhiên (deformed, malformed…) vẫn có tác dụng nhưng yếu hơn.

> Cách dùng nhanh: trong WebUI chọn Style **Noob ✦ Fix mắt / Fix tay / Fix chân / Fix mắt+tay+chân** (đã cài sẵn) – style sẽ
> chèn các thẻ dưới đây vào prompt và negative. ADetailer mặc định đã dùng prompt sửa mặt/tay ở phần 5.

---

## 1. 👁 MẮT

### Positive – thêm vào prompt
| Mục đích | Thẻ |
|---|---|
| Nét, sắc, rõ | **detailed eyes**, beautiful detailed eyes, **eye focus**, **eye reflection**, **sparkling eyes**, **glowing eyes**, **bright pupils**, **tsurime** (mắt xếch, sắc) / **tareme** (mắt cụp, hiền) / **jitome** (mắt lườm) |
| Hướng nhìn, tránh lác | **looking at viewer**, **eye contact**, **looking to the side**, **looking up**, **looking down**, **sideways glance**, **upturned eyes** |
| Lông mi, trang điểm | **eyelashes**, **long eyelashes**, **colored eyelashes**, **eyeliner**, **eyeshadow**, **mascara** |
| Đồng tử đặc biệt | **slit pupils**, **symbol-shaped pupils**, **heart-shaped pupils**, **star-shaped pupils**, **ringed eyes**, **constricted pupils**, **dilated pupils**, **white pupils** |
| Màu / nhiều màu | `<màu> eyes` (blue eyes, red eyes, aqua eyes, amber eyes, purple eyes…), **gradient eyes**, **multicolored eyes**, **heterochromia** (2 màu – chỉ dùng khi muốn) |
| Trạng thái | **half-closed eyes**, **narrowed eyes**, **wide-eyed**, **closed eyes**, **one eye closed**, **wink**, **eyes visible through hair**, **hair over one eye** |
| Góc gần | **close-up**, **portrait**, **face focus** (mắt càng to trong khung thì càng ít lỗi) |

### Negative – thêm vào negative prompt
```
bad eyes, cross-eyed, uneven eyes, asymmetrical eyes, extra eyes, missing eye, lazy eye, empty eyes, dead eyes,
deformed eyes, misaligned eyes, blurry eyes, extra pupils, no pupils, heterochromia
```
(bỏ `heterochromia` nếu nhân vật có 2 màu mắt; bỏ `empty eyes` nếu muốn phong cách mắt trống.)

### Mẹo
- Lỗi mắt lác / lệch: thêm `looking at viewer, eye contact` + giảm CFG xuống 5–6 + ADetailer `face_yolov8n.pt` (mặc định đã bật).
- Mắt mờ ở ảnh toàn thân: bật Hires fix (mặc định) hoặc **Gửi sang inpaint** → tô 2 mắt → *Chỉ vùng tô*, denoise 0.35–0.45, prompt chỉ cần `detailed eyes, <màu> eyes, looking at viewer`.

---

## 2. ✋ TAY

### Positive – thêm vào prompt
| Mục đích | Thẻ |
|---|---|
| Chất lượng | **detailed hands**, perfect hands, beautiful hands, **five fingers**, **fingernails**, **nail polish**, **hand focus** |
| **Tư thế tay rõ ràng (hiệu quả nhất)** | **hand on hip**, **hands on hips**, **hand up**, **hands up**, **arm up**, **arms up**, **hand on own chest**, **hand on own cheek**, **hand on own face**, **hand to own mouth**, **hand in own hair**, **own hands together**, **own hands clasped**, **interlocked fingers**, **crossed arms**, **arms behind back**, **arms behind head**, **arms at sides**, **hand in pocket**, **hands in pockets**, **hand on own knee** |
| Cử chỉ | **v**, **double v**, **peace sign**, **thumbs up**, **ok sign**, **pointing**, **pointing at viewer**, **waving**, **finger gun**, **index finger raised**, **salute**, **fist**, **clenched hand**, **open hand**, **spread fingers**, **fingers together**, **reaching towards viewer**, **outstretched arm**, **palm** |
| Cầm đồ (tay có việc = ít lỗi) | **holding cup**, **holding phone**, **holding book**, **holding umbrella**, **holding weapon**, **holding sword**, **holding flower**, **holding bag**, **holding hands** |
| "Giấu" tay khi cần | **gloves**, **long sleeves**, **sleeves past wrists**, **sleeves past fingers**, **hands in pockets**, **arms behind back**, **upper body** (cắt khung trên eo), **hand out of frame** |

### Negative – thêm vào negative prompt
```
bad hands, mutated hands, malformed hands, deformed hands, poorly drawn hands, extra digits, fewer digits, extra fingers,
missing fingers, fused fingers, too many fingers, long fingers, broken fingers, extra hands, extra arms, missing arms,
bad arms, disconnected limbs, floating limbs, twisted hands, claw
```

### Mẹo
- Tay ít ngón / thừa ngón: **tư thế tay cụ thể** (bảng trên) có tác dụng hơn mọi negative. Tránh để tay "tự do" trong prompt.
- ADetailer bộ 2 `hand_yolov8n.pt` (mặc định đã bật) vẽ lại tay; nếu vẫn lỗi: tăng *Inpaint denoising* 0.5 → 0.6, hoặc inpaint thủ công vùng tay với prompt `detailed hands, five fingers, <tư thế>`.
- Tay cầm đồ, tay sau lưng, tay trong túi = ba cách "né" rẻ nhất khi ảnh không cần tay.

---

## 3. 🦵 CHÂN / BÀN CHÂN

### Positive – thêm vào prompt
| Mục đích | Thẻ |
|---|---|
| Chất lượng | detailed legs, **detailed feet**, **feet**, **toes**, **toenails**, **toenail polish**, **soles**, **foot focus**, **barefoot**, **legs**, **bare legs**, **thighs**, **thigh gap**, **zettai ryouiki** |
| **Tư thế chân rõ ràng** | **standing**, **standing on one leg**, **legs together**, **legs apart**, **crossed legs**, **knees together feet apart**, **knees up**, **hugging own legs**, **sitting**, **seiza**, **wariza**, **indian style**, **kneeling**, **on one knee**, **squatting**, **leg up**, **leg lift**, **outstretched leg**, **tiptoes**, **walking**, **running**, **jumping** |
| Trang phục chân | **thighhighs**, **kneehighs**, **pantyhose**, **socks**, **white socks**, **shoes**, **boots**, **sandals**, **high heels**, **loafers**, **sneakers**, **mary janes** |
| Khung hình | **full body** (cần Hires fix để chân không mờ), **lower body**, **cowboy shot** (cắt ở đùi – né bàn chân), **foot out of frame** |

### Negative – thêm vào negative prompt
```
bad feet, bad legs, bad leg, extra legs, missing legs, deformed feet, malformed feet, poorly drawn feet, fused toes,
extra toes, missing toes, too many toes, twisted legs, bad knees, disconnected legs, extra limbs, bad proportions,
huge feet, tiny feet, wrong feet
```

### Mẹo
- Ảnh **full body** ở 832×1216: bàn chân chỉ vài chục pixel → luôn bật Hires fix ×1.5–2 hoặc vẽ ở 768×1344 / 640×1536.
- Chân lỗi nặng: dùng **cowboy shot** hoặc thêm giày/tất (bàn chân có giày dễ hơn bàn chân trần rất nhiều).
- Sửa bàn chân: ADetailer không có model chân mặc định → **inpaint** vùng chân với `detailed feet, toes, barefoot` (hoặc `shoes`), denoise 0.45–0.55, *Chỉ vùng tô*, padding 64.

---

## 4. 📋 BỘ TỔNG HỢP (copy nhanh)

**Positive (thêm sau phần mô tả nhân vật):**
```
detailed eyes, looking at viewer, eye contact, detailed hands, five fingers, detailed feet, full body, standing, arms at sides
```
(thay `standing, arms at sides` bằng tư thế bạn muốn – quan trọng là **có** tư thế.)

**Negative – NoobAI chuẩn + mắt/tay/chân:**
```
worst quality, old, early, low quality, lowres, signature, username, logo, bad hands, mutated hands, mammal, anthro, furry,
ambiguous form, feral, semi-anthro, bad anatomy, bad proportions, bad perspective,
bad eyes, cross-eyed, uneven eyes, asymmetrical eyes, extra eyes, empty eyes,
extra digits, fewer digits, extra fingers, missing fingers, fused fingers, malformed hands, extra arms, missing arms,
bad feet, bad legs, extra legs, missing legs, fused toes, extra toes, disconnected limbs, extra limbs
```

**Negative gọn (khi prompt đã dài, ≤ 75 token):**
```
worst quality, low quality, lowres, bad anatomy, bad hands, bad feet, extra digits, fewer digits, extra limbs, cross-eyed, uneven eyes
```

---

## 5. 🤖 Prompt cho ADetailer (đã đặt sẵn trong bộ cài)

| Bộ | Model | Prompt | Negative |
|---|---|---|---|
| 1 – Mặt | `face_yolov8n.pt` | `detailed face, beautiful detailed eyes, symmetrical eyes, looking at viewer, eye contact` | `bad eyes, cross-eyed, uneven eyes, asymmetrical eyes, blurry, lowres` |
| 2 – Tay | `hand_yolov8n.pt` | `detailed hands, five fingers, natural hand pose, fingernails` | `bad hands, extra digits, fewer digits, fused fingers, mutated hands, extra fingers, missing fingers` |

Prompt ADetailer **không** chứa mô tả nhân vật (màu tóc, trang phục…) – ADetailer tự nối với prompt chính nếu để trống;
ở đây ta chỉ thêm phần mô tả bộ phận. Muốn giữ màu mắt chính xác: thêm `<màu> eyes` vào prompt bộ 1.

---

## 6. 🔧 Quy trình sửa khi vẫn lỗi
1. **Tư thế cụ thể** cho tay/chân trong prompt (mục 2, 3) → giảm 70 % lỗi.
2. **Hires fix** bật (mặc định) để bộ phận nhỏ có đủ pixel.
3. **ADetailer** mặt + tay (mặc định) → tự vẽ lại.
4. Vẫn lỗi → **Inpaint**: gallery ▸ *Gửi sang inpaint* ▸ tô vùng ▸ *Chỉ vùng tô*, padding 32–64, denoise 0.4–0.55, prompt chỉ mô tả bộ phận
   (`detailed hands, five fingers, holding cup`), tạo 4 ảnh (batch count 4) chọn cái đẹp nhất.
5. Lỗi vẫn còn → đổi seed / sampler DPM++ 2M Karras / CFG 5–6; hoặc dùng ControlNet OpenPose (bộ `noob_sdxl_controlnet_openpose`) để ép dáng tay chân.
