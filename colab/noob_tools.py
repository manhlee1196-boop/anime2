# -*- coding: utf-8 -*-
"""NoobAI Tools – script tích hợp cho Stable Diffusion WebUI Forge (được bộ cài NoobAI-XL chép vào <forge>/scripts/).

Tính năng:
  1. 📐 Dropdown "Độ phân giải chuẩn" ngay dưới Width/Height (txt2img + img2img) – các bucket ~1 MP mà
     NoobAI-XL / SDXL được huấn luyện; chọn là tự điền Width/Height. Hiển thị luôn kích thước sau Hires fix.
  2. ✨ "Tự fix": tự mở Hires. fix và ADetailer (sửa mặt/tay) khi mở UI, theo <forge>/noob_tools.json
     do noobai_lib.write_settings() ghi:  {"auto_hires_fix": true, "auto_adetailer": "face+hand", "param_mode": "recommended"}
  3. ⚙️ Dropdown "Thông số khuyên dùng / tự chỉnh" phía trên Sampling method: một cú chọn đặt toàn bộ
     sampler, scheduler, steps, CFG, kích thước, batch, Hires fix (bật/tắt, upscaler, ×, steps, denoise, CFG);
     sửa tay bất kỳ thông số nào → tự chuyển sang "Tuỳ chỉnh".
  4. 🩹 Dropdown "Tự sửa mặt / mắt / tay (ADetailer)": mức Tắt / Nhẹ / Chuẩn / Mạnh / Nhiều người – đặt model,
     prompt, confidence, dilation, blur, denoise, padding cho 3 bộ ADetailer (mặt, mắt, tay) trong một cú chọn.
     Bảng mức lấy từ noob_tools.json ("ad_levels", do noobai_lib.AD_LEVELS ghi); có bảng dự phòng trong file này.
"""
import json
import os
import re

import gradio as gr

from modules import scripts

try:
    from modules.paths_internal import data_path
except Exception:  # noqa: BLE001
    data_path = os.getcwd()

CONFIG_FILE = os.path.join(data_path, "noob_tools.json")
CUSTOM = "Tuỳ chỉnh (tự kéo Width/Height)"

# (nhãn, rộng, cao) – bucket chuẩn SDXL ~1 megapixel + vài cỡ lớn hơn cho GPU khoẻ
PRESETS = [
    ("Dọc 832×1216 · 2:3 (chuẩn NoobAI, khuyên dùng)", 832, 1216),
    ("Ngang 1216×832 · 3:2", 1216, 832),
    ("Vuông 1024×1024 · 1:1", 1024, 1024),
    ("Dọc 896×1152 · 7:9", 896, 1152),
    ("Ngang 1152×896 · 9:7", 1152, 896),
    ("Dọc 768×1344 · 9:16 (điện thoại)", 768, 1344),
    ("Ngang 1344×768 · 16:9 (màn hình)", 1344, 768),
    ("Dọc 640×1536 · 5:12 (rất cao)", 640, 1536),
    ("Ngang 1536×640 · 12:5 (banner)", 1536, 640),
    ("Dọc 1024×1536 · 2:3 lớn (chậm hơn, GPU ≥ 12 GB)", 1024, 1536),
    ("Ngang 1536×1024 · 3:2 lớn (chậm hơn, GPU ≥ 12 GB)", 1536, 1024),
]
_BY_LABEL = {label: (w, h) for label, w, h in PRESETS}
_BY_SIZE = {(w, h): label for label, w, h in PRESETS}


CUSTOM_P = "✏️ Tuỳ chỉnh (tự đặt từng thông số)"

# Bộ thông số cho txt2img. None = giữ nguyên giá trị hiện tại.
TXT2IMG_PRESETS = {
    "✅ Khuyên dùng – NoobAI chuẩn (cân bằng)": dict(
        sampler="Euler a", scheduler="Automatic", steps=28, cfg=6.0, width=832, height=1216,
        batch_count=1, batch_size=1,
        hr=True, hr_upscaler="4x-AnimeSharp", hr_scale=1.5, hr_steps=14, denoise=0.4, hr_cfg=6.0),
    "⚡ Nháp nhanh – tìm ý tưởng/seed (không Hires)": dict(
        sampler="Euler a", scheduler="Automatic", steps=20, cfg=5.5, hr=False),
    "💎 Chất lượng cao – chậm hơn ~2× (Hires ×2)": dict(
        sampler="DPM++ 2M SDE", scheduler="Karras", steps=32, cfg=6.5,
        hr=True, hr_upscaler="4x-AnimeSharp", hr_scale=2.0, hr_steps=16, denoise=0.35, hr_cfg=6.5),
    "🎨 Sáng tạo – bám prompt lỏng, đa dạng hơn": dict(
        sampler="Euler a", scheduler="Automatic", steps=30, cfg=4.5,
        hr=True, hr_upscaler="4x-AnimeSharp", hr_scale=1.5, hr_steps=14, denoise=0.45, hr_cfg=4.5),
    "🎯 Chính xác – bám prompt chặt, ít ngẫu nhiên": dict(
        sampler="DPM++ 2M", scheduler="Karras", steps=30, cfg=7.5,
        hr=True, hr_upscaler="4x-AnimeSharp", hr_scale=1.5, hr_steps=15, denoise=0.3, hr_cfg=7.5),
    CUSTOM_P: None,
}

# Bộ thông số cho img2img
IMG2IMG_PRESETS = {
    "✅ Khuyên dùng – chỉnh vừa phải (denoise 0.5)": dict(sampler="Euler a", scheduler="Automatic", steps=28, cfg=6.0, denoise=0.5),
    "🪶 Nhẹ – giữ gần như nguyên ảnh (denoise 0.3)": dict(sampler="Euler a", scheduler="Automatic", steps=28, cfg=6.0, denoise=0.3),
    "🔥 Mạnh – vẽ lại nhiều theo prompt (denoise 0.7)": dict(sampler="Euler a", scheduler="Automatic", steps=30, cfg=6.5, denoise=0.7),
    "🔍 Phóng to (Ultimate SD Upscale, denoise 0.3)": dict(sampler="DPM++ 2M", scheduler="Karras", steps=24, cfg=6.0, denoise=0.3),
    CUSTOM_P: None,
}

# ---------------------------------------------------------------- 🩹 ADetailer
AD_CUSTOM = "✏️ Tuỳ chỉnh (tự đặt trong mục ADetailer)"
_AD_P = "[PROMPT], "
_AD_FACE = dict(models=["face_yolov8s.pt", "face_yolov8n.pt"],
                prompt=_AD_P + "detailed face, beautiful detailed eyes, symmetrical eyes, eye contact",
                negative=_AD_P + "bad eyes, cross-eyed, uneven eyes, asymmetrical eyes, blurry, lowres",
                confidence=0.3, dilate=4, blur=8, denoise=0.4, padding=32)
_AD_EYES = dict(models=["Anzhc_Eyes_seg_hd.pt", "mediapipe_face_mesh_eyes_only"],
                prompt=_AD_P + "detailed eyes, beautiful detailed eyes, eye focus, bright pupils, eye reflection, sparkling eyes",
                negative=_AD_P + "bad eyes, cross-eyed, uneven eyes, asymmetrical eyes, extra pupils, empty eyes, blurry eyes",
                confidence=0.3, dilate=8, blur=4, denoise=0.35, padding=32)
_AD_HAND = dict(models=["hand_yolov8s.pt", "hand_yolov8n.pt"],
                prompt=_AD_P + "detailed hands, five fingers, natural hand pose, fingernails",
                negative=_AD_P + "bad hands, extra digits, fewer digits, fused fingers, mutated hands, extra fingers, missing fingers, too many fingers",
                confidence=0.25, dilate=12, blur=8, denoise=0.5, padding=48)
AD_LEVELS_FALLBACK = {
    "off": dict(label="⛔ Tắt – không tự sửa", enable=False, tabs=[]),
    "light": dict(label="🪶 Nhẹ – chỉ sửa mặt (nhanh nhất)", enable=True, tabs=[{**_AD_FACE, "denoise": 0.35}]),
    "standard": dict(label="✅ Chuẩn – mặt + mắt + tay (khuyên dùng)", enable=True, tabs=[_AD_FACE, _AD_EYES, _AD_HAND]),
    "strong": dict(label="🔥 Mạnh – mắt/tay lỗi nặng, vẽ lại nhiều hơn", enable=True,
                   tabs=[{**_AD_FACE, "denoise": 0.45, "dilate": 8}, {**_AD_EYES, "denoise": 0.45, "dilate": 12},
                         {**_AD_HAND, "denoise": 0.6, "dilate": 16, "padding": 64, "confidence": 0.2}]),
    "multi": dict(label="👥 Nhiều người – nhận diện nhạy hơn, sửa mặt + mắt + tay", enable=True,
                  tabs=[{**_AD_FACE, "confidence": 0.2}, {**_AD_EYES, "confidence": 0.2}, {**_AD_HAND, "confidence": 0.15}]),
}
# khoá trong tab preset → hậu tố elem_id của ADetailer (script_<tab>_adetailer_<id>[_2nd|_3rd])
AD_FIELD_IDS = {
    "enable_tab": "ad_tab_enable", "model": "ad_model", "prompt": "ad_prompt", "negative": "ad_negative_prompt",
    "confidence": "ad_confidence", "dilate": "ad_dilate_erode", "blur": "ad_mask_blur",
    "denoise": "ad_denoising_strength", "padding": "ad_inpaint_only_masked_padding",
}
_AD_ID_RE = re.compile(r"^script_(txt2img|img2img)_adetailer_(ad_[a-z_]+?)(?:_(\d+)(?:st|nd|rd|th))?$")
_AD_DEFAULT_TAB = dict(models=["None"], prompt="", negative="", confidence=0.3, dilate=4, blur=4, denoise=0.4, padding=32)


def _ad_summary(level: dict | None) -> str:
    if not level:
        return "<div style='font-size:0.85em;opacity:0.8'>Tự chỉnh trong mục <b>ADetailer</b> bên dưới. Chọn lại một mức để khôi phục.</div>"
    if not level.get("enable"):
        return "<div style='font-size:0.85em;opacity:0.8'>ADetailer tắt – ảnh không được tự vẽ lại mặt/mắt/tay.</div>"
    names = ["mặt", "mắt", "tay"]
    parts = []
    for i, t in enumerate(level.get("tabs", [])[:3]):
        parts.append(f"bộ {i + 1} {names[i] if i < 3 else ''}: {t['models'][0]} · denoise {t['denoise']:g} · conf {t['confidence']:g}")
    return "<div style='font-size:0.85em;opacity:0.8'>Đã đặt: " + " | ".join(parts) + "</div>"


# elem_id của các thông số theo tab → khoá trong preset
PARAM_IDS = {
    "sampler": "{tab}_sampling", "scheduler": "{tab}_scheduler", "steps": "{tab}_steps", "cfg": "{tab}_cfg_scale",
    "width": "{tab}_width", "height": "{tab}_height", "batch_count": "{tab}_batch_count", "batch_size": "{tab}_batch_size",
    "denoise": "{tab}_denoising_strength",
    # chỉ txt2img
    "hr": "txt2img_hr-checkbox", "hr_upscaler": "txt2img_hr_upscaler", "hr_scale": "txt2img_hr_scale",
    "hr_steps": "txt2img_hires_steps", "hr_cfg": "txt2img_hr_cfg",
}
PARAM_ORDER = list(PARAM_IDS)


def _choices(comp) -> list:
    out = []
    for c in getattr(comp, "choices", None) or []:
        out.append(c[0] if isinstance(c, (list, tuple)) else c)
    return out


def _summary(preset: dict | None, tab: str) -> str:
    if not preset:
        return "<div style='font-size:0.85em;opacity:0.8'>Tự đặt từng thông số bên dưới. Chọn lại một bộ khuyên dùng để khôi phục.</div>"
    parts = []
    if "sampler" in preset:
        parts.append(f"{preset['sampler']} · {preset.get('scheduler', 'Automatic')}")
    if "steps" in preset:
        parts.append(f"{preset['steps']} steps")
    if "cfg" in preset:
        parts.append(f"CFG {preset['cfg']:g}")
    if "width" in preset:
        parts.append(f"{preset['width']}×{preset['height']}")
    if "denoise" in preset and tab == "img2img":
        parts.append(f"denoise {preset['denoise']:g}")
    if tab == "txt2img":
        if preset.get("hr"):
            parts.append(f"Hires fix {preset.get('hr_upscaler', '')} ×{preset.get('hr_scale', 1.5):g}, "
                         f"{preset.get('hr_steps', 14)} steps, denoise {preset.get('denoise', 0.4):g}")
        elif "hr" in preset:
            parts.append("Hires fix tắt")
    return "<div style='font-size:0.85em;opacity:0.8'>Đã đặt: " + " · ".join(parts) + "</div>"


def load_tool_config() -> dict:
    cfg = {"auto_hires_fix": True, "auto_adetailer": "standard", "fix_level": None, "param_mode": "recommended",
           "ad_levels": None}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:  # noqa: BLE001
        pass
    if not isinstance(cfg.get("ad_levels"), dict) or not cfg["ad_levels"]:
        cfg["ad_levels"] = AD_LEVELS_FALLBACK
    if not cfg.get("fix_level"):
        legacy = str(cfg.get("auto_adetailer", "standard")).lower()
        cfg["fix_level"] = {"face+hand": "standard", "face": "light", "off": "off"}.get(legacy, legacy)
    if cfg["fix_level"] not in cfg["ad_levels"]:
        cfg["fix_level"] = "standard" if "standard" in cfg["ad_levels"] else next(iter(cfg["ad_levels"]))
    cfg["auto_adetailer"] = cfg["fix_level"]
    return cfg


def _ui_config_defaults(tab: str, w_fallback: int, h_fallback: int) -> tuple[int, int, float]:
    """Width/Height/Upscale by mà Forge sẽ áp SAU khi component được tạo:
    ui-config.json, rồi preset Forge (config.json: forge_preset=xl → xl_t2i_width/xl_t2i_height…) áp lúc tải trang."""
    w, h, scale = w_fallback, h_fallback, 1.5
    try:
        with open(os.path.join(data_path, "ui-config.json"), "r", encoding="utf-8") as f:
            ui = json.load(f)
        w = int(ui.get(f"{tab}/Width/value", w) or w)
        h = int(ui.get(f"{tab}/Height/value", h) or h)
        scale = float(ui.get("txt2img/Upscale by/value", scale) or scale)
    except Exception:  # noqa: BLE001
        pass
    try:
        with open(os.path.join(data_path, "config.json"), "r", encoding="utf-8") as f:
            opts = json.load(f)
        preset = opts.get("forge_preset", "xl")
        kind = "t2i" if tab == "txt2img" else "i2i"
        if preset in ("sd", "xl", "flux"):
            w = int(opts.get(f"{preset}_{kind}_width", w) or w)
            h = int(opts.get(f"{preset}_{kind}_height", h) or h)
    except Exception:  # noqa: BLE001
        pass
    return w, h, scale


def _hint(w: int, h: int, scale: float) -> str:
    if not w or not h:
        return ""
    hw, hh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
    mp = w * h / 1e6
    return (f"<div style='font-size:0.85em;opacity:0.8'>Ảnh gốc <b>{w}×{h}</b> ({mp:.2f} MP) → sau Hires fix ×{scale:g}: "
            f"<b>{hw}×{hh}</b>. Tăng thêm nữa: dùng <i>Extras</i> hoặc <i>Ultimate SD Upscale</i> (xem hướng dẫn trong notebook).</div>")


class NoobTools(scripts.Script):
    def __init__(self):
        super().__init__()
        self.cfg = load_tool_config()
        self.width = {}
        self.height = {}
        self.params = {"txt2img": {}, "img2img": {}}   # tab -> khoá preset -> component
        self.preset_dd = {}
        self.preset_hint = {}
        # 🩹 ADetailer: tab -> {"enable": checkbox, "tabs": {1: {field: component}, 2: {...}, 3: {...}}}
        self.ad = {"txt2img": {"enable": None, "tabs": {}}, "img2img": {"enable": None, "tabs": {}}}
        self.ad_dd = {}
        self.ad_hint = {}
        self.ad_wired = set()

    def title(self):
        return "NoobAI Tools (độ phân giải chuẩn + tự fix)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # Lúc này toàn bộ thông số của tab đã được tạo → nối sự kiện cho dropdown ⚙️
        tab = "img2img" if is_img2img else "txt2img"
        try:
            self._wire_presets(tab)
        except Exception as e:  # noqa: BLE001
            print(f"[NoobAI Tools] không nối được preset thông số ({tab}): {e}")
        return []

    # ------------------------------------------------------------------ hooks
    def before_component(self, component, **kwargs):
        elem_id = kwargs.get("elem_id") or ""
        if elem_id in ("sampler_selection_txt2img", "sampler_selection_img2img"):
            self._build_preset_dropdown(elem_id.rsplit("_", 1)[1])

    def after_component(self, component, **kwargs):
        elem_id = kwargs.get("elem_id") or getattr(component, "elem_id", None) or ""

        for tab in ("txt2img", "img2img"):
            for key, pattern in PARAM_IDS.items():
                if elem_id == pattern.format(tab=tab):
                    self.params[tab][key] = component

        if elem_id in ("txt2img_width", "img2img_width"):
            self.width[elem_id.split("_")[0]] = component
        elif elem_id in ("txt2img_height", "img2img_height"):
            tab = elem_id.split("_")[0]
            self.height[tab] = component
            self._build_preset_ui(tab)

        # ✨ Tự fix: mở sẵn Hires. fix (txt2img) và ADetailer khi vào UI
        if self.cfg.get("auto_hires_fix") and elem_id == "txt2img_hr":
            self._open_accordion(component)
        if self.cfg.get("auto_hires_fix") and elem_id == "txt2img_hr-checkbox":
            self._check(component)
        ad_on = self.cfg.get("fix_level", "off") != "off"
        if elem_id.endswith("adetailer_ad_main_accordion") and ad_on:
            self._open_accordion(component)
        if elem_id.endswith("adetailer_ad_main_accordion-checkbox"):
            if ad_on:
                self._check(component)
            for tab in ("txt2img", "img2img"):
                if elem_id.startswith(f"script_{tab}_"):
                    self.ad[tab]["enable"] = component

        # 🩹 gom các thông số ADetailer (3 bộ) để dropdown đặt một lượt
        m = _AD_ID_RE.match(elem_id)
        if m:
            tab, field, idx = m.group(1), m.group(2), int(m.group(3) or 1)
            self.ad[tab]["tabs"].setdefault(idx, {})[field] = component

        # gallery được tạo SAU toàn bộ script (kể cả ADetailer) → lúc này mới nối dropdown 🩹
        if elem_id in ("txt2img_gallery", "img2img_gallery"):
            tab = elem_id.split("_")[0]
            try:
                self._wire_adetailer(tab)
            except Exception as e:  # noqa: BLE001
                print(f"[NoobAI Tools] không nối được mức sửa lỗi ADetailer ({tab}): {e}")

    @staticmethod
    def _open_accordion(acc):
        try:
            acc.open = True
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _check(chk):
        try:
            chk.value = True
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ UI
    def _build_preset_ui(self, tab: str):
        w, h = self.width.get(tab), self.height.get(tab)
        if w is None or h is None:
            return
        w0, h0, scale = _ui_config_defaults(tab, int(w.value or 0), int(h.value or 0))
        cur = _BY_SIZE.get((w0, h0), CUSTOM)
        choices = [label for label, _, _ in PRESETS] + [CUSTOM]

        dd = gr.Dropdown(label="📐 Độ phân giải chuẩn (NoobAI-XL / SDXL) – chọn là tự điền Width/Height",
                         choices=choices, value=cur, elem_id=f"{tab}_noob_res_preset", interactive=True)
        dd.do_not_save_to_config = True
        hint = gr.HTML(value=_hint(w0, h0, scale) if tab == "txt2img" else "", elem_id=f"{tab}_noob_res_hint")
        hint.do_not_save_to_config = True

        def apply_preset(label):
            if label in _BY_LABEL:
                pw, ph = _BY_LABEL[label]
                return gr.update(value=pw), gr.update(value=ph)
            return gr.update(), gr.update()

        def sync_dropdown(cw, ch):
            label = _BY_SIZE.get((int(cw), int(ch)), CUSTOM)
            return gr.update(value=label), (_hint(int(cw), int(ch), scale) if tab == "txt2img" else "")

        # .input = chỉ khi người dùng chọn (tránh vòng lặp với cập nhật tự động)
        dd.input(fn=apply_preset, inputs=[dd], outputs=[w, h], show_progress=False)
        w.change(fn=sync_dropdown, inputs=[w, h], outputs=[dd, hint], show_progress=False)
        h.change(fn=sync_dropdown, inputs=[w, h], outputs=[dd, hint], show_progress=False)

    # ------------------------------------------------------------------ ⚙️ bộ thông số
    def _build_preset_dropdown(self, tab: str):
        presets = TXT2IMG_PRESETS if tab == "txt2img" else IMG2IMG_PRESETS
        names = list(presets)
        initial = names[0] if str(self.cfg.get("param_mode", "recommended")).startswith("rec") else CUSTOM_P
        with gr.Row(elem_id=f"{tab}_noob_param_row"):
            dd = gr.Dropdown(label="⚙️ Thông số tạo ảnh: chọn bộ khuyên dùng hoặc tự chỉnh",
                             choices=names, value=initial, elem_id=f"{tab}_noob_param_preset", interactive=True, scale=3)
            dd.do_not_save_to_config = True
            hint = gr.HTML(value=_summary(presets[initial], tab), elem_id=f"{tab}_noob_param_hint")
            hint.do_not_save_to_config = True
        self.preset_dd[tab] = dd
        self.preset_hint[tab] = hint

        # 🩹 mức tự sửa mặt / mắt / tay (ADetailer)
        levels = self.cfg["ad_levels"]
        labels = [lv["label"] for lv in levels.values()] + [AD_CUSTOM]
        cur_key = self.cfg.get("fix_level", "standard")
        cur = levels[cur_key]["label"] if cur_key in levels else AD_CUSTOM
        with gr.Row(elem_id=f"{tab}_noob_fix_row"):
            add = gr.Dropdown(label="🩹 Tự sửa mặt / mắt / tay sau khi tạo ảnh (ADetailer)",
                              choices=labels, value=cur, elem_id=f"{tab}_noob_fix_level", interactive=True, scale=3)
            add.do_not_save_to_config = True
            ahint = gr.HTML(value=_ad_summary(levels.get(cur_key)), elem_id=f"{tab}_noob_fix_hint")
            ahint.do_not_save_to_config = True
        self.ad_dd[tab] = add
        self.ad_hint[tab] = ahint

    def _wire_adetailer(self, tab: str):
        if tab in self.ad_wired:
            return
        dd, hint = self.ad_dd.get(tab), self.ad_hint.get(tab)
        info = self.ad.get(tab) or {}
        enable, tabs = info.get("enable"), info.get("tabs") or {}
        if dd is None or not tabs:
            if dd is not None:
                hint.value = "<div style='font-size:0.85em;opacity:0.8'>Không tìm thấy ADetailer – mức sửa lỗi không hoạt động.</div>"
            return
        self.ad_wired.add(tab)
        levels = self.cfg["ad_levels"]
        by_label = {lv["label"]: lv for lv in levels.values()}

        outputs, slots = [], []       # slots: (n, field) tương ứng từng output; n=0 → checkbox bật ADetailer
        if enable is not None:
            outputs.append(enable)
            slots.append((0, "enable"))
        for n in sorted(tabs):
            for key, fid in AD_FIELD_IDS.items():
                comp = tabs[n].get(fid)
                if comp is not None:
                    outputs.append(comp)
                    slots.append((n, key))

        def apply(label):
            lv = by_label.get(label)
            if lv is None:
                return [gr.update() for _ in outputs] + [_ad_summary(None)]
            updates = []
            for n, key in slots:
                if n == 0:
                    updates.append(gr.update(value=bool(lv["enable"])))
                    continue
                t = lv["tabs"][n - 1] if n - 1 < len(lv["tabs"]) else _AD_DEFAULT_TAB
                if key == "enable_tab":
                    updates.append(gr.update(value=True))
                elif key == "model":
                    comp = tabs[n][AD_FIELD_IDS["model"]]
                    choices = _choices(comp)
                    val = next((mdl for mdl in t["models"] if not choices or mdl in choices), "None")
                    updates.append(gr.update(value=val))
                else:
                    updates.append(gr.update(value=t[key]))
            return updates + [_ad_summary(lv)]

        dd.input(fn=apply, inputs=[dd], outputs=outputs + [hint], show_progress=False)

        def to_custom():
            return gr.update(value=AD_CUSTOM), _ad_summary(None)

        for comp in outputs:
            try:
                comp.input(fn=to_custom, inputs=[], outputs=[dd, hint], show_progress=False)
            except Exception:  # noqa: BLE001
                pass

    def _wire_presets(self, tab: str):
        dd, hint = self.preset_dd.get(tab), self.preset_hint.get(tab)
        comps = self.params.get(tab, {})
        if dd is None or not comps:
            return
        presets = TXT2IMG_PRESETS if tab == "txt2img" else IMG2IMG_PRESETS
        keys = [k for k in PARAM_ORDER if k in comps]
        outputs = [comps[k] for k in keys]

        def apply(name):
            preset = presets.get(name)
            applied = dict(preset) if preset else None
            updates = []
            for k in keys:
                if not preset or k not in preset:
                    updates.append(gr.update())
                    continue
                val = preset[k]
                comp = comps[k]
                if k in ("sampler", "scheduler", "hr_upscaler"):
                    choices = _choices(comp)
                    if choices and val not in choices:
                        fallback = {"sampler": ["Euler a", "Euler", "DPM++ 2M"],
                                    "scheduler": ["Automatic", "Karras"],
                                    "hr_upscaler": ["4x-AnimeSharp", "4x-UltraSharp", "R-ESRGAN 4x+ Anime6B", "Latent"]}[k]
                        val = next((f for f in fallback if f in choices), choices[0])
                        applied[k] = val
                updates.append(gr.update(value=val))
            return updates + [_summary(applied, tab)]

        dd.input(fn=apply, inputs=[dd], outputs=outputs + [hint], show_progress=False)

        # người dùng sửa tay bất kỳ thông số nào → chuyển dropdown sang "Tuỳ chỉnh"
        def to_custom():
            return gr.update(value=CUSTOM_P), _summary(None, tab)

        for k in keys:
            try:
                comps[k].input(fn=to_custom, inputs=[], outputs=[dd, hint], show_progress=False)
            except Exception:  # noqa: BLE001
                pass
