# -*- coding: utf-8 -*-
"""NoobAI Tools – script tích hợp cho Stable Diffusion WebUI Forge (được bộ cài NoobAI-XL chép vào <forge>/scripts/).

Tính năng:
  1. 📐 Dropdown "Độ phân giải chuẩn" ngay dưới Width/Height (txt2img + img2img) – các bucket ~1 MP mà
     NoobAI-XL / SDXL được huấn luyện; chọn là tự điền Width/Height. Hiển thị luôn kích thước sau Hires fix.
  2. ✨ "Tự fix": tự mở Hires. fix và ADetailer (sửa mặt/tay) khi mở UI, theo <forge>/noob_tools.json
     do noobai_lib.write_settings() ghi:  {"auto_hires_fix": true, "auto_adetailer": "face+hand"}
"""
import json
import os

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


def load_tool_config() -> dict:
    cfg = {"auto_hires_fix": True, "auto_adetailer": "face+hand"}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:  # noqa: BLE001
        pass
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

    def title(self):
        return "NoobAI Tools (độ phân giải chuẩn + tự fix)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    # ------------------------------------------------------------------ hooks
    def after_component(self, component, **kwargs):
        elem_id = kwargs.get("elem_id") or getattr(component, "elem_id", None) or ""

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
        if self.cfg.get("auto_adetailer", "off") != "off":
            if elem_id.endswith("adetailer_ad_main_accordion"):
                self._open_accordion(component)
            if elem_id.endswith("adetailer_ad_main_accordion-checkbox"):
                self._check(component)

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
