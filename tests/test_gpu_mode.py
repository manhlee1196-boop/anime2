"""Max-GPU mode: pipeline resident on the GPU, offload only as user fallback."""
import json
import unittest
from pathlib import Path

NOTEBOOK = json.loads(Path('Animagine_XL_4_LPW_Colab.ipynb').read_text())
MODEL_CELL = ''.join(NOTEBOOK['cells'][4]['source'])
MODEL_MARKDOWN = ''.join(NOTEBOOK['cells'][3]['source'])
UI_CELL = ''.join(NOTEBOOK['cells'][8]['source'])


class GpuModeTests(unittest.TestCase):
    def test_pipeline_resident_on_gpu(self):
        self.assertIn('pipe.to("cuda")', MODEL_CELL)
        self.assertIn('torch.backends.cudnn.benchmark = True', MODEL_CELL)
        # Chỉ nói về offload trong chú thích gợi ý dự phòng, không gọi trong mã chính.
        code = '\n'.join(line for line in MODEL_CELL.splitlines()
                         if not line.lstrip().startswith('#'))
        self.assertNotIn('pipe.enable_model_cpu_offload()', code)
        self.assertNotIn('pipe.enable_vae_tiling()', code)

    def test_model_cell_reports_vram(self):
        self.assertIn('torch.cuda.mem_get_info()', MODEL_CELL)

    def test_markdown_documents_offload_fallback(self):
        self.assertIn('enable_model_cpu_offload()', MODEL_MARKDOWN)
        self.assertIn('enable_vae_tiling()', MODEL_MARKDOWN)
        self.assertIn('hết vram', MODEL_MARKDOWN.lower())

    def test_generate_and_repair_oom_errors_suggest_offload(self):
        for message in (UI_CELL.count('pipe.enable_model_cpu_offload(); pipe.enable_vae_tiling()'),):
            self.assertGreaterEqual(message, 2)

    def test_ai_upscale_restores_origin_device(self):
        start = UI_CELL.index('def run_ai_upscale')
        end = UI_CELL.index('def upscale', start)
        body = UI_CELL[start:end]
        self.assertIn('def origin_of', body)
        self.assertIn('origins[name] = origin_of(component)', body)
        self.assertIn("component.to(device=device,dtype=dtype)", body)


if __name__ == '__main__':
    unittest.main()
