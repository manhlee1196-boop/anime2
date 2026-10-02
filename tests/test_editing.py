"""CPU tests: .venv/bin/python -m unittest discover -s tests -v

Pillow/numpy come with Gradio. No weights, CUDA, or public share link required.
"""
import json
import gc
import math
import secrets
import tempfile
import time
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from PIL import Image
import numpy as np


class DummyProgress:
    def __call__(self, *args, **kwargs):
        pass


class FakePipe:
    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return types.SimpleNamespace(images=[Image.new('RGB', (kwargs['width'], kwargs['height']), 'red')])


class FakeGenerator:
    def __init__(self, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


class EditingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        notebook = json.loads(Path('Animagine_XL_4_LPW_Colab.ipynb').read_text())
        source = ''.join(notebook['cells'][8]['source'])
        source = source[source.index('from PIL import Image, ImageFilter'):source.index('if "demo" in globals():')]
        self.pipe = FakePipe()
        self.ns = dict(
            gr=types.SimpleNamespace(Error=ValueError, Progress=DummyProgress),
            Path=Path, math=math, secrets=secrets, time=time, json=json, gc=gc,
            OUTPUT_DIR=Path(self.tmp.name), MODEL_ID='test', pipe=self.pipe,
            torch=types.SimpleNamespace(inference_mode=nullcontext, Generator=FakeGenerator,
                                        cuda=types.SimpleNamespace(OutOfMemoryError=MemoryError)),
            counts=lambda *args: [[3, 3], [1, 1]],
            uuid4=__import__('uuid').uuid4,
        )
        exec(source, self.ns)
        self.base = Image.new('RGB', (512, 384), 'blue')
        self.layer = Image.new('RGBA', self.base.size, (0, 0, 0, 0))
        self.layer.paste((255, 255, 255, 255), (190, 140, 230, 180))
        self.editor = dict(background=self.base, layers=[self.layer], composite=self.base)

    def test_mask_uses_brush_alpha_not_background(self):
        _, mask = self.ns['extract_mask'](self.editor)
        self.assertEqual(mask.getbbox(), (190, 140, 230, 180))
        self.assertEqual(mask.getpixel((0, 0)), 0)
        with self.assertRaisesRegex(ValueError, 'Chưa tô'):
            self.ns['extract_mask'](dict(background=self.base, layers=[]))

    def test_array_layer_and_overlapping_masks(self):
        self.editor['layers'] = [np.array(self.layer), np.array(self.layer)]
        _, mask = self.ns['extract_mask'](self.editor)
        self.assertEqual(mask.getbbox(), (190, 140, 230, 180))

    def test_crop_and_merge_preserve_pixels_outside_mask(self):
        image, mask = self.ns['extract_mask'](self.editor)
        box = self.ns['repair_box'](image, mask, True, 96)
        self.assertGreaterEqual(box[2]-box[0], 256)
        self.assertGreaterEqual(box[0], 0)
        self.assertLessEqual(box[2], image.width)
        result = self.ns['merge_repair'](image, Image.new('RGB', (1024, 1024), 'red'), mask, box, 0)
        self.assertEqual(result.getpixel((200, 150)), (255, 0, 0))
        outside = np.asarray(mask) == 0
        np.testing.assert_array_equal(np.asarray(result)[outside], np.asarray(image)[outside])

    def test_repair_callback_saves_and_passes_mask_and_strength(self):
        before, path, png, meta, undo, status = self.ns['repair'](
            self.editor, 'hand', 'bad hands', .55, 28, 5, 123, True, 96, 0)
        self.assertEqual(path, png)
        with Image.open(path) as saved:
            self.assertEqual(saved.size, self.base.size)
        self.assertEqual(self.pipe.kwargs['strength'], .55)
        self.assertIsNotNone(self.pipe.kwargs['mask_image'])
        self.assertEqual(self.pipe.kwargs['generator'].seed, 123)
        self.assertEqual(json.loads(Path(meta).read_text())['seed'], 123)
        restored = self.ns['undo_repair'](undo)
        self.assertIsNone(restored[2])
        self.assertIsNone(restored[5])
        np.testing.assert_array_equal(np.asarray(restored[0]['background']), np.asarray(self.base))
        self.assertEqual(self.ns['use_repaired'](path)['layers'], [])

    def test_upscale_and_limits(self):
        path, _, meta, status = self.ns['upscale'](self.base, 2, False)
        with Image.open(path) as saved:
            self.assertEqual(saved.size, (1024, 768))
        self.assertIn('không phải AI', json.loads(Path(meta).read_text())['mode'])
        with self.assertRaises(ValueError): self.ns['upscale'](self.base, 3, False)
        for seed in (float('nan'), float('inf'), -2, 1.2):
            with self.assertRaises(ValueError): self.ns['validated_seed'](seed)
        with self.assertRaises(ValueError): self.ns['open_image'](None)

class AIUpscaleTests(EditingTests):
    """Network-free CPU tests; random weights do NOT test visual quality."""
    def test_rrdb_architecture_matches_checkpoint_key_layout(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch needed for architecture test')
        torch.set_num_threads(1)
        model = self.ns['build_anime6b']()
        self.assertEqual(len(model.body), 6)
        state = model.state_dict()
        self.assertEqual(state['conv_first.weight'].shape, (64,3,3,3))
        self.assertEqual(state['body.5.rdb3.conv5.weight'].shape, (64,192,3,3))
        self.assertEqual(state['conv_last.weight'].shape, (3,64,3,3))
        with torch.inference_mode():
            self.assertEqual(model(torch.zeros(1,3,5,7)).shape, (1,3,20,28))
        # Exercise safe checkpoint loading and lazy CPU caching with random weights.
        path = Path(self.tmp.name)/'mock_model.pth'
        torch.save({'params_ema':state},path)
        self.ns['download_ai_weights'] = lambda: path
        loaded = self.ns['load_ai_model'](DummyProgress())
        self.assertFalse(loaded.training)
        self.assertIs(loaded,self.ns['load_ai_model'](DummyProgress()))

    def test_tiling_covers_non_divisible_edges_and_tiny_images(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch needed for tile test')
        class Nearest4x(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))
            def forward(self,x):
                return torch.nn.functional.interpolate(x,scale_factor=4,mode='nearest')
        rng = np.random.default_rng(42)
        for w,h in ((143,81),(1,3),(3,1),(11,9)):
            image = Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8))
            result = self.ns['tiled_anime4x'](image,Nearest4x(),64,DummyProgress())
            self.assertEqual(result.size,(w*4,h*4))
            np.testing.assert_array_equal(np.asarray(result),np.asarray(image.resize((w*4,h*4),Image.Resampling.NEAREST)))

    def test_ai_callback_native_x4_and_resize_metadata(self):
        called = []
        def fake_ai(image,tile,progress):
            called.append(tile)
            return image.resize((image.width*4,image.height*4))
        self.ns['run_ai_upscale']=fake_ai
        path,_,meta,_=self.ns['upscale'](self.base,1.5,False,self.ns['AI_MODE'],64)
        with Image.open(path) as image: self.assertEqual(image.size,(768,576))
        data=json.loads(Path(meta).read_text())
        self.assertEqual(data['native_scale'],4)
        self.assertEqual(data['tile'],64)
        self.assertEqual(called,[64])
        with self.assertRaisesRegex(ValueError,'Tile'):
            self.ns['upscale'](self.base,2,False,self.ns['AI_MODE'],8)
        with self.assertRaisesRegex(ValueError,'4 megapixel'):
            self.ns['upscale'](Image.new('RGB',(2100,2000)),1.5,False,self.ns['AI_MODE'],64)
        with self.assertRaisesRegex(ValueError,'16 megapixel'):
            self.ns['upscale'](Image.new('RGB',(1100,1000)),4,False,self.ns['AI_MODE'],64)

    def test_download_failure_removes_partial_file(self):
        from unittest.mock import patch
        class BrokenResponse:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self,*args): raise OSError('network interrupted')
        path=Path(self.tmp.name)/'weights.pth'
        with patch('urllib.request.urlopen',return_value=BrokenResponse()):
            with self.assertRaises(OSError): self.ns['download_ai_weights'](path)
        self.assertFalse(path.exists())
        self.assertFalse(path.with_suffix('.download').exists())

    def test_ai_releases_gpu_on_failure(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch needed for GPU lifecycle mock')
        from unittest.mock import Mock, patch
        model = Mock()
        modules = {name: Mock() for name in ('text_encoder','text_encoder_2','unet','vae')}
        self.ns['pipe'] = types.SimpleNamespace(**modules)
        self.ns['load_ai_model'] = lambda progress: model
        def fail(*args): raise torch.cuda.OutOfMemoryError('test OOM')
        self.ns['tiled_anime4x'] = fail
        with patch('torch.cuda.is_available',return_value=True), patch('torch.cuda.empty_cache'):
            with self.assertRaisesRegex(ValueError,'Hết VRAM'):
                self.ns['run_ai_upscale'](self.base,64,DummyProgress())
        for component in modules.values(): component.to.assert_called_once_with('cpu')
        self.assertEqual(model.to.call_args_list[-1].kwargs,dict(device='cpu',dtype=torch.float32))


if __name__ == '__main__':
    unittest.main()
