"""Replay the diffusers 0.35.1 custom-pipeline revision guard offline.

Mirrors how DiffusionPipeline.from_pretrained(custom_pipeline="lpw_stable_diffusion_xl",
custom_revision=...) resolves community pipeline code: the bare pipeline name goes to
get_cached_module_file, which accepts bare versions like "0.35.1" and rejects "v0.35.1"
with the ValueError users saw on Colab.
"""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from diffusers.utils import dynamic_modules_utils as dm
except ImportError:  # pragma: no cover
    dm = None

PIPELINE_NAME = "lpw_stable_diffusion_xl"
MODULE_FILE = "pipeline.py"
VERSIONS = ["0.27.2", "0.34.0", "0.35.0", "0.35.1", "0.36.0", "0.40.0"]
DUMMY_PIPELINE = "class LPWStableDiffusionXLPipeline:\n    pass\n"


@unittest.skipIf(dm is None, 'diffusers needed to exercise the real guard')
class DiffusersRevisionTests(unittest.TestCase):
    def resolve(self, revision):
        with tempfile.TemporaryDirectory() as cache:
            remote = Path(cache) / 'mirror'
            remote.mkdir()
            source = remote / 'downloaded.py'
            source.write_text(DUMMY_PIPELINE)
            downloads = []

            def fake_download(**kwargs):
                downloads.append(kwargs)
                return str(source)

            with patch.object(dm, 'get_diffusers_versions', return_value=VERSIONS), \
                 patch.object(dm, 'hf_hub_download', side_effect=fake_download), \
                 patch.object(dm, 'HF_MODULES_CACHE', str(Path(cache) / 'modules')), \
                 patch.object(dm, 'model_info'):
                result = dm.get_cached_module_file(
                    PIPELINE_NAME, MODULE_FILE, revision=revision)
            # Assert cache contents before the TemporaryDirectory disappears.
            copied = (Path(cache) / 'modules' / 'diffusers_modules' / 'git'
                      / 'lpw_stable_diffusion_xl.py').exists()
            return result, downloads, copied

    def test_bare_revision_downloads_from_mirror(self):
        result, downloads, copied = self.resolve('0.35.1')
        # The guard accepts bare versions, then rewrites them to the mirror's v-prefixed folder.
        self.assertEqual(downloads[0]['filename'], 'v0.35.1/lpw_stable_diffusion_xl.py')
        self.assertEqual(downloads[0]['repo_id'], 'diffusers/community-pipelines-mirror')
        self.assertEqual(downloads[0]['repo_type'], 'dataset')
        self.assertEqual(len(downloads), 1)
        # The mirror branch copies the module as "<pipeline_name>.py" (no commit subfolder).
        self.assertTrue(result.endswith('lpw_stable_diffusion_xl.py'))
        self.assertTrue(copied)

    def test_v_prefixed_revision_raises_exact_error(self):
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            self.resolve('v0.35.1')

    def test_error_lists_available_versions_including_0_35_1(self):
        try:
            self.resolve('v0.35.1')
        except ValueError as error:
            message = str(error)
        else:
            self.fail('expected ValueError')
        self.assertIn('`custom_revision`: v0.35.1', message)
        self.assertIn('0.35.1', message)

    def test_notebook_pins_bare_revision_matching_guard(self):
        notebook = json.loads(Path('Animagine_XL_4_LPW_Colab.ipynb').read_text())
        model_cell = ''.join(notebook['cells'][4]['source'])
        self.assertIn('custom_revision="0.35.1"', model_cell)
        self.assertIn('0.35.1', VERSIONS)
        self.assertNotIn('custom_revision="v', model_cell)


if __name__ == '__main__':
    unittest.main()
