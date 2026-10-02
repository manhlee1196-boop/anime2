"""Community pipeline revision must use the bare version Diffusers expects."""
import json
import unittest
from pathlib import Path

NOTEBOOK = json.loads(Path('Animagine_XL_4_LPW_Colab.ipynb').read_text())


class CustomRevisionTests(unittest.TestCase):
    def test_model_cell_pins_bare_revision(self):
        source = ''.join(NOTEBOOK['cells'][4]['source'])
        self.assertIn('custom_revision="0.35.1"', source)
        self.assertNotIn('custom_revision="v', source)

    def test_custom_revision_appears_only_in_model_cell(self):
        for index, cell in enumerate(NOTEBOOK['cells']):
            if cell['cell_type'] != 'code' or 'custom_revision' not in ''.join(cell['source']):
                continue
            self.assertEqual(index, 4)

    def test_model_card_documents_optional_token(self):
        self.assertIn('HF_TOKEN', ''.join(NOTEBOOK['cells'][3]['source']))


if __name__ == '__main__':
    unittest.main()
