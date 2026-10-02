"""Check notebook prerequisites and launch without GPU, Gradio or network."""
import json
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

NOTEBOOK = json.loads(Path('Animagine_XL_4_LPW_Colab.ipynb').read_text())
LAUNCH = ''.join(NOTEBOOK['cells'][10]['source'])
UI = ''.join(NOTEBOOK['cells'][8]['source'])


class LaunchTests(unittest.TestCase):
    def test_missing_demo_does_not_ask_password(self):
        with patch('getpass.getpass') as password:
            with self.assertRaisesRegex(RuntimeError, 'Chạy ô 4'):
                exec(LAUNCH, {})
            password.assert_not_called()

    def test_unfinished_ui_does_not_launch(self):
        demo = types.SimpleNamespace(launch=Mock())
        with patch('getpass.getpass') as password:
            with self.assertRaisesRegex(RuntimeError, 'chưa sẵn sàng'):
                exec(LAUNCH, {'demo': demo, 'DEMO_READY': False})
            password.assert_not_called()
            demo.launch.assert_not_called()

    def test_successful_launch_and_password_cleanup(self):
        demo = types.SimpleNamespace(launch=Mock())
        ns = {'demo': demo, 'DEMO_READY': True}
        with patch('getpass.getpass', return_value='test-password'):
            exec(LAUNCH, ns)
        demo.launch.assert_called_once_with(share=True, server_name='0.0.0.0',
            auth=('anime', 'test-password'), show_error=False, inline=False)
        self.assertNotIn('password', ns)

    def test_short_password_cleanup(self):
        demo = types.SimpleNamespace(launch=Mock())
        ns = {'demo': demo, 'DEMO_READY': True}
        with patch('getpass.getpass', return_value='short'):
            with self.assertRaises(ValueError): exec(LAUNCH, ns)
        self.assertNotIn('password', ns)
        demo.launch.assert_not_called()

    def test_launch_failure_cleanup(self):
        ns = {'demo': types.SimpleNamespace(launch=Mock(side_effect=OSError('tunnel'))),
              'DEMO_READY': True}
        with patch('getpass.getpass', return_value='test-password'):
            with self.assertRaises(OSError): exec(LAUNCH, ns)
        self.assertNotIn('password', ns)

    def test_missing_ui_prerequisites_report_before_imports(self):
        ns = {'DEMO_READY': True}
        # Run only prerequisite guard: no third-party packages needed.
        with self.assertRaisesRegex(RuntimeError, 'Chạy thành công ô 2'):
            exec(UI.split('import gradio as gr')[0], ns)
        self.assertFalse(ns['DEMO_READY'])


if __name__ == '__main__':
    unittest.main()
