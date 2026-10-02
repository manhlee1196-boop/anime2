"""Installer control-flow tests. No pip installs or GPU access during tests."""
import json
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

NOTEBOOK = json.loads(Path('Animagine_XL_4_LPW_Colab.ipynb').read_text())
INSTALL = ''.join(NOTEBOOK['cells'][2]['source'])
MODEL_GUARD = ''.join(NOTEBOOK['cells'][4]['source']).split('import torch, importlib, inspect')[0]


class InstallTests(unittest.TestCase):
    def test_installs_then_checks_in_fresh_process(self):
        with patch('subprocess.check_call') as call, patch('builtins.print'):
            exec(INSTALL, {})
        self.assertEqual(call.call_count, 2)
        command = call.call_args_list[0].args[0]
        self.assertEqual(command[:4], [sys.executable, '-m', 'pip', 'install'])
        for package in ('gradio==5.49.1', 'pydantic==2.11.10',
                        'starlette==0.46.2', 'fastapi==0.115.12'):
            self.assertIn(package, command)
        self.assertEqual(call.call_args_list[1].args[0][:2], [sys.executable, '-c'])

    def test_install_failure_does_not_run_ui_check(self):
        with patch('subprocess.check_call', side_effect=subprocess.CalledProcessError(1, 'pip')) as call:
            with self.assertRaises(subprocess.CalledProcessError): exec(INSTALL, {})
        self.assertEqual(call.call_count, 1)

    def test_ui_check_failure_not_silenced(self):
        with patch('subprocess.check_call', side_effect=[0, subprocess.CalledProcessError(1, 'check')]):
            with self.assertRaises(subprocess.CalledProcessError): exec(INSTALL, {})

    def test_loaded_old_module_requires_restart(self):
        with patch.dict(sys.modules, {'gradio': types.SimpleNamespace(__version__='old')}):
            with patch('importlib.metadata.version', return_value='5.49.1'):
                with self.assertRaisesRegex(RuntimeError, 'Restart session'):
                    exec(MODEL_GUARD, {})


if __name__ == '__main__':
    unittest.main()
