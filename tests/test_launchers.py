"""Launcher and log plumbing tests; never install packages or start training."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def installer():
    spec = importlib.util.spec_from_file_location("installer", ROOT / "deploy/setup_environment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LauncherTests(unittest.TestCase):
    def test_installer_log_contains_subprocess_stdout_stderr_and_exit_code(self):
        module = installer()
        with tempfile.TemporaryDirectory() as temporary:
            def operation(argv):
                try:
                    module.run_command([sys.executable, "-c",
                                        "import sys; print('stdout-record'); print('stderr-record', file=sys.stderr); sys.exit(7)"])
                except subprocess.CalledProcessError as error:
                    return error.returncode
            with patch.object(module, "ROOT", Path(temporary)), patch.object(module, "main", operation):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(module.logged_main([]), 7)
            logs = list((Path(temporary) / "deployment-logs").glob("*.log"))
            self.assertEqual(len(logs), 1)
            text = logs[0].read_text(encoding="utf-8")
            self.assertIn("stdout-record", text)
            self.assertIn("stderr-record", text)

    @unittest.skipUnless(os.name == "nt", "Windows launcher")
    def test_bat_forwards_arguments_environment_and_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            conda = Path(temporary) / "fake conda.bat"
            conda.write_text("@echo off\necho %*\nexit /b 7\n", encoding="ascii")
            environment = dict(os.environ, CONDA_EXE=str(conda), PICT_ENV="custom_env",
                               PICT_TRAIN_NO_PAUSE="1")
            result = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c",
                                     str(ROOT / "train.bat"), "--models", "cnn", "ann"],
                                    env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
            self.assertIn("-n \"custom_env\"", result.stdout)
            self.assertIn("run_training.py", result.stdout)
            self.assertIn("--models cnn ann", result.stdout)

    @unittest.skipUnless(os.name == "posix", "Native Linux launcher")
    def test_shell_forwards_arguments_environment_and_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            conda = Path(temporary) / "fake conda"
            conda.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\nexit 7\n', encoding="utf-8")
            conda.chmod(0o700)
            environment = dict(os.environ, CONDA_EXE=str(conda), PICT_ENV="custom_env")
            for name in ("train.sh", "setup.sh"):
                result = subprocess.run(["bash", str(ROOT / name), "--help"], env=environment,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertIn("--help", result.stdout)
                self.assertIn("custom_env" if name == "train.sh" else "base", result.stdout)


if __name__ == "__main__":
    unittest.main()
