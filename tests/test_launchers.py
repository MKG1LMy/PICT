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

    def test_training_bootstrap_forwards_arguments_and_failure(self):
        with patch.dict(sys.modules, {"setup_environment": installer()}):
            spec = importlib.util.spec_from_file_location("bootstrap", ROOT / "deploy/start_training.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        with patch.object(module, "prepare_temp"), patch.object(module, "find_conda", return_value="conda"), \
             patch.object(module, "environment_python", return_value=Path(sys.executable)) as resolve, \
             patch.object(module.subprocess, "call", return_value=7) as run, \
             patch.dict(os.environ, {"PICT_ENV": "custom_env"}), \
             patch.object(sys, "argv", ["launcher", "--models", "cnn", "ann"]):
            self.assertEqual(module.main(), 7)
            resolve.assert_called_once_with("conda", "custom_env")
            self.assertEqual(run.call_args.args[0][0], sys.executable)
            self.assertEqual(run.call_args.args[0][2:], ["--models", "cnn", "ann"])

    def test_invalid_temp_falls_back_without_persistent_settings(self):
        module = installer()
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "not-a-directory"
            invalid.touch()
            with patch.object(module, "ROOT", Path(temporary)), \
                 patch.dict(os.environ, {key: str(invalid) for key in ("TEMP", "TMP", "TMPDIR")}), \
                 patch.object(tempfile, "tempdir", tempfile.tempdir):
                expected = Path(temporary) / ".runtime/tmp"
                self.assertEqual(module.prepare_temp(), expected)
                self.assertEqual(os.environ["TEMP"], str(expected))

    def test_environment_python_is_resolved_from_conda_not_path(self):
        import json
        from types import SimpleNamespace
        module = installer()
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "pict_model"
            python = prefix / ("python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.object(module.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps({"envs": [str(prefix)]}))):
                self.assertEqual(module.environment_python("conda", "pict_model"), python)
                with self.assertRaises(RuntimeError):
                    module.environment_python("conda", "missing")

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
