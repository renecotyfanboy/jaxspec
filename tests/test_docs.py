import os
import sys
import subprocess
from unittest import TestCase

# Allow relative imports for github workflows
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)


class TestDocs(TestCase):
    def test_basic_fit(self):
        result = subprocess.run(
            ["python3", os.path.join(source_dir, "docs/examples/plot_1_basic_fit.py")],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, f"Erreur : {result.stderr}")

    def test_build_model(self):
        result = subprocess.run(
            [
                "python3",
                os.path.join(source_dir, "docs/examples/plot_2_build_model.py"),
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, f"Erreur : {result.stderr}")

    def test_custom_component(self):
        result = subprocess.run(
            [
                "python3",
                os.path.join(source_dir, "docs/examples/plot_3_custom_component.py"),
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, f"Erreur : {result.stderr}")
