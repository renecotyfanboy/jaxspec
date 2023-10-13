import pathlib
import os
import sys
from unittest import TestCase
"""
import chex
chex.set_n_cpu_devices(n=4)
"""

from mktestdocs import check_md_file
#Allow relative imports for github workflows
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)


class TestDOC(TestCase):

    tutorial_path = source_dir / pathlib.Path("docs") / "tutorial"

    def test_basic_fit(self):

        check_md_file(fpath=self.tutorial_path / "basic_fit.md", memory=True)

    def test_model_building(self):

        check_md_file(fpath=self.tutorial_path / "build_a_model.md", memory=True)