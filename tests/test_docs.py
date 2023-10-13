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

    def test_tutorial(self):

        fpath = source_dir / pathlib.Path("docs") / "tutorial"
        # Assumes that cell-blocks depend on each other.
        check_md_file(fpath=fpath / "basic_fit.md", memory=True)
        check_md_file(fpath=fpath / "build_a_model.md", memory=True)
