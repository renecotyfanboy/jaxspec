import os
import sys
from unittest import TestCase
import runpy

# Allow relative imports for github workflows
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)


class TestDocs(TestCase):
    def test_basic_fit(self):
        runpy.run_module("docs.examples.plot_1_basic_fit", run_name="__main__")

    def test_build_model(self):
        runpy.run_module("docs.examples.plot_2_build_model", run_name="__main__")

    def test_custom_component(self):
        runpy.run_module("docs.examples.plot_3_custom_component", run_name="__main__")
