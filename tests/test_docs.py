import os
import sys
import pathlib
from testbook import testbook

# Allow relative imports for github workflows
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)

example_path = str(source_dir / pathlib.Path("docs") / "examples")


@testbook(open(example_path + "/fitting_example.ipynb"), execute=True)
def test_fitting(tb):
    pass


@testbook(open(example_path + "/fakeits.ipynb"), execute=True)
def test_fakeits(tb):
    pass


@testbook(open(example_path + "/load_data.ipynb"), execute=True)
def test_load_data(tb):
    pass
