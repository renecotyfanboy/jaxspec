import os
import sys
from testbook import testbook

# Allow relative imports for github workflows
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)


@testbook("../docs/examples/fitting_example.ipynb", execute=True)
def test_fitting(tb):
    pass


@testbook("../docs/examples/fakeits.ipynb", execute=True)
def test_fakeits(tb):
    pass


@testbook("../docs/examples/load_data.ipynb", execute=True)
def test_load_data(tb):
    pass
