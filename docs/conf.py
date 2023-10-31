import warnings
from mkdocs_gallery.gen_gallery import DefaultResetArgv

warnings.filterwarnings("ignore")

conf = {
    "ignore_pattern": "doc_variable.py",
    "reset_argv": DefaultResetArgv(),
}
