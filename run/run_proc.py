import os

import pprint
import sys

from iopath.common.file_io import PathManager
from iopath.fb.manifold import ManifoldPathHandler

_root_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root_path)

import src


def run_proc(cfg):
    # cache the cfg to local
    pathmgr = PathManager()
    pathmgr.register_handler(ManifoldPathHandler(), allow_override=True)
    local_cfg = pathmgr.get_local_path(cfg)

    ### initialize Trainer
    trainer = src.core.trainer.Trainer(local_cfg)

    ### copy yaml description file to the save folder
    src.utils.utils.copy_exp_file(trainer)

    # ### copy proc.py file to the save folder
    # src.utils.utils.copy_proc_file(trainer)

    trainer.logger.info(pprint.pformat(trainer.cfg))
    trainer.logger.info("#" * 100)

    ### run the training procedure
    trainer.run()


if __name__ == "__main__":
    cfg = src.utils.utils.parse_args().cfg
    run_proc(cfg)
