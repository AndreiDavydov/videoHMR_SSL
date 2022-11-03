import os

from iopath.common.file_io import PathManager
from iopath.fb.manifold import ManifoldPathHandler

pathmgr = PathManager()
pathmgr.register_handler(ManifoldPathHandler(), allow_override=True)

REMOTE_EXPS_DIR = "manifold://xr_body/tree/personal/andreydavydov/my_exps/"


def mkdir_remote(dir_name, add_subfolder=None, parent_dir=REMOTE_EXPS_DIR):
    """
    dir_name : str, name of the folder to create
    parent_dir : remote folder, dir_name will be made inside it
    """
    if add_subfolder is not None:
        dir_name = os.path.join(add_subfolder, dir_name)
    remote_exp_dir = os.path.join(parent_dir, dir_name)
    pathmgr.mkdirs(remote_exp_dir)  # pathmgr does not overwrite if exists
    return remote_exp_dir


def copy_file_from_local(local_file, remote_dir):
    name = local_file.split("/")[-1]
    remote_file = os.path.join(remote_dir, name)
    try:
        pathmgr.copy_from_local(local_file, remote_file, overwrite=True)
    #  canary flow logging issue
    #  Logger path that is created in Trainer is not correct
    #  TODO write a custom logger class that creates a plain text file and writes in it
    except AssertionError:
        pass


def copy_folder_from_local(local_dir, remote_dir):
    # TODO looks bad (will fail if there are subfolders)
    # works fine if there are no subfolders in copied folder
    # better to find some solution in iopath
    name_dir = local_dir.split("/")[-1]
    remote_subdir = mkdir_remote(name_dir, parent_dir=remote_dir)
    for local_name in pathmgr.ls(local_dir):
        local_file = os.path.join(local_dir, local_name)
        copy_file_from_local(local_file, remote_subdir)
