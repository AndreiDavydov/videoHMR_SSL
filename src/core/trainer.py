import datetime
import os
from os.path import join as joinpath

import src

import torch
from easydict import EasyDict as edict

# from iopath.common.file_io import PathManager
# from iopath.fb.manifold import ManifoldPathHandler

# pathmgr = PathManager()
# pathmgr.register_handler(ManifoldPathHandler(), allow_override=True)

# import src.utils.manifold_utils as manifold


class Trainer:
    """
    Class, which runs the training procedure.
    Also keep all necessary components together:
            models,
            optimizers,
            schedulers,
            datasets,
            dataloaders,
            losses,
            procedures.
    This universal module is developed to run deep learning experiments of any kind.
    """

    def __init__(self, cfg):  # initialized by <configname>.yaml file
        self.cfg = src.core.config_utils.get_config(cfg)

        if "ADD_TIMESTAMP" in self.cfg and self.cfg.ADD_TIMESTAMP:
            now = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            self.cfg.EXP_NAME += "__" + now

        self.final_output_dir = src.utils.utils.init_exp_folder(self.cfg)
        (
            self.logger,
            self.log_filename,
            log_filepath,
        ) = src.utils.utils.create_logger(self.final_output_dir, self.cfg)

        # ### create remote exp folder
        # self.remote_exp_dir = manifold.mkdir_remote(
        #     self.cfg.EXP_NAME, add_subfolder=self.cfg.OUTPUT_DIR.split("/")[-1]
        # )
        # ### copy logger file from local
        # self.remote_log_file = log_filepath
        # manifold.copy_file_from_local(self.remote_log_file, self.remote_exp_dir)
        # ### copy cfg from local
        # manifold.copy_file_from_local(self.cfg.CONFIG_FILENAME, self.remote_exp_dir)

        self.print_freq = self.cfg.PRINT_FREQ

        assert (
            self.cfg.GPUS != ""
        ), "At least one GPU device must be utilized! Check config file"

        self.gpus = [int(i) for i in self.cfg.GPUS.split(",")]
        self.device0 = torch.device(f"cuda:{self.gpus[0]}")
        self.logger.info(f"=> GPUs with indices {self.gpus} are used.")

        self.logger.info("=> Trainer init: loading models...")
        self.models = src.models.models_common.get_models_by_config(
            self.cfg, self.gpus, self.device0
        )
        (
            self.losses,
            self.losses_weights,
        ) = src.losses.losses_common.get_losses_by_config(self.cfg, self.device0)

        self.logger.info("=> Trainer init: loading optimizers and schedulers...")
        self.optim = src.optimizers.optimizers_common.get_optim_by_config(
            self.cfg, self.models
        )
        self.sched = src.optimizers.optimizers_common.get_sched_by_config(
            self.cfg, self.optim
        )

        self.cur_epoch = 0
        self.end_epoch = self.cfg.TRAINING.END_EPOCH

        ### assume that the lower the better, perf_indicator reflects current quality
        self.perf_indicator, self.perf_indicator_best = 999_999_999, 999_999_999

        self.meters = src.utils.metrics.get_meters_by_config(
            self.cfg, self.losses.keys()
        )

        self.load_state(self.cfg)

        self.logger.info("=> Trainer init: loading datasets...")
        self.datasets = src.datasets.datasets_common.get_datasets_by_config(self.cfg)
        self.dataload = src.datasets.datasets_common.get_dataloaders_by_config(
            self.cfg, self.datasets, num_gpus=len(self.gpus)
        )

        (
            self.proc,
            self.proc_setup,
        ) = src.procedures.procedures_common.get_procedures_by_config(self.cfg)
        self.logger.info("=> Trainer is initialized.\n")

    def save_state(self, best=False):
        if "FAKERUN" in self.cfg and self.cfg.FAKERUN:
            return

        self.logger.info(f"=> Epoch [{self.cur_epoch}], saving checkpoint...")

        state2save = {}

        ### save models state dicts
        for model_key in self.models:
            model_name = model_key + "_state_dict"
            if self.models[model_key] is not None:
                try:  # by default all models are of type DataParallel
                    state2save[model_name] = self.models[model_key].module.state_dict()
                except AttributeError:
                    state2save[model_name] = self.models[model_key].state_dict()

        ### save optim and sched state dicts
        state2save.update(self.optim.save_state())
        state2save.update(self.sched.save_state())

        ### current epoch and all meters at the moment when current epoch is finished
        state2save["epoch"] = self.cur_epoch
        state2save["perf_indicator"] = self.perf_indicator
        state2save["perf_indicator_best"] = self.perf_indicator_best
        ckpt_path = joinpath(self.final_output_dir, "ckpt.pth")
        torch.save(state2save, ckpt_path)
        # manifold.copy_file_from_local(ckpt_path, self.remote_exp_dir)

        ### save ckpts to avoid incident errors during saving
        if (
            self.cfg.SAVE_EVERY_EPOCH
            and self.cur_epoch % self.cfg.SAVE_EVERY_EPOCH == 0
        ):
            ckpt_path = joinpath(
                self.final_output_dir, f"ckpt_{self.cur_epoch:04d}.pth"
            )
            torch.save(state2save, ckpt_path)
            # manifold.copy_file_from_local(ckpt_path, self.remote_exp_dir)
        if best:
            self.logger.info(
                f"=> Epoch [{self.cur_epoch}] - THE BEST MODEL SO FAR...\n"
            )
            ckpt_path = joinpath(self.final_output_dir, "best.pth")
            torch.save(state2save, ckpt_path)
            # manifold.copy_file_from_local(ckpt_path, self.remote_exp_dir)

        for mode in self.meters:
            for loss_type in self.meters[mode]:
                self.meters[mode][loss_type].save_state(self.final_output_dir)

        self.local_metrics_dir = os.path.join(self.final_output_dir, "metrics")
        # manifold.copy_folder_from_local(self.local_metrics_dir, self.remote_exp_dir)

        self.logger.info(
            f"=> Epoch [{self.cur_epoch}], checkpoint is saved to {self.final_output_dir}"
        )
        # manifold.copy_file_from_local(self.remote_log_file, self.remote_exp_dir)

    def load_models(self, ckpt):
        for model in self.models:

            key = model + "_state_dict"
            if key not in ckpt: 
                self.logger.info(f"=> No key '{key}' in ckpt!")
                continue

            try:
                self.models[model].module.load_state_dict(ckpt[key], strict=False)
            except AttributeError:
                self.models[model].load_state_dict(ckpt[key], strict=False)

            self.logger.info(f'=> "{model}" model is loaded from the checkpoint')

    def load_checkpoint(self, ckpt):
        if isinstance(ckpt, str):
            # ckpt_dict = torch.load(pathmgr.get_local_path(ckpt), map_location="cpu")
            ckpt_dict = torch.load(ckpt, map_location="cpu")

        elif isinstance(ckpt, list):
            # NOTE all state dicts are combined together
            ckpt_dict = {}
            for ckpt_path in ckpt:
                ckpt_dict.update(torch.load(ckpt_path, map_location="cpu"))

        elif isinstance(ckpt, dict):
            # NOTE it is assumed that there is only one model state dict lies in every ckpt path
            ckpt_dict = {}
            for (
                model,
                ckpt_path,
            ) in ckpt.items():  # ckpt keys must coincide with all initialized models
                assert (
                    model in self.models
                ), f'ckpt model "{model}" is not initialized in the current experiment!'
                model_dict = torch.load(ckpt_path, map_location="cpu")
                # check whether "model" and "model_ckpt" (from the <model_ckpt>_state_dict) coincide
                model_ckpt = [
                    e for e in list(model_dict.keys()) if e.find("_state_dict") != -1
                ][0]
                model_ckpt = model_ckpt.split("_")[
                    0
                ]  # name of the saved model in the ckpt
                if model != model_ckpt:
                    # rename all instances with "model_ckpt" with "model"
                    model_dict[model + "_state_dict"] = model_dict[
                        model_ckpt + "_state_dict"
                    ]
                    del model_dict[model_ckpt + "_state_dict"]

                    ### saved state of the optimizer is optional
                    if model_ckpt + "_state_dict_optim" in model_dict:
                        model_dict[model + "_state_dict_optim"] = model_dict[
                            model_ckpt + "_state_dict_optim"
                        ]
                        del model_dict[model_ckpt + "_state_dict_optim"]

                    ### saved state of the scheduler is optional
                    if model_ckpt + "_state_dict_sched" in model_dict:
                        model_dict[model + "_state_dict_sched"] = model_dict[
                            model_ckpt + "_state_dict_sched"
                        ]
                        del model_dict[model_ckpt + "_state_dict_sched"]

                ckpt_dict.update(model_dict)
        return ckpt_dict

    def autoresume(self, cfg):
        ### for the presaved ckpt and start from it
        # ckpt_path = os.path.join(self.remote_exp_dir, "ckpt.pth")
        ckpt_path = os.path.join(self.final_output_dir, "ckpt.pth")
        
        cfg.TRAINING.RESUME = edict()
        cfg.TRAINING.RESUME.CKPT = ckpt_path
        try:
            self.logger.info(f"=> Trying to autoresume from {cfg.TRAINING.RESUME.CKPT}")
            self.load_state(cfg)
        except OSError:  # Path not found (manifold error)
            self.logger.info(
                "=> Auto-resume does not succeed - no presaved ckpt found!"
            )
            self.logger.info("=> Experiment will start from scratch...")
            pass

    def load_state(self, cfg):
        ### load models, optims, scheds dicts, metrics arrays, current epoch and perf_indicator
        if "RESUME" not in cfg.TRAINING:
            return

        if cfg.TRAINING.RESUME == "auto":
            self.autoresume(cfg)
            return

        assert (
            "CKPT" in cfg.TRAINING.RESUME
        ), "You want to resume training but didnot specify the existing checkpoint file"

        self.logger.info(f"=> loading checkpoint from {cfg.TRAINING.RESUME.CKPT} ...")
        ckpt = self.load_checkpoint(cfg.TRAINING.RESUME.CKPT)

        # load models state dicts
        self.load_models(ckpt)

        # in case LOAD_MODELS_ONLY == True training will start from epoch 1, "fine-tuning"
        if not (
            "LOAD_MODELS_ONLY" in cfg.TRAINING.RESUME
            and cfg.TRAINING.RESUME.LOAD_MODELS_ONLY
        ):

            # load some attributes
            self.cur_epoch = ckpt["epoch"]
            self.perf_indicator = ckpt["perf_indicator"]
            self.perf_indicator_best = ckpt["perf_indicator_best"]

            # load optim and sched state dicts
            self.optim.load_state(ckpt)
            self.sched.load_state(ckpt)

            # load metrics
            file_end = cfg.TRAINING.RESUME.CKPT.split("/")[-1]
            ckpt_name = cfg.TRAINING.RESUME.CKPT[: -len(file_end)]
            # if 'LOAD_METRICS' in cfg.TRAINING.RESUME and cfg.TRAINING.RESUME.LOAD_METRICS:
            for mode in self.meters:
                for loss_type in self.meters[mode]:
                    metrics_file = joinpath(
                        os.path.dirname(ckpt_name), "metrics", f"{loss_type}_{mode}.pth"
                    )
                    self.meters[mode][loss_type].load_state(
                        metrics_file
                    )

        self.logger.info(f"=> checkpoint is loaded. (last epoch {self.cur_epoch})")

    def run(self):
        """Runs the training procedure specified in "self.proc"."""
        self.proc_setup(self)

        ### either cur_epoch was 0 or model with its latest value is already presaved
        self.cur_epoch += 1

        for _ in range(self.cur_epoch, self.end_epoch + 1):
            self.proc(self)
            self.cur_epoch += 1
            # manifold.copy_file_from_local(self.remote_log_file, self.remote_exp_dir)

        self.logger.info(
            f'\nTraining of "{self.cfg.EXP_NAME}" experiment is finished. \n'
        )
        # manifold.copy_file_from_local(self.remote_log_file, self.remote_exp_dir)
