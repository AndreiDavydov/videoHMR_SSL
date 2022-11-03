from time import time

import torch

from src.procedures.procedures_common import status_msg


def setup(trainer):
    trainer.logger.info("=> Additional setups...")

    trainer.some_heavy_non_trainable_model = torch.randn(10, 10)
    print(
        "****************************************************************************\n"
        "*\tHere can be any setup routine.\n"
        "*\tFor example, one can initialize non-trainable modules(e.g., SMPL)\n"
        "****************************************************************************\n"
    )


def train(trainer):
    absolute_start = time()

    dl_len = len(trainer.dataload.train)
    for batch_idx, _ in enumerate(trainer.dataload.train, start=1):
        input_x = torch.randn(3, 1, 5).to(device=trainer.device0, non_blocking=True)
        gt_dummy = torch.randn(3, 1, 5).to(device=trainer.device0, non_blocking=True)
        batch_size = gt_dummy.size(0)

        trainer.optim.zero_grad()

        pred = trainer.models.dummynet(input_x)
        loss = trainer.losses.dummyloss(pred, gt_dummy)

        loss.backward()
        trainer.optim.step()

        trainer.meters.train.dummyloss.update(loss.item(), n=batch_size)
        total_time = time() - absolute_start
        status_msg(
            trainer, batch_idx, dl_len, trainer.meters.train.dummyloss, total_time
        )


def valid(trainer):

    absolute_start = time()
    trainer.logger.info("Validation is fine, just chill...")

    total_time = time() - absolute_start
    msg = f"=> Epoch [{trainer.cur_epoch}] Total {total_time:5.1f}s \t"
    trainer.logger.info(msg)

    perf_indicator = float("inf")
    return perf_indicator
