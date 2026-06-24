import logging
import time
from datetime import timedelta
from utils import seed_everything
seed_everything(1143)
import torch
torch.set_float32_matmul_precision('medium')
from omegaconf import DictConfig
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from new_model import AVSE4BaselineModule
from dataset import AVSE4DataModule

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    # ---------------------------------------------------------------
    # Checkpoint setup (EarlyStopping removed -- see note below)
    #
    # A real 35-epoch run was checked against the exact EarlyStopping
    # logic that used to be configured here (monitor="val_pesq",
    # mode="max", patience=20, min_delta=0.005), replayed step by step
    # against the actual logged PESQ values. Result: the "wait" counter
    # only reached 4 out of the 20-epoch patience by epoch 34. EarlyStopping
    # was NOT what stopped that run -- PESQ was still climbing (1.333 ->
    # 1.681) with no plateau in sight. Something external (most likely a
    # SLURM walltime limit, a manual interruption, or an OOM kill -- none
    # of which leave any trace in the Lightning metrics CSV) ended the
    # process mid-epoch-35. Check `sacct -j <jobid> --format=State,ExitCode`
    # or the job's .err file on the cluster to confirm which one.
    #
    # Per direct request, EarlyStopping is removed entirely here. Nothing
    # in this script can stop training before max_epochs is reached or
    # the job is interrupted externally. If you want a safety net back
    # later, the principle to keep is: monitor val_pesq (not val_loss),
    # because L1 plateaus long before perceptual quality does, and use a
    # patience generous enough (20+) that a slow, noisy climb is not
    # mistaken for a real plateau.
    # ---------------------------------------------------------------
    checkpoint_best_pesq = ModelCheckpoint(
        monitor="val_pesq",
        mode="max",
        filename="model-{epoch:02d}-pesq{val_pesq:.3f}",
        save_top_k=2,
        save_last=True,
    )
    checkpoint_debug_l1 = ModelCheckpoint(
        monitor="val_loss_epoch",
        mode="min",
        filename="debug-l1-{epoch:02d}-{val_loss:.3f}",
        save_top_k=1,
    )

    # Resilience checkpoint: save_last (above) only writes at the end of
    # an epoch, so if the job is killed mid-epoch (as appears to have
    # happened at epoch 35 here), everything since the last epoch
    # boundary is lost. This callback also saves every 30 minutes of
    # wall-clock time, independent of epoch boundaries, so a resume via
    # `ckpt_path=<this_checkpoint>` never loses more than ~30 minutes of
    # progress to an external interruption.
    #
    # save_top_k requires a monitored metric to rank "top" checkpoints by;
    # with no monitor, Lightning only accepts save_top_k in {0, 1, -1}.
    # save_top_k=3 with monitor=None is not a valid combination (this is
    # exactly what crashed on the cluster: "No quantity for top_k to
    # track"). save_top_k=1 here means: always keep just the single most
    # recent time-based snapshot, overwriting the previous one -- enough
    # to resume from, without accumulating unbounded checkpoint files on
    # a multi-hour run.
    checkpoint_time_based = ModelCheckpoint(
        filename="time-ckpt-{epoch:02d}-{step}",
        train_time_interval=timedelta(minutes=30),
        save_top_k=1,
    )

    callbacks = [checkpoint_best_pesq, checkpoint_debug_l1, checkpoint_time_based]

    datamodule = AVSE4DataModule(
        data_root=cfg.data.root,
        batch_size=cfg.data.batch_size,
        audio_norm=cfg.data.audio_norm,
        rgb=cfg.data.rgb,
        num_channels=cfg.data.num_channels,
    )
    model = AVSE4BaselineModule(num_channels=cfg.data.num_channels)

    trainer = Trainer(
        default_root_dir=cfg.trainer.log_dir,
        callbacks=callbacks,
        deterministic=cfg.trainer.deterministic,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        fast_dev_run=cfg.trainer.fast_dev_run,
        devices=cfg.trainer.gpus,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        strategy=cfg.trainer.strategy,
        max_epochs=cfg.trainer.max_epochs,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        detect_anomaly=cfg.trainer.detect_anomaly,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        profiler=cfg.trainer.profiler,
        # Multi-GPU DDP correctness: DualPopModulationLayer uses
        # BatchNorm1d in its temporal conv block. Without this, running
        # statistics are computed independently per rank and diverge
        # across the 3 GPUs used in this job.
        sync_batchnorm=True,
    )

    start = time.time()
    trainer.fit(model, datamodule, ckpt_path=cfg.trainer.ckpt_path)
    log.info(f"Time taken {time.time() - start} sec")
    log.info(f"Best PESQ checkpoint: {checkpoint_best_pesq.best_model_path}")
    log.info(f"Best PESQ value: {checkpoint_best_pesq.best_model_score}")


if __name__ == '__main__':
    main()