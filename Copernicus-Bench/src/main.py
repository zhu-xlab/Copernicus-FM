import datetime
import os
from pathlib import Path
import warnings
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy
from datasets.data_module import BenchmarkDataModule
from lightning.pytorch import seed_everything
from factory import create_model
import hydra
from omegaconf import DictConfig

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Seed everything
    seed_everything(cfg.seed)

    # Create output directory
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Scale learning rate for multi-GPU
    cfg.lr *= cfg.num_gpus

    # Setup logger
    experiment_name = f"{cfg.model.model_type}_{cfg.dataset.dataset_name}"
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=f"{experiment_name}_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tracking_uri=f"file:{os.path.join(cfg.output_dir, 'mlruns')}",
    )

    wandb_logger = WandbLogger(
    project=cfg.wandb_project,  # Change this to your WandB project name
    name=f"{cfg.model.model_type}_{cfg.dataset.dataset_name}_run",
    #log_model="all",  # Logs model checkpoints
    )

    # Callbacks
    if cfg.task == "regression":
        model_monitor = "val_rmse"
        monitor_mode = "min"
    elif cfg.task == "segmentation" or cfg.task == "changedetection":
        model_monitor = "val_miou"
        monitor_mode = "max"
    else:
        model_monitor = "val_acc1"
        monitor_mode = "max"

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.output_dir, "checkpoints"),
            filename="best_model-{epoch}",
            monitor=model_monitor,
            mode=monitor_mode,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Initialize trainer
    trainer = Trainer(
        #logger=mlf_logger,
        logger=[mlf_logger,wandb_logger],
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False) if cfg.strategy == "ddp" and cfg.num_gpus > 1 else cfg.strategy,
        devices=cfg.num_gpus,
        max_epochs=cfg.epochs,
        num_sanity_val_steps=0,
    )

    # Initialize data module
    #cfg.dataset.image_resolution = cfg.model.image_resolution
    data_module = BenchmarkDataModule(
        dataset_config=cfg.dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
    )

    # Create model (assumed to be a LightningModule)
    model = create_model(cfg, cfg.model, cfg.dataset)

    # Train
    trainer.fit(model, data_module, ckpt_path=cfg.resume if cfg.resume else None)

    # Test
    best_checkpoint_path = callbacks[0].best_model_path
    trainer.test(model, data_module, ckpt_path=best_checkpoint_path)


if __name__ == "__main__":
    os.environ["MODEL_WEIGHTS_DIR"] = os.getenv("MODEL_WEIGHTS_DIR", "./fm_weights")
    main()
