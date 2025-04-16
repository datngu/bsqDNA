import inspect
from datetime import datetime
from pathlib import Path
import time

import torch

# Enable Tensor Core operations for better performance on CUDA devices
torch.set_float32_matmul_precision('medium')

from .bsq import BSQDNA
from .data import create_dataloader

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def train(epochs: int = 4, batch_size: int = 1024, val_batch_size: int = 1024, data_dir: str = "test_data"):
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger
    import logging
    from lightning.pytorch.profilers import PyTorchProfiler

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    class PatchTrainer(L.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.data_dir = data_dir
            logger.info("Initialized PatchTrainer")
            logger.info(f"Using DNA data from {data_dir}")

        def training_step(self, batch, batch_idx):            
            x, _ = batch
            
            x_hat, additional_losses = self.model(x)
            loss = torch.nn.functional.cross_entropy(x_hat, x)
            
            self.log("train/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"train/{k}", v)
            return loss + sum(additional_losses.values())

        def validation_step(self, batch, batch_idx):
            logger.info(f"Starting validation step {batch_idx}")
            x, _ = batch
            with torch.no_grad():
                x_hat, additional_losses = self.model(x)
                loss = torch.nn.functional.cross_entropy(x_hat, x)
            self.log("validation/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"validation/{k}", v)
            if batch_idx == 0:
                self.logger.experiment.add_text(
                    "sample_reconstruction",
                    f"Original vs Reconstructed (first 50 bases):\n{x[0,:,:50]}\n{x_hat[0,:,:50]}",
                    self.global_step
                )
            return loss

        def configure_optimizers(self):
            logger.info("Configuring optimizers")
            return torch.optim.AdamW(self.parameters(), lr=2e-3)

        def train_dataloader(self):
            logger.info("Setting up training dataloader")
            return create_dataloader(
                self.data_dir, "train", 
                batch_size=batch_size, 
                num_workers=4,
                shuffle=True
            )

        def val_dataloader(self):
            logger.info("Setting up validation dataloader")
            return create_dataloader(
                self.data_dir, "valid", 
                batch_size=val_batch_size, 
                num_workers=4, 
                shuffle=False
            )

    class CheckPointer(L.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            fn = Path(f"checkpoints/{timestamp}_{model_name}.pth")
            fn.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model, fn)
            torch.save(model, Path(__file__).parent / f"{model_name}.pth")

    # Create the model
    model_name = "BSQDNA"
    logger.info("Creating BSQDNA model")
    model = BSQDNA()

    # Create the lightning model
    logger.info("Creating PatchTrainer")
    l_model = PatchTrainer(model)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.info("Setting up TensorBoard logger")
    tb_logger = TensorBoardLogger("logs", name=f"{timestamp}_{model_name}")
    
    # Create the profiler
    profiler = PyTorchProfiler(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"logs/profiler"),
        record_shapes=True,
        profile_memory=True,
    )

    logger.info("Creating trainer")
    trainer = L.Trainer(
        max_epochs=epochs, 
        logger=tb_logger, 
        callbacks=[CheckPointer()],
        accelerator=DEVICE,
        devices=1,
        num_sanity_val_steps=2,
        deterministic=False,
        gradient_clip_val=1.0,
        profiler=profiler,
    )
    
    try:
        logger.info("Starting training")
        trainer.fit(model=l_model)
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    from fire import Fire

    Fire(train)
