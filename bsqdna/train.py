from datetime import datetime
from pathlib import Path
import torch

# Enable Tensor Core operations for better performance on CUDA devices
torch.set_float32_matmul_precision('medium')

from .bsq import BSQDNA
from .data import create_dataloader

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def train(epochs: int = 4, batch_size: int = 1024, val_batch_size: int = 2048, data_dir: str = "test_data_h5"):
    """
    Train the BSQDNA model
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size (smaller, limited by GPU memory for training)
        val_batch_size: Validation batch size (larger, can be 2-4x training batch size)
        data_dir: Directory containing the training data
    """
    import lightning as L
    from lightning.pytorch.loggers import WandbLogger
    import logging
    import torch.cuda as cuda
    import wandb

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Log GPU memory info
    if cuda.is_available():
        logger.info(f"GPU: {cuda.get_device_name(0)}")
        logger.info(f"Total GPU Memory: {cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logger.info(f"Using batch size: {batch_size}")

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
            x, _ = batch
            with torch.no_grad():
                x_hat, additional_losses = self.model(x)
                loss = torch.nn.functional.cross_entropy(x_hat, x)
            self.log("validation/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"validation/{k}", v)
            if batch_idx == 0:
                # Log sample reconstructions to wandb
                self.logger.experiment.log({
                    "sample_reconstruction": wandb.Html(
                        f"Original vs Reconstructed (first 50 bases):<br>"
                        f"Original: {x[0,:,:50]}<br>"
                        f"Reconstructed: {x_hat[0,:,:50]}"
                    )
                })
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
                self.data_dir, "val",
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
    logger.info("Setting up Wandb logger")
    wandb_logger = WandbLogger(
        project="bsqDNA",
        name=f"{timestamp}_{model_name}",
        log_model="all"
    )

    logger.info("Creating trainer")
    trainer = L.Trainer(
        max_epochs=epochs, 
        logger=wandb_logger, 
        callbacks=[CheckPointer()],
        accelerator=DEVICE,
        devices=1,
        num_sanity_val_steps=2,
        deterministic=False,
        gradient_clip_val=1.0,
    )
    
    try:
        logger.info("Starting training")
        trainer.fit(model=l_model)
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    from fire import Fire

    Fire(train)
