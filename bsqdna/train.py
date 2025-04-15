import inspect
from datetime import datetime
from pathlib import Path

import torch

from .bsq import BSQDNA
from .data import DNADataset

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def train(epochs: int = 5, batch_size: int = 32, val_batch_size: int = 64):
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    class PatchTrainer(L.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            logger.info("Initialized PatchTrainer")

        def training_step(self, x, batch_idx):
            logger.info(f"Starting training step {batch_idx}")
            x_hat, additional_losses = self.model(x)
            loss = torch.nn.functional.cross_entropy(x_hat, x)
            self.log("train/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"train/{k}", v)
            return loss + sum(additional_losses.values())

        def validation_step(self, x, batch_idx):
            logger.info(f"Starting validation step {batch_idx}")
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
            return torch.optim.AdamW(self.parameters(), lr=1e-3)

        def train_dataloader(self):
            logger.info("Setting up training dataloader")
            dataset = DNADataset("test_data", "train")
            return torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                num_workers=0,  # Set to 0 for MPS compatibility
                shuffle=True,
                pin_memory=True
            )

        def val_dataloader(self):
            logger.info("Setting up validation dataloader")
            dataset = DNADataset("test_data", "valid")
            return torch.utils.data.DataLoader(
                dataset, 
                batch_size=val_batch_size, 
                num_workers=0,  # Set to 0 for MPS compatibility
                shuffle=True,
                pin_memory=True
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
    
    logger.info("Creating trainer")
    trainer = L.Trainer(
        max_epochs=epochs, 
        logger=tb_logger, 
        callbacks=[CheckPointer()],
        accelerator=DEVICE,  # Explicitly set accelerator to MPS
        devices=1,
        num_sanity_val_steps=1,  # Reduce sanity check steps
        limit_val_batches=1,  # Limit validation batches during sanity check
        limit_train_batches=1  # Limit training batches during sanity check
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
