import inspect
from datetime import datetime
from pathlib import Path

import torch

from .bsq import BSQDNA
from .data import DNADataset


def train(epochs: int = 5, batch_size: int = 64):
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger

    class PatchTrainer(L.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def training_step(self, x, batch_idx):
            x_hat, additional_losses = self.model(x)
            loss = torch.nn.functional.mse_loss(x_hat, x)
            self.log("train/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"train/{k}", v)
            return loss + sum(additional_losses.values())

        def validation_step(self, x, batch_idx):
            with torch.no_grad():
                x_hat, additional_losses = self.model(x)
                loss = torch.nn.functional.mse_loss(x_hat, x)
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
            return torch.optim.AdamW(self.parameters(), lr=1e-3)

        def train_dataloader(self):
            dataset = DNADataset("data", "train")
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

        def val_dataloader(self):
            dataset = DNADataset("data", "valid")
            return torch.utils.data.DataLoader(dataset, batch_size=4096, num_workers=4, shuffle=True)

    class CheckPointer(L.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            fn = Path(f"checkpoints/{timestamp}_{model_name}.pth")
            fn.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model, fn)
            torch.save(model, Path(__file__).parent / f"{model_name}.pth")

    # Create the model
    model_name = "BSQDNA"
    model = BSQDNA()

    # Create the lightning model
    l_model = PatchTrainer(model)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = TensorBoardLogger("logs", name=f"{timestamp}_{model_name}")
    trainer = L.Trainer(max_epochs=epochs, logger=logger, callbacks=[CheckPointer()])
    trainer.fit(
        model=l_model,
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(train)
