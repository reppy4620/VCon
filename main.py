import pytorch_lightning as pl

from pytorch_lightning.utilities.seed import seed_everything

from nn import VCModule
from dataset import VConDataModule
from utils import get_config


if __name__ == '__main__':
    config = get_config('config.yaml')

    seed_everything(config.seed)

    model = VCModule(config)
    vcon_dm = VConDataModule(config)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config.n_epochs,
        gradient_clip_val=1.0,
        deterministic=True
    )
    trainer.fit(model=model, datamodule=vcon_dm)
