import sys
import pathlib

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

from dataset import VConDataModule
from utils import get_config, module_from_config

if __name__ == '__main__':

    args = sys.argv
    config = get_config(args[1])

    seed_everything(config.seed)

    model = module_from_config(config)
    vcon_dm = VConDataModule(config)

    model_dir = pathlib.Path(config.model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    save_fn = str(model_dir / 'vc_{epoch:03d}-{val_loss:.2f}')
    mc = ModelCheckpoint(filepath=save_fn, save_top_k=3)

    trainer = pl.Trainer(
        gpus=1,
        checkpoint_callback=mc,
        max_epochs=config.n_epochs,
        deterministic=True,
        precision=16
    )
    trainer.fit(model=model, datamodule=vcon_dm)
