import pathlib
import pytorch_lightning as pl

from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from nn import VCModule
from dataset import VConDataModule
from utils import get_config


if __name__ == '__main__':
    config = get_config('config.yaml')

    seed_everything(config.seed)

    model = VCModule(config)
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
        gradient_clip_val=1.0,
        deterministic=True
    )
    trainer.fit(model=model, datamodule=vcon_dm)
