import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import VConDataModule
from utils import get_config, module_from_config

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str)
    parser.add_argument('-d', '--data_dir', type=str)
    parser.add_argument('-m', '--model_dir', type=str)
    args = parser.parse_args()

    config = get_config(args.config_path)
    config.data_dir = args.data_dir
    config.model_dir = args.model_dir

    pl.seed_everything(config.seed)

    model = module_from_config(config)
    vcon_dm = VConDataModule(config)

    model_dir = pathlib.Path(config.model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    save_fn = str(model_dir / 'vc_{epoch:04d}-{val_loss:.6f}')
    mc = ModelCheckpoint(
        filepath=save_fn,
        save_last=True,
        monitor='val_loss',
        save_top_k=5
    )

    trainer = pl.Trainer(
        gpus=1,
        checkpoint_callback=mc,
        max_epochs=config.n_epochs,
        gradient_clip_val=5.0,
        deterministic=True
    )
    trainer.fit(model=model, datamodule=vcon_dm)
