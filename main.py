import lightning.pytorch as pl
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint
import click
import yaml
from pprint import pprint
import torch
from model import TrainableModel
from torch.utils.data import DataLoader
from data import FaceDataset
from utils import MBMaskCollator, to_simplenamespace

@click.command()
@click.option('--cfg', default='configs/in1k_vith14_ep300.yaml')
@click.option('--p', default=False, help='Enable Profiling')
def main(cfg, p):
    with open(cfg, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = to_simplenamespace(cfg)
    pprint(cfg, indent=4)

    collator = MBMaskCollator(
        input_size=cfg.data.crop_size,
        patch_size=cfg.mask.patch_size,
        pred_mask_scale=cfg.mask.pred_mask_scale,
        enc_mask_scale=cfg.mask.enc_mask_scale,
        aspect_ratio=cfg.mask.aspect_ratio,
        nenc=cfg.mask.num_enc_masks,
        npred=cfg.mask.num_pred_masks,
        allow_overlap=cfg.mask.allow_overlap,
        min_keep=cfg.mask.min_keep)

    train_loader = DataLoader(
        FaceDataset(cfg, train=True), 
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
        collate_fn=collator,
        persistent_workers=False,
        drop_last=True
    )

    val_loader = DataLoader(
        FaceDataset(cfg, train=False), 
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
        collate_fn=collator,
        persistent_workers=False,
        drop_last=True
    )
    
    # ipe = len(train_loader) # Without gradient accumulation
    ipe = len(train_loader) // (cfg.data.total_batch_size//cfg.data.batch_size) 

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='ijepa-{epoch:02d}-{val_loss:.2f}',
        save_last=True,
        save_top_k=20
    )
    # device_stats = DeviceStatsMonitor()

    callbacks = [checkpoint_callback]

    model = TrainableModel(cfg, ipe)

    trainer = pl.Trainer(
        accelerator='gpu',
        strategy=DDPStrategy(static_graph=False, process_group_backend="gloo"),
        devices=[0, 1],
        precision='16-mixed',
        callbacks=callbacks,
        max_epochs=cfg.optimization.epochs,
        accumulate_grad_batches=cfg.data.total_batch_size//cfg.data.batch_size,
        benchmark=True,
        profiler=p,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()