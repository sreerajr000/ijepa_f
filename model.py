import lightning.pytorch as pl
import vit
import torch
import torch.nn.functional as F
from torch import optim
from utils import CosineWDSchedule, WarmupCosineSchedule, trunc_normal_, apply_masks, repeat_interleave_batch, flatten_namespace, iterate_modules
import copy

class TrainableModel(pl.LightningModule):
    def __init__(self, cfg, ipe):
        super().__init__()
        self.save_hyperparameters(flatten_namespace(cfg))
        self.cfg = cfg
        self.encoder = vit.__dict__[self.cfg.meta.model_name](
            img_size=[self.cfg.data.crop_size],
            patch_size=self.cfg.mask.patch_size)
        self.predictor = vit.__dict__['vit_predictor'](
            num_patches=self.encoder.patch_embed.num_patches,
            embed_dim=self.encoder.embed_dim,
            predictor_embed_dim=self.cfg.meta.pred_emb_dim,
            depth=self.cfg.meta.pred_depth,
            num_heads=self.encoder.num_heads)
        
        
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
        
        for m in self.encoder.modules():
            init_weights(m)

        for m in self.predictor.modules():
            init_weights(m)
        
        # Enable LORA here after loading pretrained weights, without it does not make sense
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.predictor.parameters():
            p.requires_grad = False
        self.encoder = iterate_modules(self.encoder, r=8)
        self.predictor = iterate_modules(self.predictor, r=8)
        
        
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        
        self.ipe = ipe
        self.momentum_scheduler = (self.cfg.optimization.ema[0] + i*(self.cfg.optimization.ema[1]-self.cfg.optimization.ema[0])/(ipe*self.cfg.optimization.epochs*self.cfg.optimization.ipe_scale)
                          for i in range(int(ipe*self.cfg.optimization.epochs*self.cfg.optimization.ipe_scale)+1))
        self.momentum_scheduler_index = 0

    def forward(self, batch):
        udata, masks_enc, masks_pred = batch
        imgs = udata[0]

        def forward_target():
            with torch.no_grad():
                h = self.target_encoder(imgs)
                h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                B = len(h)
                # -- create targets (masked regions of h)
                h = apply_masks(h, masks_pred)
                h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                return h
        
        def forward_context():
            z = self.encoder(imgs, masks_enc)
            z = self.predictor(z, masks_enc, masks_pred)
            return z

        h = forward_target()
        z = forward_context()
        return h, z
    
    def training_step(self, batch, batch_idx):
        h, z = self(batch)
        loss = F.smooth_l1_loss(z, h)
        self.log('loss', loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        h, z = self(batch)
        loss = F.smooth_l1_loss(z, h)
        self.log('val_loss', loss, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self):
        param_groups = [
            {
                'params': (p for n, p in self.encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        ]
        optimizer = optim.AdamW(param_groups, lr=self.cfg.optimization.start_lr, weight_decay=self.cfg.optimization.weight_decay)

        self.scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(self.cfg.optimization.warmup*self.ipe),
            start_lr=self.cfg.optimization.start_lr,
            ref_lr=self.cfg.optimization.lr,
            final_lr=self.cfg.optimization.final_lr,
            T_max=int(self.cfg.optimization.ipe_scale*self.cfg.optimization.epochs*self.ipe))
        
        self.wd_scheduler = CosineWDSchedule(
            optimizer,
            ref_wd=self.cfg.optimization.weight_decay,
            final_wd=self.cfg.optimization.final_weight_decay,
            T_max=int(self.cfg.optimization.ipe_scale*self.cfg.optimization.epochs*self.ipe))
        
        return {
            'optimizer': optimizer,
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        _new_lr = self.scheduler.step()
        _new_wd = self.wd_scheduler.step()
        self.log('lr', _new_lr, on_step=True)
        self.log('wd', _new_wd, on_step=True)
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            self.log('momentum', m, on_step=True)
            self.momentum_scheduler_index += 1
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['scheduler_state'] = self.scheduler.state_dict()
        checkpoint['wd_scheduler_state'] = self.wd_scheduler.state_dict()
        checkpoint['momentum_scheduler_index'] = self.momentum_scheduler_index


    def on_load_checkpoint(self, checkpoint):
        if 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        if 'wd_scheduler_state' in checkpoint:
            self.wd_scheduler.load_state_dict(checkpoint['wd_scheduler_state'])
        self.momentum_scheduler_index = checkpoint.get('momentum_scheduler_index', 0)
        for _ in range(self.momentum_scheduler_index):
            next(self.momentum_scheduler)

