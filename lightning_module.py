import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from my_grid_model_core import AVSEModel, Config

class BrainInspiredAVSELightning(pl.LightningModule):
    def __init__(self, config=Config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Instantiate your exact core model
        self.model = AVSEModel(
            model_type="av_modulation_v3",
            d_model=config.D,
            num_layers=config.N_LAYERS,
            pc_iterations=config.PC_ITERS
        )
        
        # Primary objective: Complex Spectrogram L1/MSE Loss
        self.l1_loss = nn.L1Loss()

    def forward(self, spec_ri, video):
        # Maps directly to the underlying core model forward pass
        return self.model(spec_ri, video)

    def training_step(self, batch, batch_idx):
        # Expecting batch format: [noisy_spectrogram, video_frames, clean_spectrogram, vap_targets]
        # Adjust unpack variables if your custom DataModule yields a different structure
        noisy_spec, video, clean_spec, *aux_targets = batch
        vap_target = aux_targets[0] if len(aux_targets) > 0 else None

        # 1. Run core forward pass (Training mode returns the 5-element tuple)
        enh, vap_p, nce_loss, mean_active, all_pred_errors = self.model(noisy_spec, video)

        # 2. Compute Primary Enhancement Loss
        loss_enh = self.l1_loss(enh, clean_spec)

        # 3. Compute Friston Predictive Coding Error Loss (Mean L2 Norm across all layers)
        loss_pc = 0.0
        if all_pred_errors:
            loss_pc = sum(torch.mean(err ** 2) for err in all_pred_errors) / len(all_pred_errors)

        # 4. Compute Voice Activity Projection (VAP) Binary Cross Entropy Loss
        loss_vap = 0.0
        if vap_target is not None:
            loss_vap = F.binary_cross_entropy(vap_p, vap_target)
        else:
            # Self-supervised/dummy target option if explicit VAP annotations aren't in your dataloader
            # Emulates a basic energy-based activity check from clean spectrum
            with torch.no_grad():
                energy = clean_spec.pow(2).sum(dim=[1, 2]).mean(dim=-1)
                simulated_target = (energy > energy.mean()).float()
            loss_vap = F.binary_cross_entropy(vap_p, simulated_target)

        # 5. Multi-task Compound Objective Optimization
        total_loss = (
            loss_enh + 
            self.config.W_NCE * nce_loss + 
            self.config.W_VAP * loss_vap + 
            self.config.W_PC * loss_pc
        )

        # Log metrics to your logger (TensorBoard/WandB) across distributed GPU nodes
        self.log("train/total_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("train/enh_loss", loss_enh, sync_dist=True)
        self.log("train/nce_loss", nce_loss, sync_dist=True)
        self.log("train/pc_error", loss_pc, sync_dist=True)
        self.log("train/sparsity_ratio", mean_active, prog_bar=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        noisy_spec, video, clean_spec, *_ = batch
        
        # 1. Core forward pass (Evaluation mode returns ONLY enhanced spectrogram tensor)
        enh = self(noisy_spec, video)
        
        # 2. Validate tracking loss
        val_loss = self.l1_loss(enh, clean_spec)
        self.log("val/loss", val_loss, prog_bar=True, sync_dist=True)
        
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config.LR, 
            eps=self.config.EPS
        )
        
        # Implementing a simple Cosine Annealing schedule matching your LR bounds
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config.EPOCHS, 
            eta_min=self.config.LR_MIN
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }