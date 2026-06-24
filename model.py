import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Import your custom bio-inspired primitives
from my_grid_model_core import AVSEModel, Config

class AVSE4BaselineModule(pl.LightningModule):
    def __init__(self, num_channels: int = 2, **kwargs):
        """
        Brain-Inspired AVSE Wrapper adaptive to multi-channel cluster waveforms.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Core Architecture 
        self.model = AVSEModel(
            model_type="av_modulation_v3",
            d_model=Config.D,
            num_layers=Config.N_LAYERS,
            pc_iterations=Config.PC_ITERS
        )
        
        # Primary objective: Spectrogram Reconstruction
        self.l1_loss = nn.L1Loss()

    def _parse_dictionary_input(self, data):
        """Extracts Tensors systematically using your pipeline's key-matching strategy."""
        if not isinstance(data, dict):
            noisy = data[0]
            video = data[1]
            clean = data[2] if len(data) > 2 else None
            vap_target = data[3] if len(data) > 3 else None
            return noisy, video, clean, vap_target

        audio_keys = ["noisy_audio", "noisy", "mix", "mixture", "audio_noisy"]
        noisy = None
        for key in audio_keys:
            if key in data:
                noisy = data[key].float() if not torch.is_complex(data[key]) else data[key]
                break
                
        video_keys = ["video", "visual", "video_frames", "v"]
        video = None
        for key in video_keys:
            if key in data:
                video = data[key].float()
                break

        clean_keys = ["clean", "target", "clean_audio", "s"]
        clean = None
        for key in clean_keys:
            if key in data:
                clean = data[key].float() if not torch.is_complex(data[key]) else data[key]
                break

        vap_target = data.get('vap_target', None)
        return noisy, video, clean, vap_target

    def _format_audio_to_4d(self, noisy, clean=None):
        """
        Dynamically transforms multi-channel waveforms or spectrograms 
        into standard 4D tensors [Batch, Channels(Real/Imag), Freq, Time].
        """
        window = torch.hann_window(Config.WIN, device=noisy.device)

        # Case 1: Raw 2D Waveform [Batch, Time]
        if noisy.dim() == 2:
            stft = torch.stft(noisy, n_fft=Config.N_FFT, hop_length=Config.HOP, 
                              win_length=Config.WIN, window=window, return_complex=True)
            noisy_spec = torch.stack([stft.real, stft.imag], dim=1)
            
            clean_spec = None
            if clean is not None:
                c_stft = torch.stft(clean, n_fft=Config.N_FFT, hop_length=Config.HOP, 
                                    win_length=Config.WIN, window=window, return_complex=True)
                clean_spec = torch.stack([c_stft.real, c_stft.imag], dim=1)
            return noisy_spec, clean_spec

        # Case 2: 3D Tensor
        if noisy.dim() == 3:
            # Subcase A: Complex Spectrogram [Batch, Freq, Time]
            if torch.is_complex(noisy):
                noisy_spec = torch.stack([noisy.real, noisy.imag], dim=1)
                clean_spec = torch.stack([clean.real, clean.imag], dim=1) if clean is not None else None
                return noisy_spec, clean_spec
            
            # Subcase B: Raw Multi-Channel Waveform [Batch, Channels, Time] (e.g., [8, 2, 184832])
            elif noisy.shape[1] <= 4 and noisy.shape[2] > 512:
                # Downmix channels to mono via mean to satisfy STFT dimension constraints
                noisy_mono = noisy.mean(dim=1)
                stft = torch.stft(noisy_mono, n_fft=Config.N_FFT, hop_length=Config.HOP, 
                                  win_length=Config.WIN, window=window, return_complex=True)
                noisy_spec = torch.stack([stft.real, stft.imag], dim=1)
                
                clean_spec = None
                if clean is not None:
                    clean_mono = clean.mean(dim=1) if clean.dim() == 3 else clean
                    c_stft = torch.stft(clean_mono, n_fft=Config.N_FFT, hop_length=Config.HOP, 
                                        win_length=Config.WIN, window=window, return_complex=True)
                    clean_spec = torch.stack([c_stft.real, c_stft.imag], dim=1)
                return noisy_spec, clean_spec
            
            # Subcase C: Real Spectrogram missing Channel axis [Batch, Freq, Time]
            else:
                noisy_spec = noisy.unsqueeze(1)
                clean_spec = clean.unsqueeze(1) if clean is not None else None
                return noisy_spec, clean_spec

        # Case 3: Already 4D [Batch, Channels, Freq, Time]
        return noisy, clean

    def forward(self, data):
        noisy, video, _, _ = self._parse_dictionary_input(data)
        noisy_spec, _ = self._format_audio_to_4d(noisy)
        
        if not self.training:
            return self.model(noisy_spec, video)
        return noisy_spec, video

    def training_step(self, batch, batch_idx):
        noisy, video, clean, vap_target = self._parse_dictionary_input(batch)
        noisy_spec, clean_spec = self._format_audio_to_4d(noisy, clean)

        # Forward execution through your bio-inspired streams
        enh, vap_p, nce_loss, mean_active, all_pred_errors = self.model(noisy_spec, video)

        # Multi-Task Losses
        loss_enh = self.l1_loss(enh, clean_spec)
        
        loss_pc = 0.0
        if all_pred_errors:
#            loss_pc = sum(torch.mean(err ** 2) for err in all_pred_errors) / len(all_pred_errors)
#            loss_pc = torch.mean(sum(err ** 2) for err in all_pred_errors) / len(all_pred_errors)


            loss_pc = sum(err for err in all_pred_errors) / len(all_pred_errors)
            loss_pc = loss_pc.mean()


        loss_vap = 0.0
        if vap_target is not None:
            if vap_p.shape != vap_target.shape:
                vap_target = F.interpolate(vap_target.unsqueeze(1), size=vap_p.shape[-1], mode='linear').squeeze(1)
            loss_vap = F.binary_cross_entropy(vap_p, vap_target.float())
        else:
            with torch.no_grad():
                energy = clean_spec.pow(2).sum(dim=[1, 2])
                simulated_target = (energy > energy.mean()).float()
                if vap_p.shape != simulated_target.shape:
                    simulated_target = F.interpolate(simulated_target.unsqueeze(1), size=vap_p.shape[-1], mode='linear').squeeze(1)
            loss_vap = F.binary_cross_entropy(vap_p, simulated_target)

        total_loss = loss_enh + (Config.W_NCE * nce_loss) + (Config.W_VAP * loss_vap) + (Config.W_PC * loss_pc)

        # Logging tracking metrics
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("metrics/enh_l1", loss_enh, sync_dist=True)
        self.log("metrics/nce_loss", nce_loss, sync_dist=True)
        self.log("metrics/pc_error", loss_pc, sync_dist=True)
        self.log("metrics/sparsity_firing_ratio", mean_active, prog_bar=True, sync_dist=True)


        return total_loss

    def validation_step(self, batch, batch_idx):
            noisy, video, clean, _ = self._parse_dictionary_input(batch)
            noisy_spec, clean_spec = self._format_audio_to_4d(noisy, clean)
    
            eval_dict = {"noisy": noisy_spec, "video": video}
            enh = self(eval_dict)
            
            val_loss = self.l1_loss(enh, clean_spec)
            
            # Log both keys to ensure full compatibility with your Hydra callbacks
            self.log("val_loss", val_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            self.log("val_loss_epoch", val_loss, prog_bar=True, sync_dist=True, on_epoch=True)
            return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=Config.LR, eps=Config.EPS)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.LR_MIN)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }