import torch
import torch.nn as nn

from my_grid_model_core import AVSEModel


class MyAVSE4Separator(nn.Module):
    def __init__(
        self,
        num_channels=2,
        checkpoint_path=None,
        n_fft=512,
        hop_length=160,
        win_length=512,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.model = AVSEModel(
            d_model=256,
            num_layers=6,
            pc_iterations=2,
        )

        self.register_buffer("window", torch.hann_window(win_length))

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state = ckpt.get("m", ckpt.get("state_dict", ckpt))
            self.model.load_state_dict(state, strict=False)

    def forward(self, mixture, visual):
        """
        Expected baseline-style inputs:
        mixture: [B, L], [B, C, L], or similar waveform tensor
        visual:  video tensor from AVSEC-4 dataset

        Returns:
        enhanced waveform
        """

        original_shape = mixture.shape

        if mixture.dim() == 3:
            # If binaural/multichannel, enhance each channel separately.
            b, c, l = mixture.shape
            mixture_flat = mixture.reshape(b * c, l)

            visual = self._prepare_video(visual)
            visual = visual.repeat_interleave(c, dim=0)

            enhanced = self._enhance_waveform(mixture_flat, visual, l)
            return enhanced.reshape(b, c, l)

        if mixture.dim() == 2:
            b, l = mixture.shape
            visual = self._prepare_video(visual)
            return self._enhance_waveform(mixture, visual, l)

        raise RuntimeError(f"Unsupported mixture shape: {original_shape}")

    def _enhance_waveform(self, mixture, visual, length):
        spec = torch.stft(
            mixture,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(mixture.device),
            return_complex=True,
        )

        spec_ri = torch.stack([spec.real, spec.imag], dim=1)

        enhanced_ri = self.model(spec_ri, visual)

        if isinstance(enhanced_ri, tuple):
            enhanced_ri = enhanced_ri[0]

        enhanced_complex = torch.complex(
            enhanced_ri[:, 0],
            enhanced_ri[:, 1],
        )

        enhanced = torch.istft(
            enhanced_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(mixture.device),
            length=length,
        )

        return enhanced

    def _prepare_video(self, visual):
        # Your VisualFrontend expects [B, 1, T, H, W].

        if visual.dim() == 4:
            visual = visual.unsqueeze(1)

        elif visual.dim() == 5:
            # If [B, T, C, H, W], convert to [B, C, T, H, W].
            if visual.shape[2] in (1, 3):
                visual = visual.permute(0, 2, 1, 3, 4)

        if visual.dim() == 5 and visual.shape[1] == 3:
            visual = visual.mean(dim=1, keepdim=True)

        visual = visual.float()

        if visual.max() > 2.0:
            visual = visual / 255.0

        return visual
