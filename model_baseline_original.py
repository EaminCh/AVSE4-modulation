import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from pytorch_lightning import LightningModule
from speechbrain.nnet.losses import cal_si_snr
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import VisualFrontend

EPS = 1e-8


def overlap_and_add(signal, frame_step):
    """Taken from https://github.com/kaituoxu/Conv-TasNet/blob/master/src/utils.py
    Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    Example
    -------
    >>> signal = torch.randn(5, 20)
    >>> overlapped = overlap_and_add(signal, 20)
    >>> overlapped.shape
    torch.Size([100])
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(
        frame_length, frame_step
    )  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(
        0, subframes_per_frame, subframe_step
    )

    # frame_old = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.clone().detach().to(signal.device.type)
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(
        *outer_dimensions, output_subframes, subframe_length
    )
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class avse4_separator(nn.Module):
    def __init__(self, N=256, L=40, B=256, H=512, P=3, X=8, R=4, C=2, num_channels=2):
        super(avse4_separator, self).__init__()

        self.encoder = Encoder(L, N, num_channels)
        self.separator = TemporalConvNet(N, B, H, P, X, R, C, num_channels)
        self.decoder = Decoder(N, L)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture, visual):
        mixture_w = self.encoder(mixture)
        est_mask = self.separator(mixture_w, visual)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source


class Encoder(nn.Module):
    def __init__(self, L, N, num_channels=2):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(num_channels,
                                  N,
                                  kernel_size=L,
                                  stride=L // 2,
                                  bias=False)

    def forward(self, mixture):
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        est_source = torch.unsqueeze(mixture_w, 1) * est_mask 
        est_source = torch.transpose(est_source, 2, 3)  # [M,  K, N]
        est_source = self.basis_signals(est_source)  # [M,  K, L]
        est_source = overlap_and_add(est_source, self.L // 2)  # M x C x T
        return est_source


class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, num_channels=2):
        super(TemporalConvNet, self).__init__()
        self.num_channels = num_channels
        self.C = C
        self.layer_norm = ChannelWiseLayerNorm(N)
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)

        # Audio TCN
        tcn_blocks = []
        tcn_blocks += [nn.Conv1d(B * 2, B, 1, bias=False)]
        for x in range(X):
            dilation = 2**x
            padding = (P - 1) * dilation // 2
            tcn_blocks += [
                TemporalBlock(B,
                              H,
                              P,
                              stride=1,
                              padding=padding,
                              dilation=dilation)
            ]
        self.tcn = _clones(nn.Sequential(*tcn_blocks), R)

        # visual blocks
        ve_blocks = []
        for x in range(5):
            ve_blocks += [VisualConv1D()]
        self.visual_conv = nn.Sequential(*ve_blocks)

        # Audio and visual seprated layers before concatenation
        self.ve_conv1x1 = _clones(nn.Conv1d(512, B, 1, bias=False), R)

        # Mask generation layer
        self.mask_conv1x1 = nn.Conv1d(B, N*num_channels, 1, bias=False)

    def forward(self, x, visual):
        visual = visual.transpose(1, 2)
        visual = self.visual_conv(visual)

        x = self.layer_norm(x)
        x = self.bottleneck_conv1x1(x)

        mixture = x

        batch, B, K = x.size()

        for i in range(len(self.tcn)):
            v = self.ve_conv1x1[i](visual)
            v = F.interpolate(v, (32 * v.size()[-1]), mode='linear')
            v = F.pad(v, (0, K - v.size()[-1]))
            x = torch.cat((x, v), 1)
            x = self.tcn[i](x)

        x = self.mask_conv1x1(x)
        x = F.relu(x)
        return x.reshape(batch, self.num_channels, B, K)

class VisualConv1D(nn.Module):
    def __init__(self):
        super(VisualConv1D, self).__init__()
        relu = nn.ReLU()
        norm_1 = nn.BatchNorm1d(512)
        dsconv = nn.Conv1d(512,
                           512,
                           3,
                           stride=1,
                           padding=1,
                           dilation=1,
                           groups=512,
                           bias=False)
        prelu = nn.PReLU()
        norm_2 = nn.BatchNorm1d(512)
        pw_conv = nn.Conv1d(512, 512, 1, bias=False)

        self.net = nn.Sequential(relu, norm_1, dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out + x


class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size,
                                               1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2,
                                                keepdim=True)  #[M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1,
                                            keepdim=True).mean(dim=2,
                                                               keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation):
        super(TemporalBlock, self).__init__()
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = GlobalLayerNorm(out_channels)
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):

        residual = x
        out = self.net(x)
        return out + residual  # look like w/o F.relu is better than w/ F.relu


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation):
        super(DepthwiseSeparableConv, self).__init__()
        depthwise_conv = nn.Conv1d(in_channels,
                                   in_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels,
                                   bias=False)

        prelu = nn.PReLU()
        norm = GlobalLayerNorm(in_channels)
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
        return self.net(x)

class AVSE4BaselineModule(LightningModule):
    def __init__(
            self, lr=0.0001,
            a_only=False, val_dataset=None, loss=None, batch_size=4, frontend_ckpt_path=None, num_channels=2
    ):
        assert num_channels in [1, 2], "Only mono and binaural audio are supported"
        super(AVSE4BaselineModule, self).__init__()
        self.lr = lr
        self.val_dataset = val_dataset
        self.a_only = a_only
        self.loss_name = loss
        self.loss = cal_si_snr
        self.model = avse4_separator(num_channels=num_channels)
        self.batch_size = batch_size
        self.visual_frontend = VisualFrontend()
        self.num_channels = num_channels
        if frontend_ckpt_path is not None:
            self.visual_frontend.load_state_dict(torch.load(frontend_ckpt_path, map_location=self.device,
                                                            weights_only=True))
        self.save_hyperparameters()

    def forward(self, data):
        """ Processes the input tensor x and returns an output tensor."""
        noisy = data["noisy_audio"].float()
        video = data["vis_feat"].float()
        visual_feat = self.visual_frontend(video)
        return self.model(noisy, visual_feat)

    def training_step(self, batch_inp, batch_idx):
        loss = self.cal_loss(batch_inp)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=self.batch_size)
        return loss

    def validation_step(self, batch_inp, batch_idx):
        loss = self.cal_loss(batch_inp)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=self.batch_size)
        return loss
    
    def enhance(self, data):
        if isinstance(data["noisy_audio"], np.ndarray):
            inputs = {
                "noisy_audio": torch.from_numpy(data["noisy_audio"][np.newaxis, ...]).to(self.device)}
            if not self.a_only:
                inputs["vis_feat"] = torch.from_numpy(data["vis_feat"][np.newaxis, ...]).to(self.device)
            clean = data["clean"]
            noisy = data["noisy_audio"]
        else:
            inputs = {
                "noisy_audio": data["noisy_audio"].to(self.device)}
            if not self.a_only:
                inputs["vis_feat"] = data["vis_feat"].to(self.device)
            clean = data["clean"].numpy()
            noisy = data["noisy_audio"].numpy()
        estimated_audio = self(inputs).cpu().numpy()[0]
        estimated_audio /= np.max(np.abs(estimated_audio))
        return clean, noisy, estimated_audio

    def on_train_epoch_end(self, *args, **kwargs):
        if self.val_dataset is not None:
            with torch.no_grad():
                tensorboard = self.logger.experiment
                for index in range(5):
                    rand_int = random.randint(0, len(self.val_dataset))
                    data = self.val_dataset[rand_int]
                    clean, noisy, estimated_audio = self.enhance(data)

                    tensorboard.add_audio("{}/{}_noisy".format(self.current_epoch, index),
                                          noisy.T, global_step=self.current_epoch,
                                          sample_rate=16000)
                    tensorboard.add_audio("{}/{}_enhanced".format(self.current_epoch, index),
                                          estimated_audio.T,
                                          global_step=self.current_epoch,
                                          sample_rate=16000)
                    tensorboard.add_audio("{}/{}_clean".format(self.current_epoch, index),
                                          clean.T, global_step=self.current_epoch,
                                          sample_rate=16000)

    def cal_loss(self, batch_inp):
        mask = batch_inp["clean"] # [B, C, T]
        pred_mask = self(batch_inp) # [B, C, T]
        
        # [T, B, C]
        mask = mask.permute(2, 0, 1)
        pred_mask = pred_mask.permute(2, 0, 1)
        loss = cal_si_snr(mask, pred_mask)
        loss[loss < -30] = -30
        return torch.mean(loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.8, patience=3, ),
                "monitor": "val_loss_epoch",
            },
        }
if __name__ == '__main__':
    num_channels = 2
    model = AVSE4BaselineModule(num_channels=num_channels)
    print(model({"noisy_audio":torch.randn(4, num_channels, 16000),
                 "vis_feat": torch.randn(4, 1, 25, 112, 112)}).shape)