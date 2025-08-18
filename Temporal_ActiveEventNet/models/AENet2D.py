from __future__ import print_function
import math
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.submodule import Gate_Dynamic_Interweave_Module
from models.submodule import feature_extraction, MobileV2_Residual, convbn, interweave_tensors, disparity_regression
import numpy as np


class hourglass2D(nn.Module):
    def __init__(self, in_channels):
        super(hourglass2D, self).__init__()

        self.expanse_ratio = 2

        self.conv1 = MobileV2_Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv3 = MobileV2_Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv4 = MobileV2_Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))

        self.redir1 = MobileV2_Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class Efficient_TemporalGRU(nn.Module):
    """
    Efficient Temporal GRU for event streams:
    1. Reduced Complexity: Reduce hidden size to 95% of original (less computation & memory)
    2. Gate Sharing: Use a single gate (shared weights) for both update/reset (reduces parameters)
    3. Simplified Activation: Use ReLU instead of tanh for candidate state (faster computation)
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, hidden_ratio=0.95, use_tanh=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_tanh = use_tanh

        self.reduced_hidden_size = int(hidden_size * hidden_ratio)

        self.gate_update = nn.Linear(input_size + self.reduced_hidden_size, self.reduced_hidden_size)
        self.gate_reset = nn.Linear(input_size + self.reduced_hidden_size, self.reduced_hidden_size)
        self.candidate = nn.Linear(input_size + self.reduced_hidden_size, self.reduced_hidden_size)

        if bidirectional:
            self.output_proj = nn.Conv2d(self.reduced_hidden_size * 2, input_size, 1, 1, 0)
        else:
            self.output_proj = nn.Conv2d(self.reduced_hidden_size, input_size, 1, 1, 0)

    def _run_direction(self, x_reshaped, T, reverse=False):
        h = torch.zeros(x_reshaped.size(0), self.reduced_hidden_size, device=x_reshaped.device, dtype=x_reshaped.dtype)
        outputs = []
        time_range = range(T - 1, -1, -1) if reverse else range(T)
        for t in time_range:
            inp = torch.cat([x_reshaped[:, t, :], h], dim=-1)
            update_gate = torch.sigmoid(self.gate_update(inp))
            reset_gate = torch.sigmoid(self.gate_reset(inp))
            candidate_inp = torch.cat([x_reshaped[:, t, :], h * reset_gate], dim=-1)
            if self.use_tanh:
                candidate = torch.tanh(self.candidate(candidate_inp))
            else:
                candidate = F.relu(self.candidate(candidate_inp))
            h = (1 - update_gate) * h + update_gate * candidate
            outputs.append(h)
        if reverse:
            outputs = outputs[::-1]
        return torch.stack(outputs, dim=1)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x_reshaped = x.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, C)
        if self.bidirectional:
            forward_out = self._run_direction(x_reshaped, T, reverse=False)[:, -1, :]
            backward_out = self._run_direction(x_reshaped, T, reverse=True)[:, 0, :]
            gru_out = torch.cat([forward_out, backward_out], dim=-1)
        else:
            gru_out = self._run_direction(x_reshaped, T, reverse=False)[:, -1, :]
        gru_out = gru_out.view(B, H, W, -1).permute(0, 3, 1, 2)
        output = self.output_proj(gru_out)
        return output


class AENet2D(nn.Module):
    def __init__(self, maxdisp, temporal_sequence_length=3):

        super(AENet2D, self).__init__()

        self.maxdisp = maxdisp
        self.temporal_sequence_length = temporal_sequence_length

        self.num_groups = 1

        self.volume_size = 48

        self.hg_size = 48

        self.dres_expanse_ratio = 3

        self.feature_extraction = feature_extraction(add_relus=True)

        self.preconv11 = nn.Sequential(convbn(320, 256, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(256, 128, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(128, 64, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 32, 1, 1, 0, 1))

        self.conv3d = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(8, 3, 3), stride=[8, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU(),
                                    nn.Conv3d(16, 32, kernel_size=(4, 3, 3), stride=[4, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(32),
                                    nn.ReLU(),
                                    nn.Conv3d(32, 16, kernel_size=(2, 3, 3), stride=[2, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU())

        self.volume11 = nn.Sequential(convbn(16, 1, 1, 1, 0, 1),
                                      nn.ReLU(inplace=True))

        self.dynamic_interweaver = Gate_Dynamic_Interweave_Module(32)  # 32

        self.dres0 = nn.Sequential(MobileV2_Residual(self.volume_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True),
                                   MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True),
                                   MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio))

        self.encoder_decoder1 = hourglass2D(self.hg_size)

        self.encoder_decoder2 = hourglass2D(self.hg_size)

        self.encoder_decoder3 = hourglass2D(self.hg_size)

        # Add temporal GRU module
        self.temporal_gru = Efficient_TemporalGRU(self.hg_size, self.hg_size // 2, num_layers=1, bidirectional=True, hidden_ratio=0.95, use_tanh=False)

        self.classif0 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1))
        self.classif1 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1))
        self.classif2 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1))
        self.classif3 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                                bias=False, dilation=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, L, R):
        # Check temporal sequence
        if len(L.shape) == 5:  # [B, T, C, H, W]
            B, T, C, H, W = L.shape
            L_reshaped = L.view(B * T, C, H, W)
            R_reshaped = R.view(B * T, C, H, W)

            features_L = self.feature_extraction(L_reshaped)  # [B*T, 320, H//4, W//4]
            features_R = self.feature_extraction(R_reshaped)

            featL = self.preconv11(features_L)  # [B*T, 32, H//4, W//4]
            featR = self.preconv11(features_R)

            featL = featL.view(B, T, -1, H // 4, W // 4)
            featR = featR.view(B, T, -1, H // 4, W // 4)

            temporal_features = []
            for t in range(T):
                featL_t = featL[:, t, :, :, :]  # [B, 32, H//4, W//4]
                featR_t = featR[:, t, :, :, :]

                # cost volume
                B_t, C_t, H_t, W_t = featL_t.shape
                volume = featL_t.new_zeros([B_t, self.num_groups, self.volume_size, H_t, W_t])
                for i in range(self.volume_size):
                    if i > 0:
                        x = self.dynamic_interweaver(featL_t[:, :, :, i:], featR_t[:, :, :, :-i])
                        x = torch.unsqueeze(x, 1)
                        x = self.conv3d(x)
                        x = torch.squeeze(x, 2)
                        x = self.volume11(x)
                        volume[:, :, i, :, i:] = x
                    else:
                        x = self.dynamic_interweaver(featL_t, featR_t)
                        x = torch.unsqueeze(x, 1)
                        x = self.conv3d(x)
                        x = torch.squeeze(x, 2)
                        x = self.volume11(x)
                        volume[:, :, i, :, :] = x

                volume = volume.contiguous()
                volume = torch.squeeze(volume, 1)

                cost0 = self.dres0(volume)
                cost0 = self.dres1(cost0) + cost0

                out1 = self.encoder_decoder1(cost0)
                temporal_features.append(out1)

            temporal_features = torch.stack(temporal_features, dim=1)  # [B, T, hg_size, H//4, W//4]
            gru_out1 = self.temporal_gru(temporal_features)  # [B, hg_size, H//4, W//4]
            out2 = self.encoder_decoder2(gru_out1)
            out3 = self.encoder_decoder3(out2)

        else:  # single event bin
            features_L = self.feature_extraction(L)
            features_R = self.feature_extraction(R)

            featL = self.preconv11(features_L)
            featR = self.preconv11(features_R)

            B, C, H, W = featL.shape
            volume = featL.new_zeros([B, self.num_groups, self.volume_size, H, W])
            for i in range(self.volume_size):
                if i > 0:
                    x = self.dynamic_interweaver(featL[:, :, :, i:], featR[:, :, :, :-i])
                    x = torch.unsqueeze(x, 1)
                    x = self.conv3d(x)
                    x = torch.squeeze(x, 2)
                    x = self.volume11(x)
                    volume[:, :, i, :, i:] = x
                else:
                    x = self.dynamic_interweaver(featL, featR)
                    x = torch.unsqueeze(x, 1)
                    x = self.conv3d(x)
                    x = torch.squeeze(x, 2)
                    x = self.volume11(x)
                    volume[:, :, i, :, :] = x

            volume = volume.contiguous()
            volume = torch.squeeze(volume, 1)

            cost0 = self.dres0(volume)
            cost0 = self.dres1(cost0) + cost0

            out1 = self.encoder_decoder1(cost0)
            out2 = self.encoder_decoder2(out1)
            out3 = self.encoder_decoder3(out2)

        if self.training:
            if len(L.shape) == 5:  # temporal sequence
                cost0 = self.classif0(gru_out1)
                cost1 = self.classif1(out2)
                cost2 = self.classif2(out3)
                cost3 = self.classif3(out3)
            else:  # single event bin
                cost0 = self.classif0(cost0)
                cost1 = self.classif1(out1)
                cost2 = self.classif2(out2)
                cost3 = self.classif3(out3)

            cost0 = torch.unsqueeze(cost0, 1)
            cost0 = F.interpolate(cost0, [self.maxdisp, L.size()[-2], L.size()[-1]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = torch.unsqueeze(cost1, 1)
            cost1 = F.interpolate(cost1, [self.maxdisp, L.size()[-2], L.size()[-1]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = torch.unsqueeze(cost2, 1)
            cost2 = F.interpolate(cost2, [self.maxdisp, L.size()[-2], L.size()[-1]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            cost3 = torch.unsqueeze(cost3, 1)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[-2], L.size()[-1]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)

            return [pred0, pred1, pred2, pred3]

        else:
            if len(L.shape) == 5:
                cost3 = self.classif3(out3)
            else:
                cost3 = self.classif3(out3)

            cost3 = torch.unsqueeze(cost3, 1)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[-2], L.size()[-1]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)

            return [pred3]

    def train_sample(self, imgL, imgR, disp_gt):
        # only save center event bin
        def get_center_frame(x):
            if isinstance(x, list):
                return [get_center_frame(xx) for xx in x]
            if torch.is_tensor(x) or isinstance(x, np.ndarray):
                if x.ndim == 5:  # [B, C, T, H, W]
                    center = x.shape[2] // 2
                    return x[:, :, center, :, :]
                elif x.ndim == 4:
                    # [C, T, H, W] or [B, T, H, W]
                    center = x.shape[1] // 2
                    return x[:, center, :, :]
                elif x.ndim == 3:
                    return x
            return x

        image_outputs = {
            "disp_est": get_center_frame(self.forward(imgL, imgR)),
            "disp_gt": get_center_frame(disp_gt),
            "imgL": get_center_frame(imgL),
            "imgR": get_center_frame(imgR)
        }
        return image_outputs
