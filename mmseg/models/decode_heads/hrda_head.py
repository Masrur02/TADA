# Obtained from: https://github.com/lhoyer/HRDA
# Modifications:
# - Add return_logits flag
# - Update debug_output
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from copy import deepcopy
import torch.nn as nn
import torch
from torch.nn import functional as F

from ...core import add_prefix
from ...ops import resize as _resize
from .. import builder
from ..builder import HEADS
from ..segmentors.hrda_encoder_decoder import crop
from .decode_head import BaseDecodeHead
from mmseg.models.utils.trav import gt_to_traversability, CITYSCAPES_TRAV_PRIOR,FOREST_TRAV_PRIOR

def scale_box(box, scale):
    y1, y2, x1, x2 = box
    # assert y1 % scale == 0
    # assert y2 % scale == 0
    # assert x1 % scale == 0
    # assert x2 % scale == 0
    y1 = int(y1 / scale)
    y2 = int(y2 / scale)
    x1 = int(x1 / scale)
    x2 = int(x2 / scale)
    return y1, y2, x1, x2



def extract_high_freq_components(x, sigma=1.5):
    # Determine kernel size: 3 standard deviations rule
    kernel_size = int(2 * (3 * sigma) + 1)  # Kernel size: 3*sigma on both sides
    kernel_size = max(3, kernel_size)  # Ensure kernel size is at least 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create 1D Gaussian kernel
    gauss_kernel_1d = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    gauss_kernel_1d = torch.exp(-gauss_kernel_1d**2 / (2 * sigma**2))
    gauss_kernel_1d = gauss_kernel_1d / gauss_kernel_1d.sum()  # Normalize

    # Create 2D Gaussian kernel by outer product
    gauss_kernel_2d = gauss_kernel_1d[:, None] * gauss_kernel_1d[None, :]  # Outer product
    gauss_kernel_2d = gauss_kernel_2d.view(1, 1, kernel_size, kernel_size)  # [1, 1, K, K]

    # Repeat kernel for each channel
    gauss_kernel_2d = gauss_kernel_2d.repeat(x.size(1), 1, 1, 1)  # [C, 1, K, K]

    # Apply Gaussian blur (low-pass filter)
    blurred = F.conv2d(x, gauss_kernel_2d, padding=kernel_size // 2, groups=x.size(1))

    # Subtract low-pass (blurred) from original to get high-frequency components
    high_freq_components = x - blurred

    return high_freq_components / (high_freq_components.abs().max() + 1e-6) # Return absolute values to represent magnitudes

class GPCM(nn.Module):
    """
    Global Prior Context Module (GPCM)
    - GAP → global semantic descriptor
    - Spatial gate using sigmoid (NOT softmax)
    - gamma initialized to 0.2 for early effect
    """
    def __init__(self, in_ch, reduction=4):
        super().__init__()
        inter = max(in_ch // reduction, 1)

        # Global descriptor from GAP
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, inter, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, in_ch, kernel_size=1, bias=False)
        )

        # Spatial gating
        self.spatial_attn = nn.Conv2d(in_ch, 1, kernel_size=1, bias=True)

        # Learnable scaling for context injection
        self.gamma = nn.Parameter(torch.tensor(0.2))
        #self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        # === 1) Global pooled descriptor ===
        gp = F.adaptive_avg_pool2d(x, 1)       # [B, C, 1, 1]
        gp = self.fc(gp)                       # [B, C, 1, 1]

        # === 2) Spatial gate ===
        mask = torch.sigmoid(self.spatial_attn(x))  # [B, 1, H, W]

        # === 3) Inject context ===
        global_context = gp * mask             # [B, C, H, W]

        return x + self.gamma * global_context




class HRGLR(nn.Module):
    def __init__(self, in_ch):
        super(HR, self).__init__()

        self.hr_feature_attn = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.chan_attn = nn.Conv2d(in_ch, in_ch, 1)

        self.proj_raw  = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_edge = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_var  = nn.Conv2d(in_ch, in_ch, 1)

        self.GPCM = GPCM(in_ch)

        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

        self.refine_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hr_features):

        # Feature norm
        feature_norm_global = hr_features.mean(dim=1, keepdim=True)

        # High freq
        hr_high_freq = extract_high_freq_components(feature_norm_global, sigma=1.5)

        # Boundary attention
        feature_norm = feature_norm_global + hr_high_freq
        hr_feature_attention = torch.sigmoid(self.hr_feature_attn(feature_norm))
        hr_boundary_feat = hr_features * hr_feature_attention

        channel_attn = torch.sigmoid(self.chan_attn(hr_boundary_feat))
        hr_boundary_feat = hr_boundary_feat * channel_attn

        # Variance weighting
        mean = hr_features.mean(dim=1, keepdim=True)
        var = ((hr_features - mean)**2).mean(dim=1, keepdim=True)

        v_mean = var.mean(dim=(2,3), keepdim=True)
        v_std = var.std(dim=(2,3), keepdim=True).clamp_min(1e-6)
        var_norm = torch.tanh((var - v_mean) / v_std) * 0.5 + 0.5

        hr_var_feat = hr_features * var_norm

        # Projections
        r = self.proj_raw(hr_features)
        b = self.proj_edge(hr_boundary_feat)
        v = self.proj_var(hr_var_feat)

        # Context from GPCM
        context = self.GPCM(hr_features)

        r = r +context
        b = b +context
        v = v +context

        fused = ( b*v) + r
        out = self.fuse(fused)

        adaptive_mask = hr_high_freq * (0.3 + 0.7 * var_norm)

        out = hr_features + self.refine_scale * (out * adaptive_mask)
        

        return out




class TraversabilityHead(nn.Module):
    def __init__(self, num_classes=19, proj_channels=64, class_priors=None):
        """
        Traversability prediction head with context, uncertainty, and boundary awareness.
        
        Args:
            num_classes: Number of semantic classes.
            proj_channels: Channels for internal feature projection.
            class_priors: List or tensor of length num_classes (max traversability per class).
            hard_prior: If True → use argmax-based class cap; else probability-weighted soft cap.
        """
        super().__init__()
        

        # --- Base projection (context encoder) ---
        self.seg_proj = nn.Conv2d(num_classes, proj_channels, kernel_size=1)
        self.context_conv = nn.Sequential(
            nn.Conv2d(proj_channels, proj_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True)
        )

        # --- Modulation branches ---
        self.uncertainty_proj = nn.Conv2d(1, proj_channels, kernel_size=1)
        self.boundary_proj = nn.Conv2d(num_classes, proj_channels, kernel_size=1)

        # --- Fusion head ---
        self.fuse = nn.Sequential(
            nn.Conv2d(proj_channels, proj_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_channels, 1, kernel_size=1)
        )

        # --- Traversability priors (class-dependent caps) ---
        default_priors = [1.0] * num_classes if class_priors is None else class_priors
        self.register_buffer(
            "class_priors",
            torch.tensor(default_priors, dtype=torch.float32).view(1, num_classes, 1, 1)
        )

        # --- Tunable hyperparameters ---
        self.boundary_scale = nn.Parameter(torch.tensor(0.2))   # reduces traversability near boundaries
        self.uncertainty_scale = nn.Parameter(torch.tensor(0.3))  # modulates entropy strength

    def forward(self, seg_logits):
        """
        seg_logits: [B, num_classes, H, W]
        returns: traversability map [B, 1, H, W] in [0, 1]
        """
        B, C, H, W = seg_logits.shape

        # --- Context feature projection ---
        feat = self.context_conv(self.seg_proj(seg_logits))  # [B, 64, H, W]

        # --- Uncertainty estimation (entropy) ---
        probs = F.softmax(seg_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
        entropy_norm = torch.sigmoid(entropy * self.uncertainty_scale)  # higher = more uncertain

        # --- Boundary detection ---
        boundary_map = extract_high_freq_components(seg_logits).abs().mean(dim=1, keepdim=True)
        boundary_mask = torch.sigmoid(-self.boundary_scale * boundary_map)  # suppress near edges

        # --- Modulation (context stability mask) ---
        modulation = entropy_norm * boundary_mask
        mod_feat = feat * (1 - modulation)

        # --- Traversability prediction ---
        trav_raw = torch.sigmoid(self.fuse(mod_feat))  # [B,1,H,W]

        # --- Class-prior-based clipping ---
        
        class_idx = torch.argmax(seg_logits, dim=1, keepdim=True)  # [B,1,H,W]
        max_trav = torch.gather(self.class_priors.expand(B, -1, H, W), 1, class_idx)
        

        trav_map = trav_raw * max_trav.clamp(0, 1)
        return trav_map







class TraversabilityLoss(nn.Module):
    def __init__(self, zero_weight=0.1, prior_weight=1):
        super().__init__()
        self.zero_weight = zero_weight
        self.prior_weight = prior_weight
        self.mse = nn.MSELoss(reduction='none')
        self.l1  = nn.L1Loss(reduction='none')

    def forward(self, pred, gt, prior_map):
        gt = F.interpolate(gt, size=pred.shape[2:], mode='nearest')

        pos_mask  = (gt > 0).float()
        zero_mask = (gt == 0).float()

        # supervised
        loss_pos  = self.mse(pred, gt) * pos_mask
        loss_zero = self.l1(pred, gt) * zero_mask * self.zero_weight

        # prior consistency: only penalize when pred > prior
        over_prior = F.relu(pred - prior_map)
        loss_prior = (over_prior ** 2) * pos_mask

        total = (loss_pos.sum() + loss_zero.sum() + self.prior_weight * loss_prior.sum()) / gt.numel()
        return total


@HEADS.register_module()
class HRDAHead(BaseDecodeHead):

    def __init__(self,
                 single_scale_head,
                 lr_loss_weight=0,
                 hr_loss_weight=0,
                 scales=[1],
                 attention_embed_dim=256,
                 attention_classwise=True,
                 enable_hr_crop=False,
                 hr_slide_inference=True,
                 fixed_attention=None,
                 debug_output_attention=False,
                 **kwargs):
        head_cfg = deepcopy(kwargs)
        attn_cfg = deepcopy(kwargs)
        if single_scale_head == 'DAFormerHead':
            attn_cfg['channels'] = attention_embed_dim
            attn_cfg['decoder_params']['embed_dims'] = attention_embed_dim
            if attn_cfg['decoder_params']['fusion_cfg']['type'] == 'aspp':
                attn_cfg['decoder_params']['fusion_cfg'] = dict(
                    type='conv',
                    kernel_size=1,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=attn_cfg['decoder_params']['fusion_cfg']
                    ['norm_cfg'])
            kwargs['init_cfg'] = None
            kwargs['input_transform'] = 'multiple_select'
            self.os = 4
        elif single_scale_head == 'DLV2Head':
            kwargs['init_cfg'] = None
            kwargs.pop('dilations')
            kwargs['channels'] = 1
            self.os = 8
        else:
            raise NotImplementedError(single_scale_head)
        super(HRDAHead, self).__init__(**kwargs)
        del self.conv_seg
        del self.dropout

        head_cfg['type'] = single_scale_head
        self.head = builder.build_head(head_cfg)

        attn_cfg['type'] = single_scale_head
        if not attention_classwise:
            attn_cfg['num_classes'] = 1
        if fixed_attention is None:
            self.scale_attention = builder.build_head(attn_cfg)
        else:
            self.scale_attention = None
            self.fixed_attention = fixed_attention
        self.lr_loss_weight = lr_loss_weight
        self.hr_loss_weight = hr_loss_weight
        self.scales = scales
        self.enable_hr_crop = enable_hr_crop
        self.hr_crop_box = None
        self.hr_slide_inference = hr_slide_inference
        self.debug_output_attention = debug_output_attention
        self.hr_refiners = nn.ModuleList([
            HRGLR(in_ch=64),
            HRGLR(in_ch=128),
            HRGLR(in_ch=320),
            HRGLR(in_ch=512),
        ])

        self.trav_head = TraversabilityHead(
            num_classes=19,
            proj_channels=64,
            class_priors=CITYSCAPES_TRAV_PRIOR,
            
        )
        self.trav_criterion = TraversabilityLoss(zero_weight=0.1,prior_weight=1)


    def set_hr_crop_box(self, boxes):
        self.hr_crop_box = boxes

    def hr_crop_slice(self, scale):
        crop_y1, crop_y2, crop_x1, crop_x2 = scale_box(self.hr_crop_box, scale)
        return slice(crop_y1, crop_y2), slice(crop_x1, crop_x2)

    def resize(self, input, scale_factor):
        return _resize(
            input=input,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=self.align_corners)

    def decode_hr(self, inp, bs):
        if isinstance(inp, dict) and 'boxes' in inp.keys():
            features = inp['features']  # level, crop * bs, c, h, w
            boxes = inp['boxes']
            dev = features[0][0].device
            h_img, w_img = 0, 0
            for i in range(len(boxes)):
                boxes[i] = scale_box(boxes[i], self.os)
                y1, y2, x1, x2 = boxes[i]
                if h_img < y2:
                    h_img = y2
                if w_img < x2:
                    w_img = x2
            preds = torch.zeros((bs, self.num_classes, h_img, w_img),
                                device=dev)
            count_mat = torch.zeros((bs, 1, h_img, w_img), device=dev)

            crop_seg_logits = self.head(features)
            for i in range(len(boxes)):
                y1, y2, x1, x2 = boxes[i]
                crop_seg_logit = crop_seg_logits[i * bs:(i + 1) * bs]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1

            assert (count_mat == 0).sum() == 0
            preds = preds / count_mat
            return preds
        else:
            return self.head(inp)

    def get_scale_attention(self, inp):
        if self.scale_attention is not None:
            att = torch.sigmoid(self.scale_attention(inp))
        else:
            att = self.fixed_attention
        return att

    def forward(self, inputs):
        assert len(inputs) == 2
        hr_inp = inputs[1]
        hr_scale = self.scales[1]
        lr_inp = inputs[0]
        lr_sc_att_inp = inputs[0]  # separate var necessary for stack hr_fusion
        lr_scale = self.scales[0]
        batch_size = lr_inp[0].shape[0]
        assert lr_scale <= hr_scale

        has_crop = self.hr_crop_box is not None
        if has_crop:
            crop_y1, crop_y2, crop_x1, crop_x2 = self.hr_crop_box

        # print_log(f'lr_inp {[f.shape for f in lr_inp]}', 'mmseg')
        lr_seg = self.head(lr_inp)
        
        # print_log(f'lr_seg {lr_seg.shape}', 'mmseg')
        if isinstance(hr_inp, dict) and 'boxes' in hr_inp.keys():
            sparse_hr_features = []
            fe = hr_inp['features']  # List of high-resolution features
              # Should print 4
          
            for i, feature in enumerate(fe):
                refined_feature = self.hr_refiners[i](feature)  # apply stage-specific refiner
                sparse_hr_features.append(refined_feature)
            hr_inp['features'] = sparse_hr_features
        
        else:
            sparse_hr_features = []
        
            for i, feature in enumerate(hr_inp):
                refined_feature = self.hr_refiners[i](feature)  # apply stage-specific refiner
                sparse_hr_features.append(refined_feature)
            hr_inp= sparse_hr_features

        hr_seg = self.decode_hr(hr_inp, batch_size)

        att = self.get_scale_attention(lr_sc_att_inp)
        if has_crop:
            mask = lr_seg.new_zeros([lr_seg.shape[0], 1, *lr_seg.shape[2:]])
            sc_os = self.os / lr_scale
            slc = self.hr_crop_slice(sc_os)
            mask[:, :, slc[0], slc[1]] = 1
            att = att * mask
        # print_log(f'att {att.shape}', 'mmseg')
        lr_seg = (1 - att) * lr_seg
        # print_log(f'scaled lr_seg {lr_seg.shape}', 'mmseg')
        up_lr_seg = self.resize(lr_seg, hr_scale / lr_scale)
        if torch.is_tensor(att):
            att = self.resize(att, hr_scale / lr_scale)

        if has_crop:
            hr_seg_inserted = torch.zeros_like(up_lr_seg)
            slc = self.hr_crop_slice(self.os)
            hr_seg_inserted[:, :, slc[0], slc[1]] = hr_seg
        else:
            hr_seg_inserted = hr_seg

        fused_seg = att * hr_seg_inserted + up_lr_seg
        trav= self.trav_head(fused_seg)
        


        if self.debug_output_attention:
            att = torch.sum(
                att * torch.softmax(fused_seg, dim=1), dim=1, keepdim=True)
            return att, None, None

        if self.debug:
            self.debug_output.update({
                'High Res':
                torch.max(hr_seg, dim=1)[1].detach().cpu().numpy(),
                'High Res Inserted':
                torch.max(hr_seg_inserted, dim=1)[1].detach().cpu().numpy(),
                'Low Res':
                torch.max(lr_seg, dim=1)[1].detach().cpu().numpy(),
                'Fused':
                torch.max(fused_seg, dim=1)[1].detach().cpu().numpy(),
            })
            if torch.is_tensor(att):
                self.debug_output['Attention'] = torch.sum(
                    att * torch.softmax(fused_seg, dim=1), dim=1,
                    keepdim=True).detach().cpu().numpy()

        return fused_seg, lr_seg, hr_seg,trav

    def reset_crop(self):
        del self.hr_crop_box
        self.hr_crop_box = None

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None,
                      return_logits=False):
        """Forward function for training."""
        if self.enable_hr_crop:
            assert self.hr_crop_box is not None
        
        
        
        trav_gt = gt_to_traversability(gt_semantic_seg, "city")
        
        
        *seg_logits, trav = self.forward(inputs)
        
        argmax_idx = seg_logits[0].argmax(dim=1, keepdim=True)  # [B,1,H,W]
        device = seg_logits[0].device
        prior_map = CITYSCAPES_TRAV_PRIOR.to(device)[argmax_idx].unsqueeze(1).float()
        loss_trav=self.trav_criterion(trav,trav_gt, prior_map)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        losses['loss_seg'] = losses['loss_seg']+ 0.3 * loss_trav
      
        self.reset_crop()
        
        return losses,seg_logits[0],trav,trav_gt

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``fused_seg`` is used."""
        *seg_logits, trav = self.forward(inputs)
        

        return seg_logits[0], trav

    def losses(self, seg_logit, seg_label, seg_weight=None):
        """Compute losses."""
        fused_seg, lr_seg, hr_seg = seg_logit
        loss = super(HRDAHead, self).losses(fused_seg, seg_label, seg_weight)
        if self.hr_loss_weight == 0 and self.lr_loss_weight == 0:
            return loss

        if self.lr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(lr_seg, seg_label,
                                                 seg_weight), 'lr'))
        if self.hr_loss_weight > 0 and self.enable_hr_crop:
            cropped_seg_label = crop(seg_label, self.hr_crop_box)
            if seg_weight is not None:
                cropped_seg_weight = crop(seg_weight, self.hr_crop_box)
            else:
                cropped_seg_weight = seg_weight
            if self.debug:
                self.debug_output['Cropped GT'] = \
                    cropped_seg_label.squeeze(1).detach().cpu().numpy()
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(hr_seg, cropped_seg_label,
                                                 cropped_seg_weight), 'hr'))
        elif self.hr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(hr_seg, seg_label,
                                                 seg_weight), 'hr'))
        loss['loss_seg'] *= (1 - self.lr_loss_weight - self.hr_loss_weight)
        if self.lr_loss_weight > 0:
            loss['lr.loss_seg'] *= self.lr_loss_weight
        if self.hr_loss_weight > 0:
            loss['hr.loss_seg'] *= self.hr_loss_weight

        if self.debug:
            self.debug_output['GT'] = \
                seg_label.squeeze(1).detach().cpu().numpy()
            # Remove debug output from cross entropy loss
            self.debug_output.pop('Seg. Pred.', None)
            self.debug_output.pop('Seg. GT', None)

        return loss
