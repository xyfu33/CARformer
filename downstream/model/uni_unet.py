import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .uniformer import uniformer_small, PatchEmbed
from monai.networks.blocks import UnetrUpBlock, UnetOutBlock


class UniSegDecoder(nn.Module):
    def __init__(self, img_size: int, in_chans: int, cls_chans=0, segmentation=False):
        super().__init__()
        self.segmentation = segmentation
        self.decoder5 = UnetrUpBlock(
                    spatial_dims=3,
                    in_channels=512,
                    out_channels=320,
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name="instance",
                    res_block=True,
                )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=320,
            out_channels=128,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        self.proj1 = PatchEmbed(
                img_size=img_size, patch_size=3, in_chans=in_chans, embed_dim=64, stride=1, padding=1)    
        
        if not self.segmentation:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.cls_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(256, cls_chans if cls_chans > 0 else in_chans),
            )
        if cls_chans==0:
            self.out = UnetOutBlock(spatial_dims=3, in_channels=64, out_channels=in_chans)
        else:
            self.out = UnetOutBlock(spatial_dims=3, in_channels=64, out_channels=cls_chans)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x0, x1, x2, x3, x4):
        dec5 = self.decoder5(x4.permute(0,1,3,4,2), x3.permute(0,1,3,4,2))
        dec4 = self.decoder4(dec5, x2.permute(0,1,3,4,2)) #128
        dec3 = self.decoder3(dec4, x1.permute(0,1,3,4,2)) # convert to C,H,W,D
        if self.segmentation:
            x_proj = self.proj1(x0)
            dec2 = self.decoder2(dec3, x_proj.permute(0,1,3,4,2))
            x_out = self.out(dec2)
            return dec5, dec4, dec3, dec2, x_out
        x_up = self.up(dec3) # 64
        if hasattr(self, 'cls_head'):
            gap = self.pool(x4).flatten(1)  # [B, 512]
            logits = self.cls_head(gap)     # [B, num_classes]
            return dec5, dec4, dec3, x_up, logits
        x_out = self.out(x_up) # 4
        return dec5, dec4, dec3, x_up, x_out


class UniUnet_SC(nn.Module):
    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=64,
                 multi_scale=False, segmentation=True, supcon_proj_dim=128,
                 use_boq: bool = False, boq_num_queries: Optional[int] = None, boq_heads: int = 4,
                 boq_dropout: float = 0.0, boq_class_queries: bool = False, boq_use_diag_head: bool = True,
                 boq_blend_alpha: float = 0.3):
        super(UniUnet_SC, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.ms = multi_scale
        self.use_boq = bool(use_boq)
        # blend weight for BoQ logits (GAP head + alpha * BoQ head)
        self.boq_blend_alpha = float(boq_blend_alpha) if self.use_boq else 0.0

        self.encoder = uniformer_small(img_size=self.input_shape, in_chans=self.in_channels)
        self.decoder = UniSegDecoder(img_size=self.input_shape, in_chans=self.in_channels, cls_chans=self.out_channels, segmentation=segmentation)
        self.supcon_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, supcon_proj_dim),
        )
        # BoQ: learnable queries that attend to encoder tokens
        self.boq_num_queries = boq_num_queries if boq_num_queries is not None else self.out_channels
        self.boq_class_queries = bool(boq_class_queries)
        # Enforce class-aligned queries when BoQ is on and M matches K (ensures class-aware slots by default).
        if self.use_boq and (self.boq_num_queries == self.out_channels) and boq_use_diag_head:
            self.boq_class_queries = True
        self.boq_use_diag_head = bool(boq_use_diag_head)
        if self.use_boq:
            self.boq_queries = nn.Parameter(torch.randn(self.boq_num_queries, 512))
            self.boq_tok_norm = nn.LayerNorm(512)
            self.boq_q_norm = nn.LayerNorm(512)
            self.boq_attn = nn.MultiheadAttention(
                embed_dim=512,
                num_heads=max(1, int(boq_heads)),
                dropout=float(boq_dropout),
                batch_first=True,
            )
            self.boq_norm = nn.LayerNorm(512)
            # shared per-slot head: class-aware via slot index, stable via shared weights
            use_cls_head = self.boq_class_queries and (self.boq_num_queries == self.out_channels) and self.boq_use_diag_head
            if use_cls_head:
                self.boq_diag_active = True
                self.boq_cls_head = nn.Sequential(
                    nn.LayerNorm(512),
                    nn.Linear(512, 1),
                )
            else:
                self.boq_diag_active = False
                self.boq_cls_head = None
            if self.boq_use_diag_head and self.boq_class_queries and self.boq_cls_head is None:
                print("[boq] Class-aware head requested but disabled (num_queries != num_classes).")
        else:
            self.boq_tok_norm = None
            self.boq_q_norm = None

        if self.ms:
            self.ms_out = nn.ModuleList([
                nn.Conv3d(init_channels*2, self.out_channels, (1, 1, 1)),
                nn.Conv3d(init_channels*1, self.out_channels, (1, 1, 1)),
            ])
            self.up_out = nn.ModuleList([
                nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ])

    def _pool_feature_gap(self, x4):
        pooled_flat = torch.nn.functional.adaptive_avg_pool3d(x4, 1).flatten(1)
        return pooled_flat, None

    def _pool_feature_boq(self, x_enc):
        """
        Cross-attend learnable queries to encoder tokens. Returns:
        - z_global: pooled descriptor [B, C]
        - z_queries: per-query embeddings [B, M, C]
        - attn: attention weights (optional downstream use)
        - logits_diag: per-class logits if diagonal head is enabled, else None
        - div_loss: intra-sample query diversity penalty (scalar)
        """
        B, C, D, H, W = x_enc.shape
        tokens = x_enc.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # [B, T, C]
        queries = self.boq_queries.unsqueeze(0).expand(B, -1, -1)  # [B, M, C]
        tokens = self.boq_tok_norm(tokens)
        queries = self.boq_q_norm(queries)
        # keep per-head weights for possible logging then average to [B, M, T]
        z_queries, attn = self.boq_attn(queries, tokens, tokens, need_weights=True, average_attn_weights=False)  # [B, M, C], [B, h, M, T]
        if attn is not None:
            if attn.dim() == 4:
                attn = attn.mean(dim=1)
            attn = attn.detach()
        z_queries = self.boq_norm(z_queries)
        z_global = z_queries.mean(dim=1)
        logits_diag = None
        if self.boq_cls_head is not None and getattr(self, "boq_diag_active", True):
            logits_diag = self.boq_cls_head(z_queries).squeeze(-1)  # [B, K]
        # diversity loss (off-diagonal similarity penalty)
        zq = F.normalize(z_queries, dim=-1)
        sim = torch.matmul(zq, zq.transpose(1, 2))
        eye = torch.eye(sim.size(-1), device=sim.device, dtype=sim.dtype).unsqueeze(0)
        offdiag = sim - eye
        div_loss = (offdiag * offdiag).mean()
        return z_global, z_queries, attn, logits_diag, div_loss

    def _forward_heads(self, x4):
        """
        Compute GAP logits, BoQ logits (if enabled), blended logits, supcon projection.
        SupCon always uses GAP pooled features to avoid collapsing BoQ queries.
        """
        pooled_gap, g = self._pool_feature_gap(x4)
        logits_gap = self.decoder.cls_head(pooled_gap)

        if self.use_boq:
            pooled_boq, _, attn, logits_diag, div_loss = self._pool_feature_boq(x4)
            logits_boq = logits_diag if logits_diag is not None else self.decoder.cls_head(pooled_boq)
            logits = logits_gap + self.boq_blend_alpha * logits_boq
            aux = {"boq_div_loss": div_loss, "boq_attn": attn, "logits_gap": logits_gap.detach(), "logits_boq": logits_boq.detach()}
        else:
            logits = logits_gap
            aux = {}
        proj = self.supcon_head(pooled_gap)
        return logits, pooled_gap, proj, g, aux

    def forward(self, x, location=None):
        x0, x1, x2, x3, x4 = self.encoder(x)
        s5, s4, s3, s2, out = self.decoder(x0, x1, x2, x3, x4)

        if self.ms and self.training:
            out4 = self.up_out[0](self.ms_out[0](s4))
            out3 = self.up_out[1](self.ms_out[1](s3))
            out = [out4, out3, out]
            logits, pooled_gap, proj, g, aux = self._forward_heads(x4)
            return logits, x4, proj, g, aux

        logits, pooled_gap, proj, g, aux = self._forward_heads(x4)

        if self.training:
            return logits, x4, proj, g, aux
        return logits

    def forward_features(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)
        _, _, _, _, _ = self.decoder(x0, x1, x2, x3, x4)
        logits, pooled_gap, proj, g, aux = self._forward_heads(x4)
        return logits, x4, proj, g, aux
