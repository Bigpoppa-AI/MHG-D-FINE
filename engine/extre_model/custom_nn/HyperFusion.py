
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple

from engine.extre_model.custom_nn.conv import Conv
from engine.extre_model.custom_nn.block import C3, C2f

__all__ = ['HyperFusion', 'DSConv_YOLO13', 'DSC3k2', 'DownsampleConv', 'FullPAD_Tunnel']

class DSConv_YOLO13(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):
        super().__init__()
        if p is None:
            p = (d * (k - 1)) // 2
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=k, stride=s,
                            padding=p, dilation=d, groups=c_in, bias=bias)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class DSBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv_YOLO13(c1, c_, k1, s=1, p=None, d=1)
        self.cv2 = DSConv_YOLO13(c_, c2, k2, s=1, p=None, d=d2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class DSC3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__(c1, c2, n, shortcut, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(DSBottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)))


class DSC3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, shortcut=False):
        super().__init__(c1, c2, n, shortcut, e)
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, e=1.0) if c3k
            else DSBottleneck(self.c, self.c, shortcut, e=1.0)
            for _ in range(n)
        )


class DownsampleConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = Conv(c1, c2, 3, 2)

    def forward(self, x):
        return self.conv(x)


class FullPAD_Tunnel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            if len(x) == 2:
                return x[0] + self.gate * x[1]
            elif len(x) == 1:
                return x[0]
            raise ValueError(f"FullPAD_Tunnel expects 1 or 2 inputs, got {len(x)}")
        return x

class DynamicCapacityPredictor(nn.Module):
    def __init__(self,
                 node_dim,
                 num_scales=3,
                 capacity_min=0.3,   
                 capacity_max=1.0,   
                 capacity_target=0.6):  
        super().__init__()
        self.num_scales = num_scales
        self.capacity_min = capacity_min
        self.capacity_max = capacity_max
        self.capacity_target = capacity_target
        
        self.complexity_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * node_dim, node_dim // 2),
                nn.LayerNorm(node_dim // 2),
                nn.GELU(),
                nn.Linear(node_dim // 2, node_dim // 4)
            ) for _ in range(num_scales)
        ])
        
        self.capacity_head = nn.Sequential(
            nn.Linear((node_dim // 4) * num_scales, node_dim // 4),
            nn.GELU(),
            nn.Linear(node_dim // 4, 1)
        )
        
        self.capacity_bias = nn.Parameter(
            torch.tensor(self._inv_sigmoid(capacity_target))
        )
        
        self.register_buffer('ema_capacity', torch.tensor(capacity_target))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.long))
        self.ema_momentum = 0.9
    
    @staticmethod
    def _inv_sigmoid(x):
        return math.log(x / (1 - x + 1e-8) + 1e-8)
    
    def _extract_complexity_features(self, x, scale_idx):
        B, C, H, W = x.shape
        
       
        feat_mean = x.mean(dim=(2, 3))  
        feat_max = x.amax(dim=(2, 3))   
        feat_var = x.var(dim=(2, 3))    
        
        stats = torch.cat([feat_mean, feat_max, feat_var], dim=-1)  # (B, 3C)
        complexity = self.complexity_encoders[scale_idx](stats)
        return complexity
    
    def forward(self, x_list):
        complexities = [
            self._extract_complexity_features(x, i)
            for i, x in enumerate(x_list)
        ]
        
       
        combined = torch.cat(complexities, dim=-1)  # (B, 3 * dim//4)
        
        capacity_logits = self.capacity_head(combined).mean() + self.capacity_bias
        
        capacity_raw = torch.sigmoid(capacity_logits)
        capacity_ratio = self.capacity_min + (self.capacity_max - self.capacity_min) * capacity_raw
        
        if self.training:
            with torch.no_grad():
                momentum = 0.5 if self.num_updates < 100 else self.ema_momentum
                self.ema_capacity = momentum * self.ema_capacity + (1 - momentum) * capacity_ratio.detach()
                self.num_updates += 1
        
        capacity_loss = (capacity_ratio - self.capacity_target).pow(2)
        
        return capacity_ratio, capacity_loss



class DensityAwareAllocator(nn.Module):

    def __init__(self,
                 node_dim,
                 max_pool_size=32,      
                 num_scales=3,
                 ema_momentum=0.9,
                 min_hyperedges=2):
        super().__init__()
        self.max_pool_size = max_pool_size
        self.num_scales = num_scales
        self.min_hyperedges = min_hyperedges
        self.ema_momentum = ema_momentum
        
        self.density_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(node_dim, node_dim // 4, 1),
                nn.GroupNorm(min(8, node_dim // 4), node_dim // 4),
                nn.GELU(),
                nn.Conv2d(node_dim // 4, 1, 1)
            ) for _ in range(num_scales)
        ])
        
        self.scale_prior_logits = nn.Parameter(
            torch.tensor([0.5, 0.0, -0.3])
        )
        
        self.register_buffer('ema_ratios', torch.ones(num_scales) / num_scales)
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.long))
    
    def estimate_density(self, x, scale_idx):
        density_logits = self.density_estimators[scale_idx](x)
        density_map = torch.sigmoid(density_logits)
        B = x.shape[0]
        flat = density_map.view(B, -1)
        k = max(1, flat.shape[1] // 20)
        topk_vals, _ = flat.topk(k, dim=1)
        density_score = topk_vals.mean(dim=1)
        return density_map, density_score
    
    def compute_allocation(self, x_list, effective_pool_size):
        density_scores = []
        density_maps = []
        for i, x in enumerate(x_list):
            d_map, d_score = self.estimate_density(x, i)
            density_scores.append(d_score)
            density_maps.append(d_map)
        
        density_scores = torch.stack(density_scores, dim=1)
        scale_prior = F.softmax(self.scale_prior_logits, dim=0)
        weighted = density_scores * scale_prior.unsqueeze(0)
        batch_ratios = weighted.mean(dim=0)
        batch_ratios = batch_ratios / (batch_ratios.sum() + 1e-8)
        
        if self.training:
            with torch.no_grad():
                momentum = 0.5 if self.num_updates < 100 else self.ema_momentum
                self.ema_ratios = momentum * self.ema_ratios + (1 - momentum) * batch_ratios.detach()
                self.num_updates += 1
            final_ratios = 0.5 * batch_ratios + 0.5 * self.ema_ratios
        else:
            final_ratios = self.ema_ratios
        
        num_per_scale = self._ratios_to_counts(final_ratios, effective_pool_size)
        return num_per_scale, final_ratios, density_maps
    
    def _ratios_to_counts(self, ratios, total):
        total = max(total, self.min_hyperedges * self.num_scales)
        remaining = total - self.min_hyperedges * self.num_scales
        
        raw = ratios * remaining
        counts = raw.round().long() + self.min_hyperedges
        
        diff = total - counts.sum().item()
        if diff != 0:
            if diff > 0:
                idx = ratios.argmax()
                counts[idx] += diff
            else:
                for _ in range(abs(diff)):
                    idx = counts.argmax()
                    if counts[idx] > self.min_hyperedges:
                        counts[idx] -= 1
        return counts.tolist()
    
    def forward(self, x_list, effective_pool_size):
        return self.compute_allocation(x_list, effective_pool_size)

class DynamicMultiScaleHyperedgeGen(nn.Module):
    def __init__(self,
                 node_dim,
                 max_pool_size=32,
                 num_heads=4,
                 dropout=0.1,
                 gumbel_tau=1.0):
        super().__init__()
        self.node_dim = node_dim
        self.max_pool_size = max_pool_size
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.gumbel_tau = gumbel_tau
        
        self.prototype_pool = nn.Parameter(
            torch.randn(max_pool_size, node_dim) * 0.02
        )
        
        self.prototype_importance = nn.Parameter(
            torch.zeros(max_pool_size)
        )
        
        self.context_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * node_dim, node_dim // 2),
                nn.LayerNorm(node_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(node_dim // 2, max_pool_size * node_dim)
            ) for _ in range(3)
        ])
        
        self.query_proj = nn.Linear(node_dim, node_dim)
        self.cross_scale_fusion = CrossScaleHyperedgeFusion(node_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)
    
    def _select_prototypes(self, all_prototypes, num_active, start_idx):
        B = all_prototypes.shape[0]
        
        end_idx = start_idx + num_active
        end_idx = min(end_idx, self.max_pool_size)
        
        importance_scores = self.prototype_importance[start_idx:end_idx]
        
        if self.training:
            noise = -torch.log(-torch.log(torch.rand_like(importance_scores) + 1e-8) + 1e-8)
            gates = torch.sigmoid((importance_scores + noise) / self.gumbel_tau)
        else:
            gates = torch.sigmoid(importance_scores)
        
        active_prototypes = all_prototypes[:, start_idx:end_idx, :]
        active_prototypes = active_prototypes * gates.view(1, -1, 1)
        
        return active_prototypes
    
    def _generate_scale_prototypes(self, X, scale_idx, num_active, start_idx):
        B, N, D = X.shape
        
        avg_context = X.mean(dim=1)
        max_context, _ = X.max(dim=1)
        context_cat = torch.cat([avg_context, max_context], dim=-1)
        
        offset = self.context_nets[scale_idx](context_cat).view(B, self.max_pool_size, D)
        all_prototypes = self.prototype_pool.unsqueeze(0) + offset
        
        prototypes = self._select_prototypes(all_prototypes, num_active, start_idx)
        return prototypes
    
    def _compute_participation(self, X, prototypes, scale_weight=1.0):
        B, N, D = X.shape
        M = prototypes.shape[1]
        
        X_proj = self.query_proj(X)
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, M, self.head_dim).transpose(1, 2)
        
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling
        logits = logits.view(B, self.num_heads, N, M).mean(dim=1)
        logits = logits * scale_weight
        logits = self.dropout(logits)
        
        return F.softmax(logits, dim=1)
    
    def forward(self, X_multi_scale, num_per_scale, ratios):
        X_small, X_medium, X_large = X_multi_scale
        
        start_indices = [0]
        for i in range(len(num_per_scale) - 1):
            start_indices.append(start_indices[-1] + num_per_scale[i])
        
        for i in range(len(start_indices)):
            if start_indices[i] + num_per_scale[i] > self.max_pool_size:
                num_per_scale[i] = max(self.max_pool_size - start_indices[i], 1)
        
        scale_weights = 1.0 + 2.0 * ratios
        
        prototypes_small = self._generate_scale_prototypes(
            X_small, 0, num_per_scale[0], start_indices[0]
        )
        A_small = self._compute_participation(X_small, prototypes_small, scale_weights[0].item())
        
        prototypes_medium = self._generate_scale_prototypes(
            X_medium, 1, num_per_scale[1], start_indices[1]
        )
        A_medium = self._compute_participation(X_medium, prototypes_medium, scale_weights[1].item())
        
        prototypes_large = self._generate_scale_prototypes(
            X_large, 2, num_per_scale[2], start_indices[2]
        )
        A_large = self._compute_participation(X_large, prototypes_large, scale_weights[2].item())
        
        prototypes_dict = {'small': prototypes_small, 'medium': prototypes_medium, 'large': prototypes_large}
        A_dict = {'small': A_small, 'medium': A_medium, 'large': A_large}
        X_dict = {'small': X_small, 'medium': X_medium, 'large': X_large}
        
        A_dict, prototypes_dict = self.cross_scale_fusion(X_dict, A_dict, prototypes_dict)
        return A_dict, prototypes_dict


class CrossScaleHyperedgeFusion(nn.Module):
    def __init__(self, node_dim, num_heads=4):
        super().__init__()
        self.cross_attn_s2m = nn.MultiheadAttention(node_dim, num_heads, batch_first=True, dropout=0.1)
        self.cross_attn_s2l = nn.MultiheadAttention(node_dim, num_heads, batch_first=True, dropout=0.1)
        self.cross_attn_m2s = nn.MultiheadAttention(node_dim, num_heads, batch_first=True, dropout=0.1)
        
        self.gate_small = nn.Sequential(nn.Linear(node_dim * 3, node_dim), nn.Sigmoid())
        self.gate_medium = nn.Sequential(nn.Linear(node_dim * 2, node_dim), nn.Sigmoid())
        
        self.norm_small = nn.LayerNorm(node_dim)
        self.norm_medium = nn.LayerNorm(node_dim)
    
    def forward(self, X_dict, A_dict, prototypes_dict):
        P_small = prototypes_dict['small']
        P_medium = prototypes_dict['medium']
        P_large = prototypes_dict['large']
        
        P_small_from_medium, _ = self.cross_attn_s2m(P_small, P_medium, P_medium)
        P_small_from_large, _ = self.cross_attn_s2l(P_small, P_large, P_large)
        
        gate_s = self.gate_small(torch.cat([P_small, P_small_from_medium, P_small_from_large], dim=-1))
        P_small_enhanced = self.norm_small(
            P_small + gate_s * (P_small_from_medium + P_small_from_large) / 2
        )
        
        P_medium_from_small, _ = self.cross_attn_m2s(P_medium, P_small, P_small)
        gate_m = self.gate_medium(torch.cat([P_medium, P_medium_from_small], dim=-1))
        P_medium_enhanced = self.norm_medium(P_medium + gate_m * P_medium_from_small)
        
        prototypes_dict['small'] = P_small_enhanced
        prototypes_dict['medium'] = P_medium_enhanced
        return A_dict, prototypes_dict


class SmallObjectHyperedgePool(nn.Module):
    def __init__(self, node_dim, num_small_hyperedges=6, num_general_hyperedges=4):
        super().__init__()
        self.num_small = num_small_hyperedges
        self.num_general = num_general_hyperedges
        
        self.small_prototypes = nn.Parameter(torch.randn(num_small_hyperedges, node_dim) * 0.02)
        self.general_prototypes = nn.Parameter(torch.randn(num_general_hyperedges, node_dim) * 0.02)
        
        self.small_detector = nn.Sequential(
            nn.Linear(node_dim, node_dim // 4),
            nn.ReLU(),
            nn.Linear(node_dim // 4, 1),
            nn.Sigmoid()
        )
        self.fusion_alpha = nn.Parameter(torch.tensor(0.6))
    
    def forward(self, X, spatial_size):
        B, N, D = X.shape
        small_scores = self.small_detector(X)
        small_mask = (small_scores > 0.5).float()
        
        similarity_small = torch.bmm(
            X, self.small_prototypes.unsqueeze(0).expand(B, -1, -1).transpose(1, 2)
        )
        A_small = F.softmax(similarity_small / math.sqrt(D), dim=1)
        
        similarity_general = torch.bmm(
            X, self.general_prototypes.unsqueeze(0).expand(B, -1, -1).transpose(1, 2)
        )
        A_general = F.softmax(similarity_general / math.sqrt(D), dim=1)
        
        weight_small = small_mask * self.fusion_alpha
        weight_general = (1 - small_mask) * (1 - self.fusion_alpha)
        
        A_combined = torch.cat([
            A_small * (1 + weight_small),
            A_general * (1 + weight_general)
        ], dim=2)
        
        P_combined = torch.cat([self.small_prototypes, self.general_prototypes], dim=0)
        return A_combined, P_combined, small_mask


class AdaHGConvolution(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.v2e_proj = nn.Linear(embed_dim, embed_dim)
        self.e2v_proj = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X, A, prototypes):
        identity = X
        X_proj = self.v2e_proj(X)
        E = torch.bmm(A.transpose(1, 2), X_proj)
        E_proj = self.e2v_proj(E)
        Y = torch.bmm(A, E_proj)
        Y = self.dropout(Y)
        Y = self.norm1(identity + Y)
        
        identity2 = Y
        Y = self.ffn(Y)
        Y = self.norm2(identity2 + Y)
        return Y

class HyperFusion(nn.Module):
    def __init__(self,
                 c_in,
                 c_out,
                 max_pool_size=32,
                 capacity_target=0.6,
                 capacity_min=0.3,
                 capacity_max=1.0,
                 num_heads=8,
                 channel_adjust=True,
                 tunnel=False,
                 use_small_obj_pool=True,
                 use_freq_enhance=False):
        super().__init__()
        
        self.channel_adjust = channel_adjust
        self.tunnel = tunnel
        self.use_small_obj_pool = use_small_obj_pool
        self.use_freq_enhance = use_freq_enhance
        self.max_pool_size = max_pool_size
        
        if isinstance(c_in, int):
            c_in_list = [c_in, c_in, c_in]
        else:
            c_in_list = c_in
            assert len(c_in_list) == 3
        
        if channel_adjust:
            self.input_projs = nn.ModuleList([
                Conv(c_in_list[i], c_out, 1) for i in range(3)
            ])
            fuse_channels = c_out
        else:
            self.input_projs = None
            assert c_in_list[0] == c_in_list[1] == c_in_list[2]
            fuse_channels = c_in_list[0]
        
        self.fuse_channels = fuse_channels
        
        self.capacity_predictor = DynamicCapacityPredictor(
            node_dim=fuse_channels,
            num_scales=3,
            capacity_min=capacity_min,
            capacity_max=capacity_max,
            capacity_target=capacity_target
        )
        
        self.allocator = DensityAwareAllocator(
            node_dim=fuse_channels,
            max_pool_size=max_pool_size,
            num_scales=3,
            ema_momentum=0.9,
            min_hyperedges=2
        )
        
        self.ms_hyperedge_gen = DynamicMultiScaleHyperedgeGen(
            node_dim=fuse_channels,
            max_pool_size=max_pool_size,
            num_heads=num_heads,
            dropout=0.1,
            gumbel_tau=1.0
        )
        
        if use_small_obj_pool:
            self.small_obj_pool = SmallObjectHyperedgePool(
                fuse_channels,
                num_small_hyperedges=6,
                num_general_hyperedges=4
            )
        
        self.hgc_small = AdaHGConvolution(fuse_channels)
        self.hgc_medium = AdaHGConvolution(fuse_channels)
        self.hgc_large = AdaHGConvolution(fuse_channels)
        
        self.multi_scale_fusion = nn.Sequential(
            Conv(fuse_channels * 3, fuse_channels, 1),
            nn.BatchNorm2d(fuse_channels),
            nn.SiLU()
        )
        
        self._last_density_maps = None
        self._last_ratios = None
        self._last_capacity = None
        self._last_capacity_loss = None
        self._last_effective_size = None
    
    def get_auxiliary_loss(self, capacity_weight=0.05, sparsity_weight=0.01):
        loss = torch.tensor(0.0, device=self.capacity_predictor.capacity_bias.device)
        
        if self._last_capacity_loss is not None:
            loss = loss + capacity_weight * self._last_capacity_loss
        
        if self._last_density_maps is not None:
            sparsity = sum(d.mean() for d in self._last_density_maps) / len(self._last_density_maps)
            loss = loss + sparsity_weight * sparsity
        
        return loss
    
    def get_stats(self):
        return {
            'capacity_ratio': self._last_capacity.item() if self._last_capacity is not None else None,
            'effective_pool_size': self._last_effective_size,
            'ema_capacity': self.capacity_predictor.ema_capacity.item(),
            'scale_ratios': self._last_ratios.detach().cpu().numpy() if self._last_ratios is not None else None,
            'ema_scale_ratios': self.allocator.ema_ratios.cpu().numpy(),
        }
    
    def forward(self, X):
        B = X[0].shape[0]
        
        if self.channel_adjust and self.input_projs is not None:
            X = [proj(x) for proj, x in zip(self.input_projs, X)]
        
        _, _, H, W = X[1].shape
        X_small_spatial = F.interpolate(X[0], size=(H, W), mode='bilinear', align_corners=False)
        X_medium_spatial = X[1]
        X_large_spatial = F.interpolate(X[2], size=(H, W), mode='bilinear', align_corners=False)
        
        capacity_ratio, capacity_loss = self.capacity_predictor(
            [X_small_spatial, X_medium_spatial, X_large_spatial]
        )
        effective_pool_size = max(
            int((capacity_ratio.detach() * self.max_pool_size).round().item()),
            6  
        )
        
        num_per_scale, ratios, density_maps = self.allocator(
            [X_small_spatial, X_medium_spatial, X_large_spatial],
            effective_pool_size
        )
        
        self._last_density_maps = density_maps
        self._last_ratios = ratios
        self._last_capacity = capacity_ratio
        self._last_capacity_loss = capacity_loss
        self._last_effective_size = effective_pool_size
        
        X_small_flat = X_small_spatial.flatten(2).transpose(1, 2)
        X_medium_flat = X_medium_spatial.flatten(2).transpose(1, 2)
        X_large_flat = X_large_spatial.flatten(2).transpose(1, 2)
        
        A_dict, prototypes_dict = self.ms_hyperedge_gen(
            [X_small_flat, X_medium_flat, X_large_flat],
            num_per_scale,
            ratios
        )
        
        if self.use_small_obj_pool:
            A_small_obj, P_small_obj, _ = self.small_obj_pool(X_small_flat, (H, W))
            A_dict['small'] = torch.cat([A_dict['small'], A_small_obj], dim=2)
            prototypes_dict['small'] = torch.cat([
                prototypes_dict['small'],
                P_small_obj.unsqueeze(0).expand(B, -1, -1)
            ], dim=1)
        
        Y_small = self.hgc_small(X_small_flat, A_dict['small'], prototypes_dict['small'])
        Y_medium = self.hgc_medium(X_medium_flat, A_dict['medium'], prototypes_dict['medium'])
        Y_large = self.hgc_large(X_large_flat, A_dict['large'], prototypes_dict['large'])
        
        Y_small_spatial = Y_small.transpose(1, 2).view(B, -1, H, W)
        Y_medium_spatial = Y_medium.transpose(1, 2).view(B, -1, H, W)
        Y_large_spatial = Y_large.transpose(1, 2).view(B, -1, H, W)
        
        Y_concat = torch.cat([Y_small_spatial, Y_medium_spatial, Y_large_spatial], dim=1)
        x_fused = self.multi_scale_fusion(Y_concat)
        
        return x_fused


# ==================== 测试代码 ====================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    X_small = torch.randn(2, 384, 80, 80).to(device)
    X_medium = torch.randn(2, 192, 40, 40).to(device)
    X_large = torch.randn(2, 192, 20, 20).to(device)
    
    model = HyperFusion(
        c_in=[384, 192, 192],
        c_out=256,
        max_pool_size=32,       
        capacity_target=0.6,    
        capacity_min=0.3,       
        capacity_max=1.0,
        num_heads=8,
        channel_adjust=True,
        tunnel=True,
        use_small_obj_pool=True,
        use_freq_enhance=False
    ).to(device)
    
    # 训练模式
    model.train()
    print("=" * 60)
    print("训练模式测试（观察动态容量变化）")
    print("=" * 60)
    for i in range(5):
        output = model([X_small, X_medium, X_large])
        aux_loss = model.get_auxiliary_loss()
        stats = model.get_stats()
        print(f"[Step {i}] output={output.shape} | "
              f"capacity={stats['capacity_ratio']:.3f} | "
              f"effective_size={stats['effective_pool_size']} | "
              f"scales={stats['scale_ratios']} | "
              f"aux_loss={aux_loss.item():.4f}")
    
    # 推理模式
    model.eval()
    print("\n" + "=" * 60)
    print("推理模式测试")
    print("=" * 60)
    with torch.no_grad():
        output = model([X_small, X_medium, X_large])
        stats = model.get_stats()
        print(f"Output: {output.shape}")
        print(f"EMA capacity: {stats['ema_capacity']:.3f}")
        print(f"Effective pool: {stats['effective_pool_size']} / {model.max_pool_size}")
        print(f"EMA scale ratios: {stats['ema_scale_ratios']}")
    
    print("\n✅ MSHyperACE V3 测试通过！")
