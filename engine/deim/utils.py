"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2023 . All Rights Reserved.     
"""
   
import math     
from typing import List, Tuple, Union
    
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter 
from scipy.spatial import KDTree
 
import torch  
import torch.nn as nn  
import torch.nn.functional as F   
     

def inverse_sigmoid(x: torch.Tensor, eps: float=1e-5) -> torch.Tensor:  
    x = x.clip(min=0., max=1.) 
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))     
   
 
def bias_init_with_prob(prior_prob=0.01):  
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))     
    return bias_init
 

def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Args: 
        value (Tensor): [bs, value_length, n_head, c]    
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]  
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]     
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]     
 
    Returns:
        output (Tensor): [bs, Length_{query}, C]     
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape  

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)    
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []    
    for level, (h, w) in enumerate(value_spatial_shapes): 
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_    
        value_l_ = value_list[level].flatten(2).permute(    
            0, 2, 1).reshape(bs * n_head, c, h, w)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute( 
            0, 2, 1, 3, 4).flatten(0, 1)   
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(   
            value_l_,
            sampling_grid_l_,    
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False)    
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points)    
    output = (torch.stack(    
        sampling_value_list, dim=-2).flatten(-2) *    
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)
   
   
    
def deformable_attention_core_func_v2(\
    value: torch.Tensor,
    value_spatial_shapes, 
    sampling_locations: torch.Tensor,  
    attention_weights: torch.Tensor,
    num_points_list: List[int],    
    method='default',   
    value_shape='default',  
    ):   
    """
    Args:     
        value (Tensor): [bs, value_length, n_head, c]    
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels * n_points, 2]    
        attention_weights (Tensor): [bs, query_length, n_head, n_levels * n_points]  
    
    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """     
    # TODO find the version
    if value_shape == 'default':     
        bs, n_head, c, _ = value[0].shape  
    elif value_shape == 'reshape':   # reshape following RT-DETR
        bs, _, n_head, c = value.shape  
        split_shape = [h * w for h, w in value_spatial_shapes]
        value = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)
    _, Len_q, _, _, _ = sampling_locations.shape 

    # sampling_offsets [8, 480, 8, 12, 2] 
    if method == 'default':
        sampling_grids = 2 * sampling_locations - 1 
    
    elif method == 'discrete':    
        sampling_grids = sampling_locations     
 
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1) 
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)  

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)  
        sampling_grid_l: torch.Tensor = sampling_locations_list[level] 
     
        if method == 'default':     
            sampling_value_l = F.grid_sample(  
                value_l,
                sampling_grid_l,    
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)

        elif method == 'discrete':    
            # n * m, seq, n, 2
            sampling_coord = (sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5).to(torch.int64)
 
            # FIX ME? for rectangle input     
            sampling_coord = sampling_coord.clamp(0, h - 1)     
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2)   
     
            s_idx = torch.arange(sampling_coord.shape[0], device=value_l.device).unsqueeze(-1).repeat(1, sampling_coord.shape[1]) 
            sampling_value_l: torch.Tensor = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]] # n l c
  
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(bs * n_head, c, Len_q, num_points_list[level])    
    
        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))   
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)
     
    return output.permute(0, 2, 1)     


def get_activation(act: str, inpace: bool=True):
    """get activation     
    """
    if act is None:  
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act
     
    act = act.lower() 
    
    if act == 'silu' or act == 'swish':    
        m = nn.SiLU() 

    elif act == 'relu':  
        m = nn.ReLU()  

    elif act == 'leaky_relu':   
        m = nn.LeakyReLU() 

    elif act == 'silu':
        m = nn.SiLU() 

    elif act == 'gelu':   
        m = nn.GELU()

    elif act == 'hardsigmoid':    
        m = nn.Hardsigmoid()
    
    else:
        raise RuntimeError('')

    if hasattr(m, 'inplace'):     
        m.inplace = inpace
 
    return m

class DensityMapGenerator: 
    """密度图生成器 - 支持多种生成策略"""
    
    def __init__(self, image_size: Tuple[int, int]): 
        """
        Args:
            image_size: (height, width) 图像尺寸
        """
        self.height, self.width = image_size    
        
    def generate_from_boxes(self, 
                           boxes: Union[List, np.ndarray],
                           method: str = 'gaussian',     
                           sigma: float = None,
                           normalize: bool = True) -> np.ndarray:
        """     
        从检测框生成密度图 
   
        Args:     
            boxes: 检测框列表，格式是： 
                   - [[x_center, y_center, w, h], ...] (xywh格式)   
            method: 生成方法   
                   - 'gaussian': 高斯核密度图（推荐）
                   - 'adaptive_gaussian': 自适应高斯核
                   - 'point': 点标注密度图   
                   - 'box_mask': 框区域密度图   
            sigma: 高斯核标准差，None时自动计算
            normalize: 是否归一化使积分等于目标数量    
 
        Returns:
            density_map: (H, W) 密度图     
        """     
        if len(boxes) == 0: 
            return np.zeros((self.height, self.width), dtype=np.float32)
     
        boxes = np.array(boxes) 
        
        centers = self._boxes_to_centers(boxes)
        
        if method == 'gaussian':
            return self._generate_gaussian(centers, sigma, normalize)
        elif method == 'adaptive_gaussian':     
            return self._generate_adaptive_gaussian(centers, boxes, normalize)  
        elif method == 'point': 
            return self._generate_point(centers, normalize)     
        elif method == 'box_mask':    
            return self._generate_box_mask(boxes, normalize) 
        else:    
            raise ValueError(f"Unknown method: {method}")  
    
    def _boxes_to_centers(self, boxes: np.ndarray) -> np.ndarray:     
        """将检测框转换为中心点坐标"""    
        if boxes.shape[1] == 4:   
            centers = boxes[:, :2].copy()  
        else:   
            raise ValueError("Boxes should have 4 columns")
        
        return centers
    
    def _generate_gaussian(self,  
                          centers: np.ndarray, 
                          sigma: float = None,   
                          normalize: bool = True) -> np.ndarray:    
        """     
        生成固定高斯核密度图（最常用方法）
    
        Args:     
            centers: (N, 2) 中心点坐标 [x, y]
            sigma: 高斯核标准差，None时自动设置为15    
            normalize: 是否归一化
        """     
        if sigma is None:     
            sigma = 15  # 默认值，适用于大多数场景    
        
        density_map = np.zeros((self.height, self.width), dtype=np.float32)    
   
        # 为每个中心点生成高斯分布
        for center in centers:   
            x, y = int(center[0]), int(center[1])     
 
            # 边界检查 
            if x < 0 or x >= self.width or y < 0 or y >= self.height:   
                continue
            
            # 在该点位置添加一个点   
            density_map[y, x] += 1    
    
        # 应用高斯滤波
        density_map = gaussian_filter(density_map, sigma=sigma, mode='constant')  
  
        # 归一化：使积分等于目标数量  
        if normalize and density_map.sum() > 0:   
            density_map = density_map * len(centers) / density_map.sum()    
   
        return density_map  
    
    def _generate_adaptive_gaussian(self, 
                                   centers: np.ndarray,
                                   boxes: np.ndarray,  
                                   normalize: bool = True) -> np.ndarray:    
        """   
        生成自适应高斯核密度图（根据目标大小和密度调整sigma）
        这是MCNN、CSRNet等论文中使用的方法
        
        Args:
            centers: (N, 2) 中心点坐标   
            boxes: (N, 4) 检测框
            normalize: 是否归一化
        """
        density_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        if len(centers) == 0:    
            return density_map
   
        # 使用KD树找到每个点的k近邻 
        k = min(3, len(centers))  # 使用3个最近邻 
        tree = KDTree(centers)
        
        for idx, center in enumerate(centers):     
            x, y = int(center[0]), int(center[1])
  
            # 边界检查
            if x < 0 or x >= self.width or y < 0 or y >= self.height: 
                continue  
            
            # 计算自适应sigma
            if len(centers) > 1: 
                distances, _ = tree.query(center, k=k+1)  # +1因为包含自己
                avg_distance = np.mean(distances[1:])  # 排除自己   
                sigma = avg_distance * 0.3  # 经验系数   
            else:
                sigma = 15  # 只有一个目标时使用默认值
            
            # 也可以考虑目标框的大小
            box = boxes[idx]  
            if boxes.shape[1] == 4:     
                # 计算框的面积作为参考    
                # xywh格式  
                box_size = np.sqrt(box[2] * box[3])
    
                # 结合距离和框大小   
                sigma = max(sigma, box_size * 0.5)
            
            # 限制sigma范围    
            sigma = np.clip(sigma, 3, 50) 
            
            # 生成高斯核   
            size = int(6 * sigma)  # 3-sigma原则    
            if size % 2 == 0:    
                size += 1    
  
            # 创建高斯核
            gaussian_kernel = self._create_gaussian_kernel(size, sigma)
            
            # 计算粘贴位置     
            half_size = size // 2
            y_start = max(0, y - half_size) 
            y_end = min(self.height, y + half_size + 1)
            x_start = max(0, x - half_size) 
            x_end = min(self.width, x + half_size + 1)  
   
            # 计算核的对应区域
            ky_start = half_size - (y - y_start) 
            ky_end = half_size + (y_end - y)
            kx_start = half_size - (x - x_start)    
            kx_end = half_size + (x_end - x)
    
            # 添加到密度图     
            density_map[y_start:y_end, x_start:x_end] += \
                gaussian_kernel[ky_start:ky_end, kx_start:kx_end]
        
        # 归一化    
        if normalize and density_map.sum() > 0:     
            density_map = density_map * len(centers) / density_map.sum()     
        
        return density_map    
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """创建2D高斯核"""
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))    
        return kernel / kernel.sum()
  
    def _generate_point(self, centers: np.ndarray, normalize: bool = True) -> np.ndarray:  
        """生成点标注密度图（简单方法，不推荐用于训练）""" 
        density_map = np.zeros((self.height, self.width), dtype=np.float32)
 
        for center in centers:    
            x, y = int(center[0]), int(center[1]) 
            if 0 <= x < self.width and 0 <= y < self.height:
                density_map[y, x] = 1.0 
        
        return density_map  
    
    def _generate_box_mask(self, boxes: np.ndarray, normalize: bool = True) -> np.ndarray: 
        """生成框区域密度图（将密度均匀分布在框内）"""
        density_map = np.zeros((self.height, self.width), dtype=np.float32)     
     
        for box in boxes:    
            x1 = int(box[0])    
            y1 = int(box[1])    
            x2 = int(box[0] + box[2])
            y2 = int(box[1] + box[3])
            
            # 边界裁剪
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.width, x2)     
            y2 = min(self.height, y2) 
 
            # 在框内均匀分布密度 
            area = (x2 - x1) * (y2 - y1) 
            if area > 0:
                density_map[y1:y2, x1:x2] += 1.0 / area
        
        if normalize and density_map.sum() > 0:
            density_map = density_map * len(boxes) / density_map.sum()
        
        return density_map
 
     
def visualize_density_map(image: np.ndarray, 
                         density_map: np.ndarray,
                         boxes: np.ndarray = None,
                         save_path: str = None):
    """
    可视化密度图
    
    Args:
        image: 原始图像 (H, W, 3) 
        density_map: 密度图 (H, W)
        boxes: 检测框（可选）
        save_path: 保存路径（可选）  
    """  
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
     
    # 原始图像
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))    
    axes[0].set_title('Original Image')
    axes[0].axis('off')  
  
    # 带框的图像
    img_with_boxes = image.copy() 
    if boxes is not None:
        for box in boxes:  
            if len(box) == 4:    
                # xywh格式     
                x1 = int(box[0] - box[2] / 2)
                y1 = int(box[1] - box[3] / 2) 
                x2 = int(box[0] + box[2] / 2)
                y2 = int(box[1] + box[3] / 2)     
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2) 
  
    axes[1].imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Bounding Boxes (Count: {len(boxes) if boxes is not None else 0})')   
    axes[1].axis('off')
    
    # 密度图
    im = axes[2].imshow(density_map, cmap='jet') 
    axes[2].set_title(f'Density Map (Sum: {density_map.sum():.2f})')     
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])   
    
    plt.tight_layout()    
    
    if save_path:   
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
 
def visualize_density_map_only(density_map: np.ndarray, save_path: str = None):
    """   
    可视化密度图（仅显示密度图）
    
    Args:
        density_map: 密度图 (H, W)    
        save_path: 保存路径（可选）
    """  
    plt.figure(figsize=(8, 6))  
    im = plt.imshow(density_map, cmap='jet')    
    plt.title(f'Density Map (Sum: {density_map.sum():.2f})')    
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)   

    plt.tight_layout() 
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')     
        print(f"Saved to {save_path}")    
