"""
Cryo-EM 3D Volume Loading and 2D Projection Generation
任务1: 3D Volume 加载与预处理
任务2: 纯净 2D 投影生成 (宽松掩码，灰色背景)
"""

import numpy as np
import mrcfile
from scipy.ndimage import zoom, gaussian_filter
from typing import Tuple, Optional
from numba import njit, prange


def load_and_normalize_volume(filepath: str, normalize_method: str = "minmax",
                               mask_radius: float = 140) -> np.ndarray:
    """
    读取并标准化 3D Density Map (.mrc 格式)
    使用宽松的球形掩码
    """
    with mrcfile.open(filepath, permissive=True) as mrc:
        volume = mrc.data.astype(np.float32)
    
    print(f"原始体积数据形状: {volume.shape}")
    print(f"原始数据范围: [{volume.min():.4f}, {volume.max():.4f}]")
    
    # 应用宽松的球形掩码
    n = volume.shape[0]
    center = (n - 1) / 2.0
    z, y, x = np.ogrid[:n, :n, :n]
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
    
    # 硬球形掩码
    hard_mask = dist_from_center <= mask_radius
    volume = volume * hard_mask
    print(f"应用宽松球形掩码 (半径={mask_radius})")
    
    if normalize_method == "minmax":
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            volume = (volume - vmin) / (vmax - vmin)
        else:
            volume = np.zeros_like(volume)
        print(f"归一化后范围: [{volume.min():.4f}, {volume.max():.4f}]")
    elif normalize_method == "zscore":
        mean = volume.mean()
        std = volume.std()
        if std > 0:
            volume = (volume - mean) / std
        print(f"标准化后均值: {volume.mean():.4f}, 标准差: {volume.std():.4f}")
    
    return volume


def sample_euler_angles_hemisphere(n_projections: int, seed: int = 42) -> np.ndarray:
    """在半球空间内均匀采样欧拉角 (Rot, Tilt, Psi)"""
    rng = np.random.RandomState(seed)
    
    rot = rng.uniform(0, 360, n_projections)
    cos_tilt = rng.uniform(0, 1, n_projections)
    tilt = np.degrees(np.arccos(cos_tilt))
    psi = rng.uniform(0, 360, n_projections)
    
    return np.stack([rot, tilt, psi], axis=1)


@njit(parallel=True, cache=True)
def project_single_volume_numba(volume: np.ndarray, rot: float, tilt: float) -> np.ndarray:
    """
    使用 Numba 加速的单个体积投影
    """
    n = volume.shape[0]
    output_size = 256
    center = (n - 1) / 2.0
    
    # 角度转弧度
    alpha = np.deg2rad(rot)
    beta = np.deg2rad(tilt)
    
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    
    # 旋转矩阵 (Z then Y rotation, inverse)
    r00, r01, r02 = ca*cb, -sa, ca*sb
    r10, r11, r12 = sa*cb, ca, sa*sb
    r20, r21, r22 = -sb, 0, cb
    
    projection = np.zeros((output_size, output_size), dtype=np.float32)
    
    # 缩放因子
    scale = n / output_size
    
    for py in prange(output_size):
        for px in range(output_size):
            # 将像素坐标映射到体积坐标系
            x_proj = (px - output_size/2 + 0.5) * scale
            y_proj = (py - output_size/2 + 0.5) * scale
            
            # 沿 Z 轴积分
            total = 0.0
            for z_idx in range(n):
                z_proj = z_idx - center
                
                # 旋转回原始体积坐标
                x = r00 * x_proj + r01 * y_proj + r02 * z_proj + center
                y = r10 * x_proj + r11 * y_proj + r12 * z_proj + center
                z = r20 * x_proj + r21 * y_proj + r22 * z_proj + center
                
                # 三线性插值
                x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
                
                if 0 <= x0 < n-1 and 0 <= y0 < n-1 and 0 <= z0 < n-1:
                    fx, fy, fz = x - x0, y - y0, z - z0
                    
                    c000 = volume[z0, y0, x0]
                    c001 = volume[z0, y0, x0+1]
                    c010 = volume[z0, y0+1, x0]
                    c011 = volume[z0, y0+1, x0+1]
                    c100 = volume[z0+1, y0, x0]
                    c101 = volume[z0+1, y0, x0+1]
                    c110 = volume[z0+1, y0+1, x0]
                    c111 = volume[z0+1, y0+1, x0+1]
                    
                    c00 = c000 * (1-fx) + c001 * fx
                    c01 = c010 * (1-fx) + c011 * fx
                    c10 = c100 * (1-fx) + c101 * fx
                    c11 = c110 * (1-fx) + c111 * fx
                    
                    c0 = c00 * (1-fy) + c01 * fy
                    c1 = c10 * (1-fy) + c11 * fy
                    
                    val = c0 * (1-fz) + c1 * fz
                    total += val
            
            projection[py, px] = total
    
    return projection


def create_circular_mask(output_size: int, radius: float, soften_width: float) -> np.ndarray:
    """创建圆形软掩码"""
    h, w = output_size, output_size
    center_y, center_x = h / 2.0, w / 2.0
    
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # 软边界掩码
    soft_mask = np.ones((h, w), dtype=np.float32)
    
    # 过渡区域
    edge_start = radius - soften_width
    edge_end = radius
    
    # 在过渡区域内线性插值
    transition_region = (dist_from_center > edge_start) & (dist_from_center <= edge_end)
    ratio = (dist_from_center[transition_region] - edge_start) / soften_width
    soft_mask[transition_region] = 1 - ratio
    soft_mask[dist_from_center > edge_end] = 0
    
    return soft_mask


def project_volume_numba(volume: np.ndarray, euler_angles: np.ndarray,
                          output_size: int = 256) -> np.ndarray:
    """使用 Numba 加速的批量投影生成"""
    n_projections = len(euler_angles)
    projections = np.zeros((n_projections, output_size, output_size), dtype=np.float32)
    
    # 如果体积太大，先降采样
    vol_shape = volume.shape[0]
    if vol_shape > output_size * 2:
        print(f"预处理: 降采样体积 {vol_shape} -> {output_size * 2}")
        zoom_factor = (output_size * 2) / vol_shape
        volume = zoom(volume, zoom_factor, order=1)
        print(f"处理后体积形状: {volume.shape}")
    
    for i in range(n_projections):
        rot, tilt, psi = euler_angles[i]
        proj = project_single_volume_numba(volume, rot, tilt)
        projections[i] = proj
        
        if (i + 1) % 1000 == 0 or (i + 1) == 100 or (i + 1) == 10 or i == 0:
            print(f"已生成 {i + 1}/{n_projections} 张投影")
    
    return projections


def generate_clean_projections(volume_path: str, n_projections: int = 10000,
                                output_size: int = 256, seed: int = 42,
                                save_path: Optional[str] = None,
                                normalize_method: str = "minmax",
                                bg_gray: float = 0.52) -> Tuple[np.ndarray, np.ndarray]:
    """主函数: 生成纯净 2D 投影"""
    print("=" * 60)
    print("任务1: 加载并预处理 3D Volume")
    print("=" * 60)
    volume = load_and_normalize_volume(volume_path, normalize_method, mask_radius=140)
    
    print("\n" + "=" * 60)
    print("任务2: 生成纯净 2D 投影")
    print("=" * 60)
    print(f"生成 {n_projections} 张投影图，尺寸 {output_size}x{output_size}")
    
    euler_angles = sample_euler_angles_hemisphere(n_projections, seed)
    print(f"欧拉角范围: Rot=[{euler_angles[:,0].min():.1f}, {euler_angles[:,0].max():.1f}], "
          f"Tilt=[{euler_angles[:,1].min():.1f}, {euler_angles[:,1].max():.1f}], "
          f"Psi=[{euler_angles[:,2].min():.1f}, {euler_angles[:,2].max():.1f}]")
    
    projections = project_volume_numba(volume, euler_angles, output_size)
    
    # 先归一化投影（在应用掩码前）
    print("\n归一化投影图像...")
    for i in range(n_projections):
        pmin, pmax = projections[i].min(), projections[i].max()
        if pmax > pmin:
            projections[i] = (projections[i] - pmin) / (pmax - pmin)
    
    # 创建圆形掩码
    print(f"应用圆形掩码，背景灰度={bg_gray}...")
    proj_radius = 110  # 稍微收紧以消除方形伪影
    soften = 15
    circular_mask = create_circular_mask(output_size, proj_radius, soften)
    
    # 应用掩码：信号区保持归一化值，背景设为灰色
    for i in range(n_projections):
        projections[i] = projections[i] * circular_mask + bg_gray * (1 - circular_mask)
    
    print(f"投影数据形状: {projections.shape}")
    print(f"投影数据范围: [{projections.min():.4f}, {projections.max():.4f}]")
    
    if save_path:
        print(f"\n保存结果到: {save_path}")
        np.savez_compressed(save_path, 
                            projections=projections, 
                            euler_angles=euler_angles,
                            volume_shape=volume.shape)
        print("保存完成!")
    
    return projections, euler_angles


if __name__ == "__main__":
    volume_path = "/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/emd_54951.map"
    save_path = "/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/clean_projections.npz"
    
    projections, euler_angles = generate_clean_projections(
        volume_path=volume_path,
        n_projections=10000,
        output_size=256,
        seed=42,
        save_path=save_path,
        normalize_method="minmax",
        bg_gray=0.52
    )
    
    print("\n" + "=" * 60)
    print("完成!")
    print(f"投影形状: {projections.shape}")
    print(f"欧拉角形状: {euler_angles.shape}")
    print("=" * 60)
