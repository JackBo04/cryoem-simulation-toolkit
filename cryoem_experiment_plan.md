# Cryo-EM 粒子生成仿真实验计划

## 实验目标

生成用于 cryo-EM 粒子 picking、CTF 估计和粒子分类算法研究的仿真数据集。

---

## Task 1: 3D Volume 加载与预处理

### 需求
读取标准 `.mrc` 格式的 3D Density Map 并标准化。

### 输入
- 文件路径：`data/apoferritin.mrc` 或 `data/ribosome.mrc`
- 推荐数据源：EMPIAR-10028 (Apoferritin) 或 EMPIAR-10017 (Ribosome)

### 处理流程
```python
def load_and_normalize_mrc(filepath: str) -> np.ndarray:
    volume = mrcfile.open(filepath).data
    # 标准化方案 A: 零均值单位方差
    volume = (volume - volume.mean()) / volume.std()
    # 方案 B: [0, 1] 范围
    # volume = (volume - volume.min()) / (volume.max() - volume.min())
    return volume.astype(np.float32)
```

### 参数选择
| 参数 | 值 | 理由 |
|------|-----|------|
| 数据类型 | float32 | 兼容傅里叶变换精度需求 |
| 标准化方式 | 零均值单位方差 | 便于后续 CTF 频域处理 |

### 验证
- [ ] Volume shape 应为 3D tuple (如 256³ 或 384³)
- [ ] 无 NaN/Inf 值
- [ ] 均值 ≈ 0, 标准差 ≈ 1 (如使用零均值方案)

### 输出
`volume: np.ndarray` — shape `(D, H, W)`

---

## Task 2: 纯净 2D 投影生成

### 需求
根据随机欧拉角生成 3D volume 的 2D 正交投影。

### 输入
- `volume`: Task 1 输出的 3D array
- `n_projections`: 10,000
- `image_size`: 256 × 256

### 欧拉角采样策略
```python
def sample_euler_angles(n: int, strategy: str = "hemisphere") -> np.ndarray:
    """
    半球空间均匀采样
    Rot:   [0°, 360°)    绕 Z 轴 (in-plane rotation)
    Tilt:  [0°, 90°]     绕 Y 轴 (primary tilt) — 半球限制
    Psi:   [0°, 360°)    绕 Z'' (in-plane rotation after tilt)
    """
    angles = np.zeros((n, 3))
    angles[:, 0] = np.random.uniform(0, 360, n)       # Rot
    angles[:, 1] = np.random.uniform(0, 90, n)        # Tilt (hemisphere)
    angles[:, 2] = np.random.uniform(0, 360, n)       # Psi
    return angles
```

### 投影实现
```python
import torch
from scipy.ndimage import rotate

def project_volume(volume: np.ndarray, angles: np.ndarray, 
                  output_size: int = 256) -> np.ndarray:
    """使用 PyTorch 实现 GPU 加速投影"""
    volume_tensor = torch.tensor(volume, dtype=torch.float32)
    volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    
    # 创建 3D 旋转矩阵 (Euler ZYZ 约定)
    rot, tilt, psi = angles
    # ... 构建 4x4 变换矩阵 ...
    
    grid = F.affine_grid(...).to(device)
    rotated = F.grid_sample(volume_tensor, grid, mode='bilinear', padding_mode='zeros')
    
    # 沿 Z 轴积分投影
    projection = rotated.sum(dim=2).squeeze()  # (H, W)
    return projection.numpy()
```

### 参数选择
| 参数 | 值 | 理由 |
|------|-----|------|
| Rot | U[0°, 360°) | 完整 in-plane 旋转 |
| Tilt | U[0°, 90°] | 半球采样，避免重复投影 |
| Psi | U[0°, 360°) | 完整 in-plane 旋转 |
| 投影数量 | 10,000 | 足够训练深度学习模型 |
| 图像尺寸 | 256×256 | 平衡计算效率与细节保留 |

### 验证
- [ ] 输出 shape: `(10000, 256, 256)`
- [ ] 投影强度分布合理（非全黑/全白）
- [ ] 不同角度的投影有明显差异

### 输出
- `clean_projections: np.ndarray` — shape `(10000, 256, 256)`
- `gt_angles: np.ndarray` — shape `(10000, 3)`, 单位：度

---

## Task 3: 完美软掩膜生成

### 需求
基于纯净投影生成精确的蛋白质软掩膜。

### 输入
- `clean_projections`: Task 2 输出

### 处理流程
```python
def generate_soft_mask(projections: np.ndarray, 
                       threshold: float = None,
                       gaussian_sigma: float = 4.0) -> np.ndarray:
    """
    步骤:
    1. Otsu 自动阈值或固定阈值二值化
    2. 形态学膨胀 (kernel=3×3)
    3. 高斯模糊生成软边缘
    """
    masks = []
    for proj in projections:
        # 1. 阈值二值化
        if threshold is None:
            # Otsu's method
            threshold = threshold_otsu(proj)
        binary = (proj > threshold).astype(np.float32)
        
        # 2. 形态学膨胀
        binary = binary_dilation(binary, iterations=1)
        
        # 3. 高斯模糊 → 软掩膜
        soft_mask = gaussian_filter(binary, sigma=gaussian_sigma)
        soft_mask = np.clip(soft_mask, 0, 1)  # 确保 [0,1] 范围
        masks.append(soft_mask)
    
    return np.stack(masks)
```

### 参数选择
| 参数 | 值 | 理由 |
|------|-----|------|
| 阈值方法 | Otsu 自动 | 适应不同密度范围的投影 |
| 膨胀迭代 | 1 次 | 轻微扩展确保边缘包裹 |
| Gaussian σ | 4.0 像素 | 边缘平滑，10% 边缘宽度 |
| 输出范围 | [0, 1] | 兼容掩膜乘法操作 |

### 验证
- [ ] 输出 shape: `(10000, 256, 256)`
- [ ] 数值范围: [0, 1]
- [ ] 掩膜边缘平滑过渡（非硬边界）
- [ ] 蛋白质区域被完整覆盖

### 输出
`soft_masks: np.ndarray` — shape `(10000, 256, 256)`

---

## Task 4: 物理退化模拟 (CTF + Noise)

### 需求
模拟真实 cryo-EM 成像的 CTF 调制和噪声退化。

### CTF 参数
| 参数 | 符号 | 值 | 依据 |
|------|------|-----|------|
| 加速电压 | V | 300 kV | 主流 cryo-EM |
| 球差 | Cs | 2.7 mm | Talos / Titan Krios |
| 色差 | Cc | 2.0 mm | 典型值 |
| 电子波长 | λ | 0.0197 Å | V=300kV 计算值 |
| 像素尺寸 | d | 1.0 Å/pixel | 可调 |
| Defocus | Δz | U[1.0, 3.0] μm | 欠焦量 |

### CTF 公式
```python
def compute_ctf(freq: np.ndarray, defocus: float,
                Cs: float = 2.7e7,  # Å
                V: float = 300e3,   # V
                Cc: float = 2.0e7,  # Å
                d: float = 1.0):    # Å/pixel
    """计算 Contrast Transfer Function"""
    # 电子波长
    h = 6.626e-34
    e = 1.602e-19
    m0 = 9.109e-31
    lambda_ = h / np.sqrt(2 * m0 * e * V * (1 + e * V / (2 * m0 * c**2))) * 1e10
    
    # 空间频率
    k = freq / d  # cycles/Å
    
    # 色差和能量 spread
    delta_E = 0.8  # eV
    delta_E_over_E = delta_E / V
    
    # Ewald 球曲率修正
    w = 1 + Cs * lambda_**2 * k**2 - Cc * lambda_ * delta_E_over_E
    
    # 振幅和相位调制
    chi = np.pi * lambda_ * k**2 * (Cs * lambda_**2 * k**2 - 2 * defocus * 1e4) / w
    
    # CTF = -sin(χ) (相位对比) + 可选幅度衰减
    ctf = -np.sin(chi) * np.exp(-0.5 * (Cc * lambda_ * delta_E_over_E * k**2)**2)
    
    return ctf
```

### 完整退化流程
```python
def apply_ctf_and_noise(projection: np.ndarray, defocus: float,
                        snr: float = 0.05) -> np.ndarray:
    """模拟 CTF 调制 + 高斯白噪声"""
    # 1. FFT
    F = fft2(projection)
    
    # 2. 构建频率网格
    ky, kx = np.fft.fftfreq(256, d=1.0)
    KX, KY = np.meshgrid(kx, ky)
    freq = np.sqrt(KX**2 + KY**2)
    
    # 3. CTF 调制
    ctf = compute_ctf(freq, defocus)
    F_ctf = F * ctf
    
    # 4. 逆 FFT
    image_ctf = np.real(ifft2(F_ctf))
    
    # 5. 添加噪声 (SNR = 0.05)
    signal_power = np.var(image_ctf)
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), image_ctf.shape)
    
    return image_ctf + noise
```

### 参数选择
| 参数 | 值 | 理由 |
|------|-----|------|
| Defocus U/V | U[1.0, 3.0] μm | 典型欠焦范围 |
| Defocus 椭圆率 | 0.9 (V = 0.9U) | 轻微像散 |
| SNR | 0.05 | 极低信噪比，接近真实 cryo-EM |
| 噪声类型 | Gaussian 白噪声 | 简化模型 |

### 验证
- [ ] 输出 shape: `(10000, 256, 256)`
- [ ] 图像肉眼可见退化
- [ ] CTF 条纹在功率谱中可见

### 输出
- `noisy_projections: np.ndarray` — shape `(10000, 256, 256)`
- `ctf_params: dict` — 包含每张图的 defocus 等参数

---

## Task 5: 数据打包与输出

### 需求
生成 RELION/CryoSPARC 兼容的标准格式文件。

### 输出文件 1: `noisy_particles.mrcs`
```python
import mrcfile

with mrcfile.new('noisy_particles.mrcs', overwrite=True) as mrc:
    mrc.set_data(noisy_projections.astype(np.float32))
```

### 输出文件 2: `masked_particles.mrcs`
```python
# 逐像素相乘
masked = noisy_projections * soft_masks

with mrcfile.new('masked_particles.mrcs', overwrite=True) as mrc:
    mrc.set_data(masked.astype(np.float32))
```

### 输出文件 3: `ground_truth.star`
```python
def write_relion_star(particles_path: str,
                      angles: np.ndarray,
                      ctf_params: np.ndarray,
                      output_path: str):
    """生成符合 RELION 3.x 规范的 .star 文件"""
    
    header = """data_
loop_
_rlnMicrographName #1
_rlnCoordinateX #2
_rlnCoordinateY #3
_rlnAngleRot #4
_rlnAngleTilt #5
_rlnAnglePsi #6
_rlnDefocusU #7
_rlnDefocusV #8
_rlnDefocusAngle #9
_rlnVoltage #10
_rlnSphericalAberration #11
_rlnAmplitudeContrast #12
"""
    
    rows = []
    for i in range(len(angles)):
        rot, tilt, psi = angles[i]
        defocus_u, defocus_v, defocus_angle = ctf_params[i]
        
        row = f"{particles_path}/{i:06d}.mrc "
        row += f"{128.0} {128.0} "  # 中心坐标
        row += f"{rot:.4f} {tilt:.4f} {psi:.4f} "
        row += f"{defocus_u:.2f} {defocus_v:.2f} {defocus_angle:.2f} "
        row += "300.000 2.700 0.100\n"  # 固定 CTF 参数
        rows.append(row)
    
    with open(output_path, 'w') as f:
        f.write(header + ''.join(rows))
```

### STAR 文件格式要求
| 字段 | 含义 |
|------|------|
| `rlnMicrographName` | 图像相对路径 |
| `rlnCoordinateX/Y` | 粒子坐标 |
| `rlnAngleRot` | Euler α (绕 Z) |
| `rlnAngleTilt` | Euler β (绕 Y) |
| `rlnAnglePsi` | Euler γ (绕 Z'') |
| `rlnDefocusU/V` | 欠焦量 (Å) |
| `rlnDefocusAngle` | 像散角 |

### 验证
- [ ] `.mrcs` 文件可被 RELION/CryoSPARC 读取
- [ ] `.star` 文件格式校验通过
- [ ] 图像尺寸和数量正确

### 输出
- `noisy_particles.mrcs` — 含噪图像栈
- `masked_particles.mrcs` — 掩膜处理后图像栈
- `ground_truth.star` — RELION 格式元数据

---

## 实验参数汇总表

| 参数 | 符号 | 值 |
|------|------|-----|
| 投影数量 | N | 10,000 |
| 图像尺寸 | - | 256 × 256 |
| 像素尺寸 | d | 1.0 Å/pixel |
| 加速电压 | V | 300 kV |
| 球差 | Cs | 2.7 mm |
| Defocus 范围 | Δz | 1.0–3.0 μm |
| SNR | - | 0.05 |
| 软掩膜 σ | - | 4.0 |

---

## 建议实现顺序

1. **Step 1**: 环境配置 + 依赖安装
2. **Step 2**: Task 1 (Volume 加载) — 快速验证数据流
3. **Step 3**: Task 2 (投影生成) — 核心算法，优先优化
4. **Step 4**: Task 3 (掩膜生成) — 独立模块
5. **Step 5**: Task 4 (CTF+噪声) — 物理准确性关键
6. **Step 6**: Task 5 (数据导出) — 格式验证
7. **Step 7**: 端到端测试 + 可视化验证

---

## 可选扩展

- [ ] 批量生成多种蛋白质 (Apoferritin + Ribosome + Virus)
- [ ] 添加 Ewald 球效应 (更真实的投影几何)
- [ ] 模拟 DQE/MTF 探测器响应
- [ ] 生成 CryoSPARC 兼容的 `.cs` 格式
- [ ] 增加粒子污染/碎片作为 hard negative
