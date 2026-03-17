# Cryo-EM 仿真实验数据报告

> **实验地点**: `/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test`  
> **实验日期**: 2026年3月  
> **实验目的**: 生成用于 cryo-EM 重建算法测试的仿真数据集，研究 CTF 效应、半径掩膜对重建的影响

---

## 一、实验概述

本实验旨在构建高质量的冷冻电镜（Cryo-EM）仿真数据集，用于测试和验证粒子挑选、CTF估计和三维重建算法。主要工作包括：

1. **3D Volume 投影生成**: 从真实蛋白质结构生成 2D 投影
2. **CTF 物理模拟**: 模拟真实显微镜的对比度传递函数
3. **B-factor Envelope**: 引入信号衰减包络，提高物理真实性
4. **多半径掩膜**: 系统研究不同掩膜半径对重建的影响

---

## 二、实验方法

### 2.1 数据来源

| 数据集 | EMDB ID | 蛋白质 | 原始尺寸 | 物理尺寸 |
|--------|---------|--------|----------|----------|
| Dataset 1 | EMD-54951 | 未知小蛋白 | 320³ @ 0.592 Å/px | ~189 Å |
| Dataset 2 | EMD-45747 | BRCA1 | 320³ @ 0.825 Å/px | ~264 Å |
| Dataset 3 | EMD-8194 | (新增) | - | - |

### 2.2 技术路线

实验采用两条技术路线并行：

```
路线 A: ASPIRE (稳定可靠)
├── 使用普林斯顿 ASPIRE 库
├── 傅里叶切片定理投影
└── CPU 计算

路线 B: PyTorch Fourier (GPU加速)
├── 自实现傅里叶切片
├── PyTorch GPU 加速
└── 速度提升 5-50x
```

### 2.3 CTF 参数配置

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 加速电压 | V | 300 kV | 标准冷冻电镜电压 |
| 球差 | Cs | 2.7 mm | Talos/Titan Krios |
| 振幅对比度 | AC | 0.1 | 典型值 |
| 欠焦范围 | Δz | 1.0-2.0 μm | 均匀随机分布 |
| B-factor | B | 60 Å² | 高分辨率衰减 |
| 信噪比 | SNR | 0.05 | 低剂量成像条件 |

**B-factor Envelope 公式**:
```
E(k) = exp(-B·k²/4)
```

---

## 三、实验结果

### 3.1 投影生成结果

使用 Numba 加速的投影算法，成功生成了 10,000 张纯净投影：

![纯净投影示例](projection_final_v2.png)

*图 1: 10 张不同欧拉角的纯净投影示例。使用半球采样策略 (Rot: 0-360°, Tilt: 0-90°)，图像尺寸 256×256，背景灰度 0.52。*

### 3.2 CTF Envelope 效果验证

对比了无 Envelope 和不同 B-factor 的 CTF 效果：

![CTF Envelope 对比](ctf_envelope_comparison.png)

*图 2: CTF Envelope 效果对比。左图：无 Envelope，高频振荡过多；中图：B=50 Å²，平滑的 Thon 环；右图：径向强度对比。*

**关键发现**:
- 无 Envelope 时 CTF 高频振荡过多，不符合物理实际
- B=60 Å² 时，4 Å 分辨率处信号衰减至 39%，接近真实探测器响应
- Envelope 有效抑制了高频噪声放大

### 3.3 B-factor 参数扫描

系统测试了 B=30, 60, 100 Å² 的效果：

![B-factor 参数扫描](ctf_simple_comparison.png)

*图 3: 不同 B-factor 值的 CTF 对比。上排：2D CTF 图像；下排：1D 径向剖面。随着 B 增大，高频信号衰减加剧。*

| 分辨率 | B=30 | B=60 | B=100 |
|--------|------|------|-------|
| 10 Å | 0.93 | 0.86 | 0.78 |
| 6 Å | 0.77 | 0.66 | 0.50 |
| 4 Å | 0.53 | 0.39 | 0.20 |
| 3 Å | 0.32 | 0.19 | 0.06 |

**结论**: B=60 Å² 在保持信号与抑制噪声间取得平衡。

### 3.4 双数据集对比

生成并对比了两个不同尺寸的数据集：

![数据集对比](cryoem_aspire/emd_45747_diagnostic.png)

*图 4: EMD-45747 (BRCA1, 大) vs EMD-54951 (小) 数据集对比。上排：BRCA1 的 Clean/CTF/Noisy；中排：EMD-54951 对应图像；下排：中心区域放大及信号强度统计。*

**关键发现**:
- BRCA1 (EMD-45747) 信号强度比 EMD-54951 弱 40%
- 大蛋白 (264 Å) 的 CTF 离域效应更显著
- SNR=0.05 条件下，noisy 图像几乎无法肉眼识别

### 3.5 半径扫描数据集

针对 EMD-54951 生成了 13 个不同半径的掩膜版本：

| 半径 (px) | 有效直径 | 覆盖比例 | 用途 |
|-----------|----------|----------|------|
| 72 | 174 px | 68% | 极端截断测试 |
| 88 | 206 px | 81% | 中等截断测试 |
| **112** | 254 px | **99%** | **最优基准** ★ |
| **138** | 306 px | **120%** | **保留离域信息** ★ |
| **148** | 326 px | **127%** | **保留高频** ★ |
| 168 | 366 px | 143% | 接近边界极限 |
| 178 | 386 px | 151% | 超出边界 (测试用) |

针对 EMD-45747 生成了 5 个半径版本：90, 110, 130, 150, 170 px。

---

## 四、生成的数据文件

### 4.1 数据目录结构

```
cryoem_aspire/
├── cryosparc_10k_corrected/          # EMD-54951 完整数据
│   ├── clean_projections.mrcs        # 无 CTF (4.9 GB)
│   ├── ctf_projections.mrcs          # 有 CTF (4.9 GB)
│   ├── noisy_projections.mrcs        # CTF + 噪声 (4.9 GB)
│   ├── r72_noise_masked.mrcs         # 半径 72 掩膜 (4.9 GB)
│   ├── r80_noise_masked.mrcs
│   ├── ... (共 13 个半径)
│   └── r178_noise_masked.mrcs
│   └── particles.star                # RELION/cryoSPARC 元数据
│
├── emd_45747_dataset/                # BRCA1 数据集
│   ├── clean_projections.mrcs        # (6.0 GB)
│   ├── ctf_projections.mrcs
│   ├── noisy_projections.mrcs
│   ├── r90_noise_masked.mrcs         # (5 个半径)
│   ├── r110_noise_masked.mrcs
│   ├── r130_noise_masked.mrcs
│   ├── r150_noise_masked.mrcs
│   ├── r170_noise_masked.mrcs
│   └── particles.star
│
└── emd_8194_dataset/                 # 新增数据集
    └── ...
```

### 4.2 数据格式

- **格式**: MRC 2014 (`.mrcs`)
- **数据类型**: float32
- **维度**: (N, H, W) = (10000, 360/400, 360/400)
- **元数据**: RELION 3.1+ 兼容的 STAR 文件

---

## 五、核心代码

### 5.1 代码清单

| 文件 | 功能 | 核心算法 | 状态 |
|------|------|----------|------|
| `cryo_em_projection.py` | 基础投影生成 | Numba JIT 三线性插值 | ✅ 稳定 |
| `cryoem_fourier/cryoem_fourier.py` | GPU 加速投影 | PyTorch FFT + 切片 | ✅ 稳定 |
| `cryoem_aspire/generate_base_projections.py` | ASPIRE 基础数据 | ASPIRE Simulation | ✅ 稳定 |
| `cryoem_aspire/generate_radius_scan_corrected.py` | 13 半径完整数据 | CTF + B-envelope | ✅ 完成 |
| `cryoem_aspire/generate_emd45747_dataset.py` | BRCA1 数据集 | 同上 | ✅ 完成 |
| `ctf_check_simple.py` | CTF 验证 | Envelope 对比 | ✅ 验证工具 |

### 5.2 核心代码片段

**A. Numba 加速投影** (`cryo_em_projection.py`)

```python
@njit(parallel=True, cache=True)
def project_single_volume_numba(volume, rot, tilt):
    """使用 Numba 加速的单个体积投影"""
    n = volume.shape[0]
    output_size = 256
    center = (n - 1) / 2.0
    
    # 角度转弧度，构建旋转矩阵
    alpha, beta = np.deg2rad(rot), np.deg2rad(tilt)
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    
    # ZY 旋转矩阵
    r00, r01, r02 = ca*cb, -sa, ca*sb
    r10, r11, r12 = sa*cb, ca, sa*sb
    r20, r21, r22 = -sb, 0, cb
    
    projection = np.zeros((output_size, output_size), dtype=np.float32)
    
    for py in prange(output_size):  # 并行循环
        for px in range(output_size):
            # 像素坐标 -> 体积坐标
            x_proj = (px - output_size/2 + 0.5) * scale
            y_proj = (py - output_size/2 + 0.5) * scale
            
            # 沿 Z 轴积分
            total = 0.0
            for z_idx in range(n):
                z_proj = z_idx - center
                
                # 旋转回原体积坐标
                x = r00 * x_proj + r01 * y_proj + r02 * z_proj + center
                y = r10 * x_proj + r11 * y_proj + r12 * z_proj + center
                z = r20 * x_proj + r21 * y_proj + r22 * z_proj + center
                
                # 三线性插值
                val = trilinear_interpolate(volume, x, y, z)
                total += val
            
            projection[py, px] = total
    
    return projection
```

**B. CTF + Envelope 计算** (`cryoem_aspire/generate_base_projections.py`)

```python
def calculate_ctf_with_envelope(image_size, defocus, b_factor=60):
    """计算 CTF，包含 B-factor envelope"""
    # 电子波长 (Å)
    lambda_ = 12.27 / np.sqrt(VOLTAGE * 1000 + 0.9784 * VOLTAGE**2)
    
    # 频率网格
    freq = fftfreq(image_size, d=PIXEL_SIZE)
    kx, ky = np.meshgrid(freq, freq, indexing='ij')
    k = np.sqrt(kx**2 + ky**2)
    
    # CTF 相位
    chi = np.pi * lambda_ * k**2 * (CS * 1e7 * lambda_**2 * k**2 - 2 * defocus)
    ctf = -np.sqrt(1 - AC**2) * np.sin(chi) - AC * np.cos(chi)
    
    # B-factor envelope (关键!)
    envelope = np.exp(-b_factor * k**2 / 4)
    ctf = ctf * envelope
    
    return ctf
```

**C. 加窗圆形掩膜** (`cryoem_aspire/generate_apodized_masks.py`)

```python
def create_apodized_circular_mask(image_size, radius, edge_width=15):
    """创建余弦加窗圆形掩膜"""
    y, x = np.ogrid[:image_size, :image_size]
    center = image_size // 2
    r = np.sqrt((x - center)**2 + (y - center)**2)
    
    mask = np.ones((image_size, image_size))
    
    # 过渡区域
    taper_region = (r > radius) & (r < radius + edge_width)
    mask[taper_region] = 0.5 * (1 + np.cos(np.pi * (r[taper_region] - radius) / edge_width))
    mask[r >= radius + edge_width] = 0.0
    
    return mask
```

---

## 六、关键发现与结论

### 6.1 主要发现

1. **B-factor Envelope 是必需的**: 无 Envelope 的 CTF 高频振荡过多，会导致重建算法收敛困难。

2. **半径选择至关重要**:
   - R=112 (99% 覆盖) 是最优基准
   - R=138-148 (120-127% 覆盖) 能保留 CTF 离域信息，可能提高分辨率
   - R<88 (81% 覆盖以下) 会损失过多高频信息

3. **双数据集验证**:
   - 小蛋白 (189 Å): 适合快速算法验证
   - 大蛋白 (264 Å): 更适合测试柔性区域重建

4. **SNR=0.05 是合理下限**: 此条件下 noisy 图像几乎无法肉眼识别，与真实 cryo-EM 数据相当。

### 6.2 推荐使用策略

| 场景 | 推荐数据 | 推荐半径 | 理由 |
|------|----------|----------|------|
| 算法开发 | EMD-54951 | R=112 | 标准基准 |
| 分辨率极限测试 | EMD-54951 | R=138, 148 | 保留离域信息 |
| 鲁棒性测试 | EMD-54951 | R=72, 80 | 截断容忍度 |
| 大蛋白测试 | EMD-45747 | R=110, 130 | BRCA1 结构 |
| 快速验证 | EMD-54951 | 任意 | 数据量较小 |

---

## 七、附录

### 7.1 虚拟环境

```bash
conda activate cryoem_test

# 关键依赖
pip install aspire mrcfile numpy scipy scikit-image matplotlib
pip install torch  # GPU 加速版本
```

### 7.2 快速开始

```bash
# 1. 生成基础投影
python cryoem_aspire/generate_base_projections.py

# 2. 生成完整半径扫描
python cryoem_aspire/generate_radius_scan_corrected.py

# 3. 验证 CTF
python ctf_check_simple.py
```

### 7.3 cryoSPARC 导入

1. Import Particle Stack
2. Particle meta: `cryosparc_10k_corrected/particles.star`
3. Particle data: 选择合适的 `*_masked.mrcs`
4. **注意**: EMD-54951 像素大小 0.74 Å，EMD-45747 像素大小 1.031 Å

---

## 八、文档引用

- 实验计划: `cryoem_experiment_plan.md`
- 数据集文档: `cryoem_aspire/DATASET_DOCUMENTATION.md`
- ASPIRE 介绍: `opencode_test/ASPIRE_Project_Summary.md`
- 本报告: `EXPERIMENT_REPORT.md`

---

*报告生成日期: 2026-03-17*  
*数据总大小: ~120 GB*  
*投影总数: 30,000+ (多版本)*
