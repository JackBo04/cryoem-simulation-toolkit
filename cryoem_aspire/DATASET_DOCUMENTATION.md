# Cryo-EM 模拟数据集文档

## 项目概述

本项目生成两组基于真实 cryo-EM 结构的模拟数据集，用于测试和验证冷冻电镜图像处理算法。数据集包含带 CTF 和 B-factor envelope 的投影图像，以及不同半径的加窗掩膜版本。

---

## 数据集汇总

| 数据集 | 模型 | 蛋白质 | 物理尺寸 | 图像尺寸 | 像素大小 | 状态 |
|--------|------|--------|----------|----------|----------|------|
| **Dataset 1** | EMD-54951 | 未知结构 (小蛋白) | ~189 Å | 360×360 px | 0.74 Å | ✅ 完成 (13个半径) |
| **Dataset 2** | EMD-45747 | BRCA1 (大复合物) | ~264 Å | 400×400 px | 1.031 Å | ✅ 完成 (5个半径) |

---

## Dataset 1: EMD-54951

### 模型信息
- **EMD ID**: [EMD-54951](https://www.ebi.ac.uk/emdb/EMD-54951)
- **原始尺寸**: 320×320×320 voxels @ 0.592 Å/pixel
- **物理尺寸**: 189.4 Å
- **分子量**: 较小蛋白 (具体信息需查询 EMDB)

### 数据处理流程
1. **Downsample**: 320 → 256 (ASPIRE)
2. **物理尺寸保持**: 189.4 Å (不变)
3. **计算像素大小**: 189.4/256 = **0.74 Å** ✅
4. **Padding**: 256×256 → **360×360** (52px 每边)
5. **生成投影**: 10,000 张

### 图像参数
| 参数 | 值 |
|------|-----|
| 图像尺寸 | 360×360 px |
| 像素大小 | **0.74 Å** |
| 粒子直径 | ~256 px (~189 Å) |
| 粒子半径 | ~128 px |
| 电压 | 300 kV |
| 球差 (Cs) | 2.7 mm |
| 振幅对比度 | 0.1 |
| 欠焦范围 | 1.0-2.0 μm |
| B-factor | 60 Å² |
| SNR | 0.05 |
| 加窗宽度 | 15 px |

### 半径扫描 (13个半径)

从紧包到扩散，覆盖不同场景：

| 半径 | 有效直径 | 覆盖比例 | 状态 | 用途 |
|------|----------|----------|------|------|
| 72 | 174 px | 68% | ✅ | 极端截断测试 |
| 80 | 190 px | 74% | ✅ | 严重截断测试 |
| 88 | 206 px | 81% | ✅ | 中等截断测试 |
| 96 | 222 px | 87% | ✅ | 轻微截断测试 |
| 104 | 238 px | 93% | ✅ | 接近完整 ★ |
| 112 | 254 px | 99% | ✅ | 最优范围 ★ |
| 120 | 270 px | 105% | ✅ | 包含少量背景 |
| 128 | 286 px | 112% | ✅ | 包含背景 |
| 138 | 306 px | 120% | 🔵 | 保留离域信息 ★ |
| 148 | 326 px | 127% | 🔵 | 保留高频信息 ★ |
| 158 | 346 px | 135% | 🟠 | 临界 (7px余量) |
| 168 | 366 px | 143% | 🔴 | 超出边界 3px |
| 178 | 386 px | 151% | 🔴 | 超出边界 13px |

### 生成的文件

**位置**: `cryosparc_10k_corrected/`

**基础投影** (~15 GB):
- `clean_projections.mrcs` (4.9 GB) - 无 CTF
- `ctf_projections.mrcs` (4.9 GB) - CTF + B-factor envelope
- `noisy_projections.mrcs` (4.9 GB) - CTF + 噪声

**掩膜数据** (13个 × 4.9 GB = ~64 GB):
- `r72_noise_masked.mrcs` 到 `r178_noise_masked.mrcs`

**元数据**:
- `particles.star` - RELION/cryoSPARC 格式元数据

**可视化**:
- `masks_all_radii.png` - 所有掩膜对比
- `masks_all_cross_sections.png` - 截面曲线
- `projection_comparison.png` - 投影对比
- `particle_region_analysis.png` - 粒子区域分析
- `ctf_envelope_analysis.png` - CTF envelope分析

---

## Dataset 2: EMD-45747 (BRCA1)

### 模型信息
- **EMD ID**: [EMD-45747](https://www.ebi.ac.uk/emdb/EMD-45747)
- **蛋白质**: BRCA1 (乳腺癌易感基因1蛋白)
- **原始尺寸**: 320×320×320 voxels @ 0.825 Å/pixel
- **物理尺寸**: 264.0 Å
- **分子量**: 大蛋白复合物 (~220 kDa)
- **结构特点**: 包含多个结构域的大型柔性复合物

### 数据处理流程
1. **Downsample**: 320 → 256 (ASPIRE)
2. **物理尺寸保持**: 264.0 Å (不变)
3. **计算像素大小**: 264/256 = **1.031 Å** ✅
4. **Padding**: 256×256 → **400×400** (72px 每边)
5. **生成投影**: 10,000 张

### 图像参数
| 参数 | 值 |
|------|-----|
| 图像尺寸 | 400×400 px |
| 像素大小 | **1.031 Å** |
| 粒子直径 | ~256 px (~264 Å) |
| 粒子半径 | ~128 px |
| 电压 | 300 kV |
| 球差 (Cs) | 2.7 mm |
| 振幅对比度 | 0.1 |
| 欠焦范围 | 1.0-2.0 μm |
| B-factor | 60 Å² |
| SNR | 0.05 |
| 加窗宽度 | 15 px |

### 半径扫描 (5个半径)

针对更大粒子的优化选择：

| 半径 | 有效直径 | 覆盖比例 | 到边界 | 用途 |
|------|----------|----------|--------|------|
| 90 | 210 px | 82% | 95 px | 中等截断测试 |
| 110 | 250 px | 98% | 75 px | ★ 接近完整 (Baseline) |
| 130 | 290 px | 113% | 55 px | ★ 保留离域信息 |
| 150 | 330 px | 129% | 35 px | 大范围测试 |
| 170 | 370 px | 145% | 15 px | 接近边界极限 |

### 生成的文件

**位置**: `emd_45747_dataset/`

**基础投影** (~18 GB):
- `clean_projections.mrcs` (6.0 GB) - 无 CTF
- `ctf_projections.mrcs` (6.0 GB) - CTF + B-factor envelope
- `noisy_projections.mrcs` (6.0 GB) - CTF + 噪声

**掩膜数据** (5个 × 6.0 GB = ~30 GB):
- `r90_noise_masked.mrcs`
- `r110_noise_masked.mrcs`
- `r130_noise_masked.mrcs`
- `r150_noise_masked.mrcs`
- `r170_noise_masked.mrcs`

**元数据**:
- `particles.star` - RELION/cryoSPARC 格式元数据

**可视化**:
- `dataset_overview.png` - 完整数据概览

---

## 两组数据集对比

| 特征 | EMD-54951 | EMD-45747 (BRCA1) |
|------|-----------|-------------------|
| **物理尺寸** | 189 Å | 264 Å (+40%) |
| **图像尺寸** | 360×360 px | 400×400 px |
| **像素大小** | 0.74 Å | 1.031 Å |
| **粒子直径(像素)** | 256 px | 256 px |
| **图像边界到粒子边缘** | 52 px | 72 px |
| **半径范围** | 72-178 px (13个) | 90-170 px (5个) |
| **扫参密度** | 密集 (研究离域) | 精简 (关键节点) |
| **总数据量** | ~79 GB | ~36 GB |

### 关键差异

1. **物理尺寸**: EMD-45747 比 EMD-54951 大 40%
2. **像素大小**: 由于物理尺寸不同，像素大小也不同
3. **半径策略**: 
   - EMD-54951: 密集扫描，包含超出边界的极端情况
   - EMD-45747: 精简选择，聚焦关键覆盖比例

---

## 生成脚本

### 主脚本列表

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `generate_radius_scan_corrected.py` | 生成 EMD-54951 完整数据集 | EMD-54951.map | Dataset 1 (13半径) |
| `generate_base_projections.py` | 生成基础投影 (clean/CTF/noisy) | EMD-54951.map | 基础投影文件 |
| `generate_large_radii.py` | 生成 EMD-54951 大半径 (138-178) | 已有基础投影 | r138-r178 掩膜 |
| `generate_emd45747_dataset.py` | 生成 EMD-45747 完整数据集 | EMD-45747.map | Dataset 2 (5半径) |

### 脚本特点

1. **ASPIRE 集成**: 使用 ASPIRE 库生成投影
2. **CTF 模拟**: 完整 CTF 计算 + B-factor envelope
3. **噪声模型**: 基于 SNR 的高斯噪声
4. **加窗掩膜**: 余弦渐变边缘 (apodization)
5. **MRC 格式**: 标准 MRCS 格式，兼容 cryoSPARC/RELION

### 关键代码模块

```python
# CTF 计算 (含 B-factor envelope)
def calculate_ctf_with_envelope(image_size, defocus, b_factor=60):
    lambda_ = 12.27 / np.sqrt(VOLTAGE * 1000 + 0.9784 * VOLTAGE**2)
    freq = fftfreq(image_size, d=PIXEL_SIZE)
    k = np.sqrt(kx**2 + ky**2)
    chi = np.pi * lambda_ * k**2 * (CS * 1e7 * lambda_**2 * k**2 - 2 * defocus)
    ctf = -np.sqrt(1 - AC**2) * np.sin(chi) - AC * np.cos(chi)
    envelope = np.exp(-b_factor * k**2 / 4)  # B-factor envelope
    return ctf * envelope

# 加窗掩膜
def create_apodized_circular_mask(image_size, radius, edge_width=15):
    mask = np.ones((image_size, image_size))
    taper_region = (r > radius) & (r < radius + edge_width)
    mask[taper_region] = 0.5 * (1 + np.cos(np.pi * (r - radius) / edge_width))
    mask[r >= radius + edge_width] = 0.0
    return mask
```

---

## 虚拟环境

### 环境名称
```
cryoem_test
```

### 主要依赖
- **Python**: 3.x
- **ASPIRE**: 用于生成投影
- **mrcfile**: MRC 文件读写
- **NumPy/SciPy**: 数值计算
- **scikit-image**: 图像处理 (Otsu 阈值等)

### 激活环境
```bash
conda activate cryoem_test
```

### 运行脚本示例
```bash
# EMD-54951 完整生成
conda run -n cryoem_test python generate_radius_scan_corrected.py

# EMD-45747 生成
conda run -n cryoem_test python generate_emd45747_dataset.py
```

---

## CTF 与加窗参数

### CTF 参数 (两组数据集相同)

| 参数 | 值 | 说明 |
|------|-----|------|
| 加速电压 | 300 kV | 典型冷冻电镜电压 |
| 球差 (Cs) | 2.7 mm | 标准值 |
| 振幅对比度 | 0.1 | 典型值 (0.07-0.1) |
| 欠焦范围 | 1.0-2.0 μm | 典型冷冻电镜欠焦 |
| B-factor | 60 Å² | 高分辨率信号衰减 |
| SNR | 0.05 | 低剂量条件 |

### B-factor Envelope 效果

```
E(k) = exp(-B * k² / 4)
```

| 分辨率 | Envelope 值 | 说明 |
|--------|-------------|------|
| 10 Å | 0.86 | 高信度 |
| 6 Å | 0.66 | 中等信度 |
| 4 Å | 0.39 | 低信度 |
| 3 Å | 0.19 | 几乎无信号 |

### 加窗 (Apodization) 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 边缘宽度 | 15 px | 余弦渐变宽度 |
| 函数 | 0.5*(1+cos(π*(r-R)/w)) | 平滑过渡到0 |
| 外部填充 | 高斯噪声 | 同方差噪声 |

---

## 使用指南

### cryoSPARC 导入

1. **Import Particle Stack**
2. **Particle meta**: `particles.star`
3. **Particle data**: 选择合适的 `*_masked.mrcs`
4. **像素大小**:
   - EMD-54951: **0.74 Å**
   - EMD-45747: **1.031 Å**

### 推荐测试策略

#### 对于 EMD-54951 (小蛋白，密集扫描)

| 阶段 | 半径 | 目的 |
|------|------|------|
| Baseline | R=104, 112 | 建立性能基准 |
| 截断测试 | R=72, 80, 88, 96 | 算法鲁棒性 |
| 离域效应 | R=138, 148 | 保留高频信息 ★ |
| 极限测试 | R=168, 178 | 超出边界情况 |

#### 对于 EMD-45747 (大蛋白，精选半径)

| 阶段 | 半径 | 目的 |
|------|------|------|
| Baseline | R=110 | 98% 覆盖，最优基准 ★ |
| 离域保留 | R=130 | 113% 覆盖，保留CTF离域 ★ |
| 截断测试 | R=90 | 82% 覆盖，测试容忍度 |
| 背景测试 | R=150, 170 | 大范围，含背景噪声 |

---

## 注意事项

### ⚠️ 重要提醒

1. **像素大小不同**:
   - 两组数据集的像素大小不同 (0.74 Å vs 1.031 Å)
   - 导入 cryoSPARC 时务必设置正确！

2. **离域效应 (Delocalization)**:
   - CTF 导致高频信息分散在粒子外
   - 小半径掩膜会损失高频信息
   - 建议：为保留 5Å 信息，需要 R ≥ 130-140

3. **边界限制**:
   - 大半径 (R>160) 可能接近或超出图像边界
   - R=168, 178 在 EMD-54951 中已超出边界

4. **B-factor Envelope**:
   - 所有投影都包含 B=60 Å² 的 envelope
   - 高频信号已被衰减

---

## 技术细节

### 文件格式

**MRCS 文件**:
- 格式: MRC 2014 格式
- 数据类型: float32
- 维度: (N, H, W) = (10000, 360/400, 360/400)
- Header cella: 物理尺寸 (Å)

**STAR 文件**:
- 格式: RELION 3.1+ 格式
- 字段: 标准 cryoSPARC 兼容字段
- 角度: Rot, Tilt, Psi (度)
- 欠焦: DefocusU, DefocusV, DefocusAngle

### 计算资源

| 数据集 | 生成时间 | 内存需求 | 存储需求 |
|--------|----------|----------|----------|
| EMD-54951 | ~30 min | ~20 GB | ~79 GB |
| EMD-45747 | ~20 min | ~20 GB | ~36 GB |

---

## 引用与参考

### 数据库条目
- **EMD-54951**: https://www.ebi.ac.uk/emdb/EMD-54951
- **EMD-45747**: https://www.ebi.ac.uk/emdb/EMD-45747

### 相关论文
- (需根据具体结构添加相关文献)

### 软件
- **ASPIRE**: https://github.com/ComputationalCryoEM/ASPIRE-Python
- **cryoSPARC**: https://cryosparc.com
- **RELION**: https://relion.readthedocs.io

---

## 作者与联系

- **生成日期**: 2026-03-08
- **生成脚本**: /mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/cryoem_aspire/*.py
- **虚拟环境**: `cryoem_test` (conda)

---

## 附录: 快速参考卡

### EMD-54951 快速导入
```bash
# 像素大小: 0.74 Å
# 推荐半径: R=112 (baseline), R=138-148 (delocalization)
# 数据位置: cryosparc_10k_corrected/
```

### EMD-45747 快速导入
```bash
# 像素大小: 1.031 Å
# 推荐半径: R=110 (baseline), R=130 (delocalization)
# 数据位置: emd_45747_dataset/
```

### 文件命名规范
```
{r}_{fill}_masked.mrcs
  r: 半径 (72, 80, ..., 178 或 90, 110, ..., 170)
  fill: 填充模式 (noise = 高斯噪声)
```
