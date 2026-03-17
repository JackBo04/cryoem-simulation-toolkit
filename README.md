# Cryo-EM 仿真实验核心代码包

> **版本**: v1.0  
> **日期**: 2026-03-17  
> **作者**: 实验团队  
> **用途**: 冷冻电镜仿真数据生成与 CTF 效应研究

---

## 📦 包内容说明

本代码包包含 cryo-EM 仿真实验的核心代码和文档，用于生成带 CTF 效应的蛋白质投影数据，支持多半径掩膜扫描。

```
cryoem_core_package/
├── README.md                          # 本文件
├── EXPERIMENT_REPORT.md               # 完整实验报告
├── cryoem_experiment_plan.md          # 实验计划文档
│
├── cryo_em_projection.py              # 核心：Numba加速投影生成
├── ctf_check_simple.py                # CTF Envelope 效果验证
├── ctf_simple_check.py                # CTF 参数简化验证
├── ctf_detailed_comparison.py         # CTF 详细对比分析
│
├── cryoem_aspire/                     # ASPIRE 方法实现
│   ├── generate_base_projections.py   # 基础投影生成 (clean/CTF/noisy)
│   ├── generate_radius_scan_corrected.py  # 13半径完整扫描 (EMD-54951)
│   ├── generate_emd45747_dataset.py   # BRCA1 数据集生成
│   ├── generate_emd8194_dataset.py    # EMD-8194 数据集生成
│   ├── generate_apodized_masks.py     # 加窗掩膜生成
│   └── DATASET_DOCUMENTATION.md       # 数据集文档
│
└── cryoem_fourier/                    # 傅里叶切片方法 (GPU加速)
    ├── cryoem_fourier.py              # PyTorch GPU 实现
    ├── cryoem_simple.py               # 简化版本
    └── README.md                      # 方法说明
```

---

## 🚀 快速开始

### 1. 环境要求

```bash
# Python >= 3.9
# 推荐环境: conda

conda create -n cryoem_test python=3.9
conda activate cryoem_test

# 安装依赖
pip install numpy scipy mrcfile scikit-image matplotlib
pip install numba  # 用于 CPU 加速
pip install torch  # 用于 GPU 加速 (可选)
pip install aspire  # ASPIRE 库 (生成完整数据集需要)
```

### 2. 快速运行

**A. 生成基础投影 (Numba 加速)**
```bash
python cryo_em_projection.py
```
输出: `clean_projections.npz` (纯净投影 + 欧拉角)

**B. 验证 CTF Envelope 效果**
```bash
python ctf_check_simple.py
```
输出: `ctf_simple_comparison.png` (对比图)

**C. 生成完整数据集 (需要 ASPIRE)**
```bash
# EMD-54951 (小蛋白, 13个半径)
python cryoem_aspire/generate_radius_scan_corrected.py

# EMD-45747 (BRCA1, 5个半径)
python cryoem_aspire/generate_emd45747_dataset.py
```

**D. GPU 加速版本 (PyTorch)**
```bash
python cryoem_fourier/cryoem_fourier.py
```

---

## 📊 核心功能

### 1. 投影生成 (`cryo_em_projection.py`)

- **算法**: Numba JIT + 并行三线性插值
- **速度**: 生成 10,000 张 256×256 投影约 5-10 分钟
- **特点**: 
  - 半球欧拉角采样 (Rot: 0-360°, Tilt: 0-90°)
  - 圆形软掩膜 (边缘渐变)
  - 灰色背景 (避免边界伪影)

### 2. CTF 模拟

关键参数:
| 参数 | 值 | 说明 |
|------|-----|------|
| 电压 | 300 kV | 加速电压 |
| Cs | 2.7 mm | 球差 |
| AC | 0.1 | 振幅对比度 |
| 欠焦 | 1.0-2.0 μm | 随机分布 |
| B-factor | 60 Å² | 信号衰减包络 |
| SNR | 0.05 | 信噪比 |

**B-factor Envelope 公式**:
```
E(k) = exp(-B·k²/4)
```

### 3. 半径扫描

针对 EMD-54951 的 13 个半径配置:

| 半径 (px) | 覆盖比例 | 推荐用途 |
|-----------|----------|----------|
| 72 | 68% | 极端截断测试 |
| 112 | **99%** | **最优基准** ★ |
| 138 | **120%** | **保留离域信息** ★ |
| 148 | **127%** | **保留高频信息** ★ |
| 178 | 151% | 边界极限测试 |

---

## 📖 文档说明

| 文档 | 内容 |
|------|------|
| `EXPERIMENT_REPORT.md` | 完整实验报告，包含结果分析、图片、结论 |
| `cryoem_experiment_plan.md` | 实验设计文档，5个Task的详细说明 |
| `cryoem_aspire/DATASET_DOCUMENTATION.md` | 数据集详细文档，参数、格式、使用指南 |
| `cryoem_fourier/README.md` | 傅里叶切片方法说明 |

---

## 🔬 实验结果摘要

### 主要发现

1. **B-factor Envelope 是必需的**
   - 无 Envelope 时 CTF 高频振荡过多，不符合物理实际
   - B=60 Å² 在 4 Å 分辨率处信号衰减至 39%

2. **最优半径 R=112** (99% 覆盖)
   - 完整包含蛋白质信号
   - 不过度包含背景噪声

3. **离域信息保留 R=138-148** (120-127% 覆盖)
   - CTF 导致高频信息分散在粒子外
   - 更大半径有助于保留高分辨率信息

### 数据集统计

| 数据集 | 蛋白质 | 图像尺寸 | 像素大小 | 半径数 | 总大小 |
|--------|--------|----------|----------|--------|--------|
| EMD-54951 | 小蛋白 | 360×360 | 0.74 Å | 13 | ~79 GB |
| EMD-45747 | BRCA1 | 400×400 | 1.031 Å | 5 | ~36 GB |

---

## 💡 使用建议

### 场景 1: 快速验证算法
```bash
python cryo_em_projection.py  # 生成 10k 纯净投影
```

### 场景 2: 测试 CTF 效应
```bash
python ctf_check_simple.py  # 可视化 CTF 效果
python cryoem_aspire/generate_base_projections.py  # 生成 CTF 数据
```

### 场景 3: 半径影响研究
```bash
python cryoem_aspire/generate_radius_scan_corrected.py  # 完整 13 半径
```

### 场景 4: 大蛋白测试
```bash
python cryoem_aspire/generate_emd45747_dataset.py  # BRCA1
```

---

## ⚠️ 注意事项

1. **数据路径**: 代码中使用了绝对路径 `/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/`，使用前请修改为本地路径

2. **内存需求**: 
   - 生成 10k 投影需要 ~20 GB 内存
   - 建议分批生成或降低分辨率

3. **ASPIRE 依赖**: 完整数据集生成需要安装 ASPIRE 库
   ```bash
   pip install aspire
   ```

4. **GPU 支持**: 傅里叶切片方法支持 CUDA 加速，需安装 PyTorch GPU 版本

---

## 📂 输出文件格式

- **格式**: MRC 2014 (`.mrcs`)
- **数据类型**: float32
- **维度**: (N, H, W) = (10000, 360/400, 360/400)
- **元数据**: RELION/cryoSPARC 兼容 STAR 文件

---

## 📚 引用与参考

- **EMDB**: https://www.ebi.ac.uk/emdb/
  - EMD-54951, EMD-45747, EMD-8194
- **ASPIRE**: https://github.com/ComputationalCryoEM/ASPIRE-Python
- **cryoSPARC**: https://cryosparc.com
- **RELION**: https://relion.readthedocs.io

---

## 📧 联系与支持

如有问题，请参考:
1. `EXPERIMENT_REPORT.md` - 详细实验结果
2. `cryoem_aspire/DATASET_DOCUMENTATION.md` - 数据集文档
3. 各代码文件的头部注释

---

*Generated: 2026-03-17*  
*Total Files: 13 code files + 4 docs*
