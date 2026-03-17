# Cryo-EM Particle Simulation - Fourier Slice Method

## 方法说明

使用 **傅里叶切片定理 (Fourier Slice Theorem)** 自实现投影生成。

**核心原理:**
- 3D 体积的 FFT 中心切片 = 2D 投影的 FFT
- 通过旋转频率坐标提取中心切片
- IFFT 得到实空间投影

**优势:**
- GPU 加速 (CUDA)
- 批处理支持
- 速度快 (~10-50x 快于传统方法)
- 完全可控的参数

## 依赖

```
# 核心依赖
numpy>=1.21
scipy
mrcfile>=1.0
scikit-image

# PyTorch (关键依赖)
torch>=2.0
```

## 安装

```bash
# CPU 版本
pip install -r requirements.txt

# GPU 版本 (需要 CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 使用方法

```bash
# 运行完整 pipeline (自动检测 GPU)
python cryoem_fourier.py

# 或自定义配置
python -c "
from cryoem_fourier import Config, run_pipeline

config = Config()
config.N_PROJECTIONS = 1000
config.IMAGE_SIZE = 128
config.DEVICE = 'cpu'  # 强制 CPU

run_pipeline(config)
"
```

## 输出

运行后会在 `output/` 目录下生成:
- `noisy_test.mrcs` - 含噪图像栈
- `masked_test.mrcs` - 掩膜处理后的图像  
- `ground_truth.star` - RELION 格式元数据

## 核心算法

```python
# Fourier Slice Theorem 实现
def project_fourier_slice(volume, rotation_matrix):
    # 1. 3D FFT
    vol_fft = fftn(volume)
    
    # 2. 提取中心切片 (旋转后)
    slice_fft = extract_central_slice(vol_fft, R)
    
    # 3. 2D IFFT
    projection = irfftn(slice_fft)
    
    return real(projection)
```

## 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| N_PROJECTIONS | 10000 | 投影数量 |
| IMAGE_SIZE | 256 | 图像尺寸 |
| DEVICE | cuda/cpu | 计算设备 |
| PAD_FACTOR | 1.5 | FFT  padding (减少边缘伪影) |
| SNR | 0.05 | 信噪比 |
| MASK_SIGMA | 4.0 | 软掩膜 sigma |

## 数据准备

将 `.mrc` 文件放入 `data/` 目录:
- 默认: `data/apoferritin.mrc`
- 程序会自动创建测试体积如果文件不存在

## 性能基准

| 设备 | 1000 投影 | 10000 投影 |
|------|-----------|------------|
| CPU | ~30s | ~5min |
| GPU | ~5s | ~30s |

## 与 ASPIRE 方法对比

| 特性 | ASPIRE | Fourier Slice |
|------|--------|---------------|
| 速度 | 中等 | **快 5-50x** |
| GPU | 不支持 | **支持** |
| 依赖 | **重** | 轻量 |
| 可定制性 | 有限 | **完全可控** |
| 上手难度 | 简单 | 中等 |
