"""
Generate Apodized Circular Masks with Two Background Fill Methods
=================================================================
基于 generate_10k_with_envelope.py 的生成方式，修改掩膜部分：

1. Apodized Mask: 余弦渐变边缘 (cosine taper)，边缘宽度 10-20 px
2. 两种背景填充方式：
   - outside_zero: 外部置 0 (传统方式)
   - outside_noise: 外部填充同方差高斯噪声

生成4组数据：
- r=128, outside_zero
- r=128, outside_noise  
- r=100, outside_zero
- r=100, outside_noise

其他参数保持与 generate_10k_with_envelope.py 一致
"""

import numpy as np
import mrcfile
from pathlib import Path
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.filters import threshold_otsu
from aspire.volume import Volume
from aspire.source import Simulation
import time

OUTPUT_DIR = Path("/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/cryoem_aspire/cryosparc_10k_apodized")
OUTPUT_DIR.mkdir(exist_ok=True)

MAP_FILE = "/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/emd_54951.map"

# 参数 (与 generate_10k_with_envelope.py 一致)
N_PROJECTIONS = 10000
IMAGE_SIZE = 256
PIXEL_SIZE = 1.06
VOLTAGE = 300
CS = 2.7
AC = 0.1  # 振幅对比度
SNR = 0.05
SEED = 42
B_FACTOR = 60  # B-factor in Å²

# Soft mask 参数 (用于生成 oracle soft masks，非 apodized mask)
MASK_GAUSSIAN_SIGMA = 4.0
MASK_DILATION_ITER = 1

# Apodized mask 参数
APODIZATION_WIDTH = 15  # 余弦渐变边缘宽度 (像素)


def calculate_ctf_with_envelope(image_size, defocus, b_factor=B_FACTOR):
    """计算CTF，包含B-factor envelope"""
    lambda_ = 12.27 / np.sqrt(VOLTAGE * 1000 + 0.9784 * VOLTAGE**2)
    freq = fftfreq(image_size, d=PIXEL_SIZE)
    kx, ky = np.meshgrid(freq, freq, indexing='ij')
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1e-10
    
    # CTF相位
    chi = np.pi * lambda_ * k**2 * (CS * 1e7 * lambda_**2 * k**2 - 2 * defocus)
    # 完整CTF（含振幅对比度）
    ctf = -np.sqrt(1 - AC**2) * np.sin(chi) - AC * np.cos(chi)
    
    # B-factor envelope
    envelope = np.exp(-b_factor * k**2 / 4)
    ctf = ctf * envelope
    
    return ctf


def create_apodized_circular_mask(image_size, radius, edge_width=APODIZATION_WIDTH):
    """
    创建带余弦渐变边缘的圆形掩膜 (apodized circular mask)
    
    Parameters:
    -----------
    image_size : int
        图像尺寸
    radius : float
        圆半径（中心到余弦渐变开始位置）
    edge_width : float
        余弦渐变边缘宽度（从1降到0）
    
    Returns:
    --------
    mask : np.ndarray
        2D apodized mask, 值域 [0, 1]
    """
    y, x = np.ogrid[-image_size//2:image_size//2, -image_size//2:image_size//2]
    r = np.sqrt(x**2 + y**2)
    
    # Apodization: 余弦渐变
    # r <= radius: mask = 1
    # radius < r < radius + edge_width: mask = cosine taper
    # r >= radius + edge_width: mask = 0
    
    mask = np.ones((image_size, image_size), dtype=np.float32)
    
    # 渐变区域
    taper_region = (r > radius) & (r < radius + edge_width)
    # 余弦渐变: 0.5 * (1 + cos(π * (r - radius) / edge_width))
    mask[taper_region] = 0.5 * (1 + np.cos(np.pi * (r[taper_region] - radius) / edge_width))
    
    # 外部区域置0
    mask[r >= radius + edge_width] = 0.0
    
    return mask


def generate_oracle_soft_masks(projections_clean, sigma=4.0, dilation_iter=1):
    """生成 Oracle Soft Masks (与原始脚本一致)"""
    print(f"[Mask] Generating Oracle Soft Masks (sigma={sigma}, dilation={dilation_iter})...")
    
    n, h, w = projections_clean.shape
    soft_masks = np.zeros((n, h, w), dtype=np.float32)
    
    for i in range(n):
        proj = projections_clean[i]
        
        try:
            thresh = threshold_otsu(proj)
        except:
            thresh = proj.mean() + 0.5 * proj.std()
        
        binary_mask = (proj > thresh).astype(np.float32)
        
        if dilation_iter > 0:
            binary_mask = binary_dilation(binary_mask, iterations=dilation_iter).astype(np.float32)
        
        soft_mask = gaussian_filter(binary_mask, sigma=sigma)
        soft_masks[i] = np.clip(soft_mask, 0, 1)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n} masks")
    
    print(f"[Mask] Done: range=[{soft_masks.min():.4f}, {soft_masks.max():.4f}]")
    return soft_masks


def generate_base_data():
    """生成基础数据 (clean, noisy, oracle masked)"""
    start_time = time.time()
    
    print("=" * 70)
    print("Generate Base Data (10,000 projections with B-factor envelope)")
    print("=" * 70)
    print(f"Number: {N_PROJECTIONS}")
    print(f"Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Pixel Size: {PIXEL_SIZE} Å")
    print(f"Defocus Range: 1.0-2.0 μm")
    print(f"SNR: {SNR}")
    print(f"CTF: Full CTF with B-factor envelope (B={B_FACTOR}Å²)")
    print("=" * 70)
    
    # 1. Load volume
    print("\n[1/6] Loading volume...")
    t0 = time.time()
    vol = Volume.load(MAP_FILE, dtype=np.float32)
    vol = vol.downsample(IMAGE_SIZE)
    print(f"      Volume shape: {vol.shape}, Time: {time.time()-t0:.2f}s")
    
    # 2. Generate angles
    print("\n[2/6] Generating angles...")
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    angles = np.zeros((N_PROJECTIONS, 3))
    angles[:, 0] = rng.uniform(0, 360, N_PROJECTIONS)   # Rot
    angles[:, 1] = rng.uniform(0, 90, N_PROJECTIONS)    # Tilt (hemisphere)
    angles[:, 2] = rng.uniform(0, 360, N_PROJECTIONS)   # Psi
    angles_rad = np.deg2rad(angles)
    print(f"      Rot: {angles[:, 0].min():.1f}°-{angles[:, 0].max():.1f}°")
    print(f"      Tilt: {angles[:, 1].min():.1f}°-{angles[:, 1].max():.1f}°")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # 3. Generate clean projections
    print("\n[3/6] Generating clean projections...")
    t0 = time.time()
    src = Simulation(
        L=IMAGE_SIZE,
        n=N_PROJECTIONS,
        vols=vol,
        angles=angles_rad,
        seed=SEED,
        dtype=np.float32,
    )
    proj_clean = src.projections[:].asnumpy()
    print(f"      Projections shape: {proj_clean.shape}")
    print(f"      Value range: [{proj_clean.min():.4f}, {proj_clean.max():.4f}]")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # 4. Generate defocus values
    print("\n[4/6] Generating defocus values...")
    t0 = time.time()
    rng = np.random.default_rng(SEED + 1)
    defocus = rng.uniform(1.0e4, 2.0e4, N_PROJECTIONS)  # 1.0-2.0 μm
    defocus_v = defocus * (0.95 + 0.05 * rng.random(N_PROJECTIONS))
    defocus_angle = rng.uniform(0, 180, N_PROJECTIONS)
    print(f"      Defocus range: {defocus.min()/1e4:.2f}-{defocus.max()/1e4:.2f} μm")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # 5. Apply CTF with envelope
    print("\n[5/6] Applying CTF with B-factor envelope...")
    t0 = time.time()
    
    proj_ctf = np.zeros_like(proj_clean)
    for i in range(N_PROJECTIONS):
        ctf = calculate_ctf_with_envelope(IMAGE_SIZE, defocus[i])
        proj_fft = fft2(proj_clean[i])
        proj_ctf[i] = np.real(ifft2(proj_fft * ctf))
        
        if (i + 1) % 2000 == 0:
            print(f"      Processed {i + 1}/{N_PROJECTIONS}")
    
    print(f"      CTF range: [{proj_ctf.min():.4f}, {proj_ctf.max():.4f}]")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # 6. Add noise
    print("\n[6/6] Adding noise (SNR=0.05)...")
    t0 = time.time()
    rng = np.random.default_rng(SEED + 2)
    
    proj_noisy = np.zeros_like(proj_ctf)
    noise_per_image = []  # 保存每帧的噪声，用于后续outside_noise填充
    
    for i in range(N_PROJECTIONS):
        signal_power = np.mean(proj_ctf[i]**2)
        noise_var = signal_power / SNR
        noise = rng.normal(0, np.sqrt(noise_var), proj_ctf[i].shape)
        proj_noisy[i] = proj_ctf[i] + noise
        noise_per_image.append(noise)
        
        if (i + 1) % 2000 == 0:
            print(f"      Processed {i + 1}/{N_PROJECTIONS}")
    
    print(f"      Noisy range: [{proj_noisy.min():.4f}, {proj_noisy.max():.4f}]")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Base data generation complete! Time: {total_time/60:.1f} min")
    print("=" * 70)
    
    return {
        "proj_clean": proj_clean,
        "proj_ctf": proj_ctf,
        "proj_noisy": proj_noisy,
        "noise_per_image": noise_per_image,  # 用于 outside_noise 填充
        "defocus": defocus,
        "defocus_v": defocus_v,
        "defocus_angle": defocus_angle,
        "angles": angles,
    }


def apply_apodized_masks(base_data):
    """应用 apodized masks，生成4组数据"""
    proj_noisy = base_data["proj_noisy"]
    noise_per_image = base_data["noise_per_image"]
    
    configs = [
        ("r128_zero", 128, "zero"),
        ("r128_noise", 128, "noise"),
        ("r100_zero", 100, "zero"),
        ("r100_noise", 100, "noise"),
    ]
    
    results = {}
    
    for config_name, radius, fill_mode in configs:
        print(f"\n{'='*70}")
        print(f"Generating: {config_name} (radius={radius}, fill={fill_mode})")
        print(f"{'='*70}")
        
        # 创建 apodized mask
        mask = create_apodized_circular_mask(IMAGE_SIZE, radius, APODIZATION_WIDTH)
        print(f"Mask: apodized circle r={radius}, edge_width={APODIZATION_WIDTH}")
        print(f"Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
        
        # 应用 mask
        n = len(proj_noisy)
        masked_data = np.zeros_like(proj_noisy)
        
        for i in range(n):
            if fill_mode == "zero":
                # 外部置 0
                masked_data[i] = proj_noisy[i] * mask
            else:
                # 外部填充同方差高斯噪声
                # 内部：保留带噪声的信号
                inside = proj_noisy[i] * mask
                # 外部：填充与信号内部同方差的高斯噪声
                # 噪声的方差应该与添加到信号中的噪声方差相同
                signal_power = np.mean(base_data["proj_ctf"][i]**2)
                noise_var = signal_power / SNR
                outside_noise = np.random.normal(0, np.sqrt(noise_var), proj_noisy[i].shape)
                outside = outside_noise * (1 - mask)
                
                masked_data[i] = inside + outside
            
            if (i + 1) % 2000 == 0:
                print(f"  Processed {i + 1}/{n}")
        
        print(f"Result range: [{masked_data.min():.4f}, {masked_data.max():.4f}]")
        
        # 保存
        output_path = OUTPUT_DIR / f"{config_name}_masked.mrcs"
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            mrc.set_data(masked_data.astype(np.float32))
            mrc.header.cella.x = IMAGE_SIZE * PIXEL_SIZE
            mrc.header.cella.y = IMAGE_SIZE * PIXEL_SIZE
            mrc.header.cella.z = N_PROJECTIONS * PIXEL_SIZE
        
        print(f"Saved: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1e9:.2f} GB")
        
        # 保存 mask 样本
        mask_path = OUTPUT_DIR / f"{config_name}_mask_sample.mrc"
        with mrcfile.new(str(mask_path), overwrite=True) as mrc:
            mrc.set_data(mask.astype(np.float32)[np.newaxis, :, :])
            mrc.header.cella.x = IMAGE_SIZE * PIXEL_SIZE
            mrc.header.cella.y = IMAGE_SIZE * PIXEL_SIZE
            mrc.header.cella.z = PIXEL_SIZE
        
        results[config_name] = {
            "data": masked_data,
            "mask": mask,
            "path": output_path,
        }
    
    return results


def export_star(base_data):
    """导出 STAR 元数据"""
    print("\n" + "=" * 70)
    print("Exporting STAR metadata...")
    print("=" * 70)
    
    star_path = OUTPUT_DIR / "particles.star"
    
    lines = [
        "# RELION star file for cryoSPARC",
        "",
        "data_particles",
        "",
        "loop_"
    ]
    
    fields = [
        "_rlnImageName",
        "_rlnDefocusU",
        "_rlnDefocusV",
        "_rlnDefocusAngle",
        "_rlnVoltage",
        "_rlnSphericalAberration",
        "_rlnAmplitudeContrast",
        "_rlnMagnification",
        "_rlnDetectorPixelSize",
        "_rlnAngleRot",
        "_rlnAngleTilt",
        "_rlnAnglePsi",
    ]
    
    for i, f in enumerate(fields, 1):
        lines.append(f"{f} #{i}")
    
    angles = base_data["angles"]
    defocus = base_data["defocus"]
    defocus_v = base_data["defocus_v"]
    defocus_angle = base_data["defocus_angle"]
    
    for i in range(N_PROJECTIONS):
        vals = [
            f"{i+1:06d}@particles.mrcs",
            f"{defocus[i]:.2f}",
            f"{defocus_v[i]:.2f}",
            f"{defocus_angle[i]:.2f}",
            f"{VOLTAGE:.2f}",
            f"{CS:.4f}",
            f"{AC:.4f}",
            "10000.00",
            f"{PIXEL_SIZE * 1e-4:.6f}",
            f"{angles[i,0]:.6f}",
            f"{angles[i,1]:.6f}",
            f"{angles[i,2]:.6f}",
        ]
        lines.append('\t'.join(vals))
    
    with open(star_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"STAR: {star_path}")
    return star_path


def create_visualization(base_data, apodized_results):
    """创建可视化对比图"""
    print("\n" + "=" * 70)
    print("Creating visualization...")
    print("=" * 70)
    
    import matplotlib.pyplot as plt
    
    # 取第一帧
    sample_idx = 0
    noisy_sample = base_data["proj_noisy"][sample_idx]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Original noisy and Oracle mask
    axes[0, 0].imshow(noisy_sample, cmap='gray')
    axes[0, 0].set_title('Noisy (no mask)')
    axes[0, 0].axis('off')
    
    # Load oracle soft mask sample if exists
    oracle_mask_path = Path("/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/cryoem_aspire/cryosparc_10k_envelope/soft_masks.mrcs")
    if oracle_mask_path.exists():
        with mrcfile.open(str(oracle_mask_path), 'r') as mrc:
            oracle_mask = mrc.data[sample_idx]
        axes[0, 1].imshow(noisy_sample * oracle_mask, cmap='gray')
        axes[0, 1].set_title('Oracle Soft Masked')
        axes[0, 1].axis('off')
    
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')
    
    # Row 2: r=128 variants
    for idx, (name, title) in enumerate([("r128_zero", "r=128, outside=0"), 
                                          ("r128_noise", "r=128, outside=noise")]):
        data = apodized_results[name]["data"][sample_idx]
        mask = apodized_results[name]["mask"]
        
        axes[1, idx*2].imshow(data, cmap='gray')
        axes[1, idx*2].set_title(title)
        axes[1, idx*2].axis('off')
        
        axes[1, idx*2+1].imshow(mask, cmap='viridis', vmin=0, vmax=1)
        axes[1, idx*2+1].set_title(f"{title} - mask")
        axes[1, idx*2+1].axis('off')
    
    # Row 3: r=100 variants
    for idx, (name, title) in enumerate([("r100_zero", "r=100, outside=0"), 
                                          ("r100_noise", "r=100, outside=noise")]):
        data = apodized_results[name]["data"][sample_idx]
        mask = apodized_results[name]["mask"]
        
        axes[2, idx*2].imshow(data, cmap='gray')
        axes[2, idx*2].set_title(title)
        axes[2, idx*2].axis('off')
        
        axes[2, idx*2+1].imshow(mask, cmap='viridis', vmin=0, vmax=1)
        axes[2, idx*2+1].set_title(f"{title} - mask")
        axes[2, idx*2+1].axis('off')
    
    plt.suptitle(f'Apodized Circular Masks (edge width={APODIZATION_WIDTH}px)\n'
                 f'B-factor={B_FACTOR}Å², SNR={SNR}', fontsize=14)
    plt.tight_layout()
    
    vis_path = OUTPUT_DIR / "visualization.png"
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {vis_path}")


def create_readme():
    """创建 README 文档"""
    readme_path = OUTPUT_DIR / "README.txt"
    with open(readme_path, 'w') as f:
        f.write(f"""Apodized Circular Mask Data Sets
=================================

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Base script: generate_10k_with_envelope.py

Modifications:
1. Hard circular masks -> Apodized masks (cosine taper edge)
2. Two background fill modes: outside=0 or outside=noise

Parameters:
-----------
- Image size: {IMAGE_SIZE}x{IMAGE_SIZE}
- Pixel size: {PIXEL_SIZE} Å
- B-factor: {B_FACTOR} Å²
- SNR: {SNR}
- Defocus: 1.0-2.0 μm
- Apodization edge width: {APODIZATION_WIDTH} px

Generated Files:
----------------
1. r128_zero_masked.mrcs
   - Radius: 128 px
   - Outside filled with: 0
   
2. r128_noise_masked.mrcs
   - Radius: 128 px  
   - Outside filled with: Gaussian noise (same variance as signal noise)
   
3. r100_zero_masked.mrcs
   - Radius: 100 px
   - Outside filled with: 0
   
4. r100_noise_masked.mrcs
   - Radius: 100 px
   - Outside filled with: Gaussian noise (same variance as signal noise)

5. *_mask_sample.mrc
   - Sample masks for visualization

6. particles.star
   - Shared metadata (defocus, angles, etc.)

Apodization Function:
---------------------
For radius r from center:
- r <= R: mask = 1
- R < r < R + edge_width: mask = 0.5 * (1 + cos(π*(r-R)/edge_width))
- r >= R + edge_width: mask = 0

cryoSPARC Import:
-----------------
1. Import Particle Stack
2. Particle meta: particles.star
3. Particle data: Choose one of the 4 MRCS files above
4. Set pixel size: {PIXEL_SIZE} Å
""")
    print(f"README: {readme_path}")


def main():
    print("=" * 70)
    print("Generate Apodized Circular Masks")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Apodization edge width: {APODIZATION_WIDTH}px")
    print("=" * 70)
    
    # 1. 生成基础数据
    base_data = generate_base_data()
    
    # 2. 应用 apodized masks
    apodized_results = apply_apodized_masks(base_data)
    
    # 3. 导出 STAR
    export_star(base_data)
    
    # 4. 创建可视化
    create_visualization(base_data, apodized_results)
    
    # 5. 创建 README
    create_readme()
    
    print("\n" + "=" * 70)
    print("ALL GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.mrcs")):
        print(f"  - {f.name} ({f.stat().st_size/1e9:.2f} GB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
