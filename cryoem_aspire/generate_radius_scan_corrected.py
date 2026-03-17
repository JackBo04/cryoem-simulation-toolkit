"""
Generate Radius Scan with Apodized Masks - CORRECTED VERSION
============================================================
修复版本：
1. 增大图像尺寸到 360×360，给粒子留出余量
2. 使用正确的像素大小 0.74 Å (匹配 downsample 后的实际值)
3. 保持 apodization width = 15 px
4. 半径扫描：72, 80, 88, 96, 104, 112, 120, 128 px

输入：从 cryosparc_10k_envelope 重新生成基础数据
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

# 输出目录
OUTPUT_DIR = Path("/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/cryoem_aspire/cryosparc_10k_corrected")
OUTPUT_DIR.mkdir(exist_ok=True)

MAP_FILE = "/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/emd_54951.map"

# ========== 修正后的参数 ==========
N_PROJECTIONS = 10000
IMAGE_SIZE = 360          # 增大图像尺寸
PIXEL_SIZE = 0.74         # 正确的像素大小 (189.4 Å / 256 ≈ 0.74 Å)
VOLTAGE = 300
CS = 2.7
AC = 0.1
SNR = 0.05
SEED = 42
B_FACTOR = 60

# Apodization 参数
APODIZATION_WIDTH = 15

# 半径扫描列表
RADII = [72, 80, 88, 96, 104, 112, 120, 128]

# 模型原始参数 (用于正确 downsample)
ORIGINAL_PIXEL_SIZE = 0.592  # Å
ORIGINAL_SIZE = 320


def calculate_ctf_with_envelope(image_size, defocus, b_factor=B_FACTOR):
    """计算CTF，包含B-factor envelope"""
    lambda_ = 12.27 / np.sqrt(VOLTAGE * 1000 + 0.9784 * VOLTAGE**2)
    freq = fftfreq(image_size, d=PIXEL_SIZE)
    kx, ky = np.meshgrid(freq, freq, indexing='ij')
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1e-10
    
    chi = np.pi * lambda_ * k**2 * (CS * 1e7 * lambda_**2 * k**2 - 2 * defocus)
    ctf = -np.sqrt(1 - AC**2) * np.sin(chi) - AC * np.cos(chi)
    
    envelope = np.exp(-b_factor * k**2 / 4)
    ctf = ctf * envelope
    
    return ctf


def create_apodized_circular_mask(image_size, radius, edge_width=APODIZATION_WIDTH):
    """创建带余弦渐变边缘的圆形掩膜"""
    y, x = np.ogrid[-image_size//2:image_size//2, -image_size//2:image_size//2]
    r = np.sqrt(x**2 + y**2)
    
    mask = np.ones((image_size, image_size), dtype=np.float32)
    
    taper_region = (r > radius) & (r < radius + edge_width)
    mask[taper_region] = 0.5 * (1 + np.cos(np.pi * (r[taper_region] - radius) / edge_width))
    mask[r >= radius + edge_width] = 0.0
    
    return mask


def generate_base_data():
    """生成基础数据 (clean, noisy)"""
    start_time = time.time()
    
    print("=" * 70)
    print("Generate Base Data - CORRECTED VERSION")
    print("=" * 70)
    print(f"Number: {N_PROJECTIONS}")
    print(f"Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Pixel Size: {PIXEL_SIZE} Å (corrected)")
    print(f"Physical size: {IMAGE_SIZE * PIXEL_SIZE:.1f} Å")
    print(f"Defocus Range: 1.0-2.0 μm")
    print(f"SNR: {SNR}")
    print(f"CTF: Full CTF with B-factor envelope (B={B_FACTOR}Å²)")
    print("=" * 70)
    
    # 1. Load volume
    print("\n[1/6] Loading volume...")
    t0 = time.time()
    vol = Volume.load(MAP_FILE, dtype=np.float32)
    
    # 首先 downsample 到 256 (保持物理尺寸)
    vol_down = vol.downsample(256)
    print(f"      After downsample(256): {vol_down.shape}")
    
    # 然后将投影扩展到 360×360
    # 由于 Volume.downsample 不能直接到 360，我们先生成 256 的投影再 pad
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # 2. Generate angles
    print("\n[2/6] Generating angles...")
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    angles = np.zeros((N_PROJECTIONS, 3))
    angles[:, 0] = rng.uniform(0, 360, N_PROJECTIONS)
    angles[:, 1] = rng.uniform(0, 90, N_PROJECTIONS)
    angles[:, 2] = rng.uniform(0, 360, N_PROJECTIONS)
    angles_rad = np.deg2rad(angles)
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # 3. Generate clean projections at 256
    print("\n[3/6] Generating clean projections (256x256)...")
    t0 = time.time()
    src = Simulation(
        L=256,
        n=N_PROJECTIONS,
        vols=vol_down,
        angles=angles_rad,
        seed=SEED,
        dtype=np.float32,
    )
    proj_256 = src.projections[:].asnumpy()
    print(f"      Projections shape: {proj_256.shape}")
    print(f"      Value range: [{proj_256.min():.4f}, {proj_256.max():.4f}]")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # Pad to 360×360
    print("\n[4/6] Padding projections to 360x360...")
    t0 = time.time()
    pad_size = (IMAGE_SIZE - 256) // 2
    proj_clean = np.pad(proj_256, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 
                        mode='constant', constant_values=0)
    print(f"      Padded shape: {proj_clean.shape}")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # 5. Generate defocus values
    print("\n[5/6] Generating defocus values...")
    t0 = time.time()
    rng = np.random.default_rng(SEED + 1)
    defocus = rng.uniform(1.0e4, 2.0e4, N_PROJECTIONS)
    defocus_v = defocus * (0.95 + 0.05 * rng.random(N_PROJECTIONS))
    defocus_angle = rng.uniform(0, 180, N_PROJECTIONS)
    print(f"      Defocus range: {defocus.min()/1e4:.2f}-{defocus.max()/1e4:.2f} μm")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # 6. Apply CTF with envelope (at 360x360)
    print("\n[6/6] Applying CTF with B-factor envelope...")
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
    
    # 7. Add noise
    print("\n[7/6] Adding noise (SNR=0.05)...")
    t0 = time.time()
    rng = np.random.default_rng(SEED + 2)
    
    proj_noisy = np.zeros_like(proj_ctf)
    for i in range(N_PROJECTIONS):
        signal_power = np.mean(proj_ctf[i]**2)
        noise_var = signal_power / SNR
        noise = rng.normal(0, np.sqrt(noise_var), proj_ctf[i].shape)
        proj_noisy[i] = proj_ctf[i] + noise
        
        if (i + 1) % 2000 == 0:
            print(f"      Processed {i + 1}/{N_PROJECTIONS}")
    
    print(f"      Noisy range: [{proj_noisy.min():.4f}, {proj_noisy.max():.4f}]")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Base data generation complete! Time: {total_time/60:.1f} min")
    print("=" * 70)
    
    return {
        "proj_clean": proj_clean,
        "proj_ctf": proj_ctf,
        "proj_noisy": proj_noisy,
        "defocus": defocus,
        "defocus_v": defocus_v,
        "defocus_angle": defocus_angle,
        "angles": angles,
    }


def apply_apodized_masks(base_data):
    """应用 apodized masks，生成不同半径的数据"""
    proj_noisy = base_data["proj_noisy"]
    proj_ctf = base_data["proj_ctf"]
    
    print("\n" + "=" * 70)
    print("Generating Apodized Masks with Different Radii")
    print("=" * 70)
    print(f"Fixed apodization width: {APODIZATION_WIDTH}px")
    print(f"Radius list: {RADII}")
    print(f"Fill mode: noise")
    print("=" * 70)
    
    for radius in RADII:
        config_name = f"r{radius}_noise"
        
        print(f"\n{'='*70}")
        print(f"Generating: {config_name} (radius={radius}px)")
        print(f"{'='*70}")
        
        # 创建 apodized mask
        t0 = time.time()
        mask = create_apodized_circular_mask(IMAGE_SIZE, radius, APODIZATION_WIDTH)
        print(f"Mask created in {time.time()-t0:.2f}s")
        print(f"  Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
        print(f"  Effective mask radius: {radius} + {APODIZATION_WIDTH} = {radius + APODIZATION_WIDTH}px")
        
        # 应用 mask
        t0 = time.time()
        n = len(proj_noisy)
        masked_data = np.zeros_like(proj_noisy)
        rng = np.random.default_rng(42 + radius)
        
        for i in range(n):
            inside = proj_noisy[i] * mask
            
            signal_power = np.mean(proj_ctf[i]**2)
            noise_var = signal_power / SNR
            outside_noise = rng.normal(0, np.sqrt(noise_var), proj_noisy[i].shape)
            outside = outside_noise * (1 - mask)
            
            masked_data[i] = inside + outside
            
            if (i + 1) % 2000 == 0:
                print(f"  Processed {i + 1}/{n}")
        
        print(f"Data masked in {time.time()-t0:.2f}s")
        print(f"  Result range: [{masked_data.min():.4f}, {masked_data.max():.4f}]")
        
        # 保存 MRCS
        t0 = time.time()
        output_path = OUTPUT_DIR / f"{config_name}_masked.mrcs"
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            mrc.set_data(masked_data.astype(np.float32))
            mrc.header.cella.x = IMAGE_SIZE * PIXEL_SIZE
            mrc.header.cella.y = IMAGE_SIZE * PIXEL_SIZE
            mrc.header.cella.z = n * PIXEL_SIZE
        
        print(f"Saved: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1e9:.2f} GB")
        print(f"  Save time: {time.time()-t0:.2f}s")
        
        # 保存 mask 样本
        mask_path = OUTPUT_DIR / f"{config_name}_mask_sample.mrc"
        with mrcfile.new(str(mask_path), overwrite=True) as mrc:
            mrc.set_data(mask.astype(np.float32)[np.newaxis, :, :])
            mrc.header.cella.x = IMAGE_SIZE * PIXEL_SIZE
            mrc.header.cella.y = IMAGE_SIZE * PIXEL_SIZE
            mrc.header.cella.z = PIXEL_SIZE
        
        print(f"Mask sample: {mask_path}")


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


def create_visualization(base_data):
    """创建可视化"""
    print("\n" + "=" * 70)
    print("Creating visualization...")
    print("=" * 70)
    
    import matplotlib.pyplot as plt
    
    # 读取所有 mask samples
    masks = {}
    for radius in RADII:
        mask_path = OUTPUT_DIR / f"r{radius}_noise_mask_sample.mrc"
        with mrcfile.open(str(mask_path), 'r') as mrc:
            masks[radius] = mrc.data[0]
    
    # 创建可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, radius in enumerate(RADII):
        ax = axes[idx]
        im = ax.imshow(masks[radius], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'R = {radius}px\n(edge={APODIZATION_WIDTH}px)', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Apodized Circular Masks - Radius Scan (CORRECTED)\n'
                 f'Image: {IMAGE_SIZE}x{IMAGE_SIZE}px | Pixel: {PIXEL_SIZE}Å | Edge: {APODIZATION_WIDTH}px', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    vis_path = OUTPUT_DIR / "masks_visualization.png"
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Masks visualization saved: {vis_path}")
    
    # 创建 cross-section 对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    center = IMAGE_SIZE // 2
    x = np.arange(IMAGE_SIZE) - center
    
    for radius in RADII:
        mask = masks[radius]
        profile = mask[center, :]
        ax.plot(x, profile, label=f'R={radius}', linewidth=2)
    
    ax.set_xlabel('Distance from center (px)', fontsize=12)
    ax.set_ylabel('Mask value', fontsize=12)
    ax.set_title(f'Apodized Mask Cross-sections (edge_width={APODIZATION_WIDTH}px)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', title='Radius')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-200, 200)
    ax.set_ylim(-0.05, 1.05)
    
    for radius in RADII:
        ax.axvline(x=radius, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=radius + APODIZATION_WIDTH, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
    
    profile_path = OUTPUT_DIR / "masks_cross_section.png"
    plt.savefig(profile_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Cross-section saved: {profile_path}")
    
    # 创建样本粒子图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sample_idx = 0
    proj_noisy = base_data["proj_noisy"][sample_idx]
    
    axes[0].imshow(proj_noisy, cmap='gray')
    axes[0].set_title(f'Noisy Projection (Frame {sample_idx})\n{IMAGE_SIZE}x{IMAGE_SIZE}px')
    axes[0].axis('off')
    
    # 加圆标记 256 px 直径的粒子区域
    from matplotlib.patches import Circle
    circle_256 = Circle((center, center), 128, fill=False, color='red', linewidth=2, label='256px diameter')
    axes[0].add_patch(circle_256)
    axes[0].legend(loc='upper right')
    
    # 径向剖面
    axes[1].plot(proj_noisy[center, :], 'b-', linewidth=1.5)
    axes[1].axvline(center - 128, color='red', linestyle='--', alpha=0.7, label='Particle edge')
    axes[1].axvline(center + 128, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Position (px)')
    axes[1].set_ylabel('Intensity')
    axes[1].set_title('Horizontal Cross-section')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 直方图
    axes[2].hist(proj_noisy.ravel(), bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Pixel Value')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Value Distribution')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Sample Projection Analysis - CORRECTED\nPixel size: {PIXEL_SIZE}Å | Particle ~189Å (~256px)', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    sample_path = OUTPUT_DIR / "sample_projection.png"
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Sample projection saved: {sample_path}")


def create_readme():
    """创建 README 文档"""
    readme_path = OUTPUT_DIR / "README.txt"
    with open(readme_path, 'w') as f:
        f.write(f"""Apodized Circular Mask - Radius Scan (CORRECTED VERSION)
=========================================================

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Base script: generate_radius_scan_corrected.py

CORRECTIONS MADE:
-----------------
1. IMAGE_SIZE = {IMAGE_SIZE} (was 256) - 增大图像给粒子留出余量
2. PIXEL_SIZE = {PIXEL_SIZE} Å (was 1.06 Å) - 使用正确的像素大小
   - Original map: 320×320×320 @ 0.592 Å/pixel = 189.4 Å
   - After downsample(256): 256×256×256 @ 0.74 Å/pixel = 189.4 Å
   - Previous incorrect: 256×256 @ 1.06 Å/pixel = 271.4 Å (+43% error!)

Parameters:
-----------
- Image size: {IMAGE_SIZE}x{IMAGE_SIZE} px
- Pixel size: {PIXEL_SIZE} Å (CORRECTED)
- Physical size: {IMAGE_SIZE * PIXEL_SIZE:.1f} Å
- B-factor: {B_FACTOR} Å²
- SNR: {SNR}
- Defocus: 1.0-2.0 μm
- Apodization edge width: {APODIZATION_WIDTH} px (fixed)
- Background fill: Gaussian noise (same variance as signal noise)

Particle Size:
--------------
- Original model physical size: ~189 Å
- In current pixels: ~189/{PIXEL_SIZE} ≈ 256 px
- Particle occupies ~{256/IMAGE_SIZE*100:.0f}% of image (better!)

Radius Scan:
------------
{RADII}

Generated Files:
----------------
For each radius R:
  - r{{R}}_noise_masked.mrcs
    Apodized circular mask with radius R, outside filled with noise
  - r{{R}}_noise_mask_sample.mrc
    Sample mask for visualization

Metadata:
---------
- particles.star
  Corrected metadata with PIXEL_SIZE = {PIXEL_SIZE} Å

Apodization Function:
---------------------
For distance r from center:
- r <= R: mask = 1
- R < r < R + edge_width: mask = 0.5 * (1 + cos(π*(r-R)/edge_width))
- r >= R + edge_width: mask = 0

cryoSPARC Import:
-----------------
1. Import Particle Stack
2. Particle meta: particles.star
3. Particle data: Choose one of the r*_noise_masked.mrcs files
4. Set pixel size: {PIXEL_SIZE} Å (IMPORTANT!)

Visualization Files:
--------------------
- masks_visualization.png: All masks side by side
- masks_cross_section.png: Cross-section profiles of all masks
- sample_projection.png: Sample projection analysis
""")
    print(f"README: {readme_path}")


def main():
    start_time = time.time()
    
    print("=" * 70)
    print("Generate Radius Scan - CORRECTED VERSION")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Pixel size: {PIXEL_SIZE} Å (CORRECTED from 1.06 Å)")
    print(f"Physical size: {IMAGE_SIZE * PIXEL_SIZE:.1f} Å")
    print(f"Apodization edge width: {APODIZATION_WIDTH}px")
    print(f"Radius scan: {RADII}")
    print("=" * 70)
    
    # 1. 生成基础数据
    base_data = generate_base_data()
    
    # 2. 应用 apodized masks
    apply_apodized_masks(base_data)
    
    # 3. 导出 STAR
    export_star(base_data)
    
    # 4. 创建可视化
    create_visualization(base_data)
    
    # 5. 创建 README
    create_readme()
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("ALL GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.mrcs")):
        print(f"  - {f.name} ({f.stat().st_size/1e9:.2f} GB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
