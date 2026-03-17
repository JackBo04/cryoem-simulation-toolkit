"""
Generate EMD-8194 Dataset
==========================
基于 EMD-8194 模型生成完整数据集：
- 模型尺寸: 234×234×234 @ 0.637 Å = 149 Å
- 图像尺寸: 320×320 (pad from 256)
- 像素大小: 0.582 Å (149/256)
- 粒子半径: ~128 px (~74 Å)

包含:
1. ctf_projections.mrcs - CTF + B-factor envelope
2. noisy_projections.mrcs - CTF + 噪声
3. r90, r110, r130, r150 掩膜版本 (outside=noise)
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

OUTPUT_DIR = Path("/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/cryoem_aspire/emd_8194_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

MAP_FILE = "/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/cryoem_aspire/emd_8194.map.gz"

# ========== 参数设置 ==========
N_PROJECTIONS = 10000
IMAGE_SIZE = 520
PROJECTION_SIZE = 256  # ASPIRE downsample 尺寸
PIXEL_SIZE = 0.582     # Å (149/256)
VOLTAGE = 300
CS = 2.7
AC = 0.1
SNR = 0.05
SEED = 42
B_FACTOR = 60          # B-factor envelope

# 加窗参数
APODIZATION_WIDTH = 15

# 半径扫描 (基于粒子半径 128 px)
RADII = [90, 110, 130, 150]


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
    """生成基础数据"""
    start_time = time.time()
    
    print("=" * 70)
    print("Generate EMD-8194 Base Data")
    print("=" * 70)
    print(f"Model: EMD-8194 (234×234×234 @ 0.637Å = 149Å)")
    print(f"Output size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"Pixel size: {PIXEL_SIZE:.3f} Å")
    print(f"Particle radius: ~128 px (~74 Å)")
    print(f"Number: {N_PROJECTIONS}")
    print(f"SNR: {SNR}")
    print(f"CTF: Full CTF with B-factor envelope (B={B_FACTOR}Å²)")
    print("=" * 70)
    
    # 1. Load volume
    print("\n[1/5] Loading volume...")
    t0 = time.time()
    vol = Volume.load(MAP_FILE, dtype=np.float32)
    vol_down = vol.downsample(PROJECTION_SIZE)
    print(f"      Original: {vol.shape}")
    print(f"      Downsampled: {vol_down.shape}")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # 2. Generate angles
    print("\n[2/5] Generating angles...")
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    angles = np.zeros((N_PROJECTIONS, 3))
    angles[:, 0] = rng.uniform(0, 360, N_PROJECTIONS)
    angles[:, 1] = rng.uniform(0, 90, N_PROJECTIONS)
    angles[:, 2] = rng.uniform(0, 360, N_PROJECTIONS)
    angles_rad = np.deg2rad(angles)
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # 3. Generate clean projections
    print("\n[3/5] Generating clean projections...")
    t0 = time.time()
    src = Simulation(
        L=PROJECTION_SIZE,
        n=N_PROJECTIONS,
        vols=vol_down,
        angles=angles_rad,
        seed=SEED,
        dtype=np.float32,
    )
    proj_256 = src.projections[:].asnumpy()
    print(f"      Shape: {proj_256.shape}")
    print(f"      Range: [{proj_256.min():.4f}, {proj_256.max():.4f}]")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # Pad to target size
    print(f"\n[4/5] Padding to {IMAGE_SIZE}×{IMAGE_SIZE}...")
    t0 = time.time()
    pad_size = (IMAGE_SIZE - PROJECTION_SIZE) // 2
    proj_clean = np.pad(proj_256, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 
                        mode='constant', constant_values=0)
    print(f"      Padded shape: {proj_clean.shape}")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # 4. Generate defocus
    print("\n[5/5] Generating defocus values...")
    t0 = time.time()
    rng = np.random.default_rng(SEED + 1)
    defocus = rng.uniform(1.0e4, 2.0e4, N_PROJECTIONS)
    defocus_v = defocus * (0.95 + 0.05 * rng.random(N_PROJECTIONS))
    defocus_angle = rng.uniform(0, 180, N_PROJECTIONS)
    print(f"      Defocus: {defocus.min()/1e4:.2f}-{defocus.max()/1e4:.2f} μm")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # 6. Apply CTF
    print("\n[6/5] Applying CTF with B-factor envelope...")
    t0 = time.time()
    proj_ctf = np.zeros_like(proj_clean)
    for i in range(N_PROJECTIONS):
        ctf = calculate_ctf_with_envelope(IMAGE_SIZE, defocus[i])
        proj_fft = fft2(proj_clean[i])
        proj_ctf[i] = np.real(ifft2(proj_fft * ctf))
        if (i + 1) % 2000 == 0:
            print(f"      Processed {i + 1}/{N_PROJECTIONS}")
    print(f"      Range: [{proj_ctf.min():.4f}, {proj_ctf.max():.4f}]")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # 7. Add noise
    print("\n[7/5] Adding noise (SNR=0.05)...")
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
    print(f"      Range: [{proj_noisy.min():.4f}, {proj_noisy.max():.4f}]")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # Save base projections (only ctf and noisy)
    print("\n[8/5] Saving base projections...")
    for name, data in [("ctf", proj_ctf), ("noisy", proj_noisy)]:
        path = OUTPUT_DIR / f"{name}_projections.mrcs"
        with mrcfile.new(str(path), overwrite=True) as mrc:
            mrc.set_data(data.astype(np.float32))
            mrc.header.cella.x = IMAGE_SIZE * PIXEL_SIZE
            mrc.header.cella.y = IMAGE_SIZE * PIXEL_SIZE
            mrc.header.cella.z = N_PROJECTIONS * PIXEL_SIZE
        print(f"      {name}: {path.name} ({path.stat().st_size/1e9:.2f} GB)")
    
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
    """应用 apodized masks"""
    proj_noisy = base_data["proj_noisy"]
    proj_ctf = base_data["proj_ctf"]
    
    print("\n" + "=" * 70)
    print("Generating Apodized Masks")
    print("=" * 70)
    print(f"Radii: {RADII}")
    print(f"Apodization width: {APODIZATION_WIDTH}px")
    
    for radius in RADII:
        config_name = f"r{radius}_noise"
        
        print(f"\n{'='*70}")
        print(f"Generating: {config_name}")
        print(f"{'='*70}")
        
        # Create mask
        t0 = time.time()
        mask = create_apodized_circular_mask(IMAGE_SIZE, radius, APODIZATION_WIDTH)
        print(f"Mask created in {time.time()-t0:.2f}s")
        print(f"  Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
        
        effective_radius = radius + APODIZATION_WIDTH
        boundary = IMAGE_SIZE // 2
        if effective_radius > boundary:
            print(f"  ⚠️  Warning: Effective radius ({effective_radius}px) exceeds boundary ({boundary}px)")
        else:
            print(f"  Effective radius: {effective_radius}px")
        
        # Apply mask
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
        
        # Save
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
        
        # Save mask sample
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
    
    # Read all mask samples
    masks = {}
    for radius in RADII:
        mask_path = OUTPUT_DIR / f"r{radius}_noise_mask_sample.mrc"
        with mrcfile.open(str(mask_path), 'r') as mrc:
            masks[radius] = mrc.data[0]
    
    # Sample projection
    sample_idx = 0
    proj_clean = base_data["proj_clean"][sample_idx]
    proj_ctf = base_data["proj_ctf"][sample_idx]
    proj_noisy = base_data["proj_noisy"][sample_idx]
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Projections
    ax1 = plt.subplot(3, 4, 1)
    im1 = ax1.imshow(proj_ctf, cmap='gray', vmin=-0.05, vmax=0.05)
    ax1.set_title('CTF (B-factor=60Å²)', fontsize=11, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = plt.subplot(3, 4, 2)
    im2 = ax2.imshow(proj_noisy, cmap='gray', vmin=-0.1, vmax=0.1)
    ax2.set_title(f'Noisy (SNR={SNR})', fontsize=11, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # CTF effect
    ax3 = plt.subplot(3, 4, 3)
    ctf_effect = proj_ctf - proj_clean
    im3 = ax3.imshow(ctf_effect, cmap='RdBu_r', vmin=-0.05, vmax=0.05)
    ax3.set_title('CTF Effect', fontsize=11, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Empty subplot
    ax4 = plt.subplot(3, 4, 4)
    ax4.axis('off')
    
    # Row 2: Masks
    for idx, radius in enumerate(RADII):
        ax = plt.subplot(3, 4, 5 + idx)
        im = ax.imshow(masks[radius], cmap='viridis', vmin=0, vmax=1)
        
        effective_r = radius + APODIZATION_WIDTH
        particle_r = 128
        coverage = (effective_r * 2) / (particle_r * 2) * 100
        
        ax.set_title(f'R={radius}px ({coverage:.0f}% coverage)', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 3: Cross-sections and analysis
    # Cross-section
    ax = plt.subplot(3, 4, 9)
    center = IMAGE_SIZE // 2
    x = np.arange(IMAGE_SIZE) - center
    for radius in RADII:
        profile = masks[radius][center, :]
        ax.plot(x, profile, label=f'R={radius}', linewidth=2)
    ax.axvline(-128, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(128, color='gray', linestyle='--', alpha=0.5, label='Particle edge')
    ax.set_xlabel('Distance from center (px)')
    ax.set_ylabel('Mask value')
    ax.set_title('Mask Cross-sections')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-200, 200)
    
    # Projection histogram
    ax = plt.subplot(3, 4, 10)
    ax.hist(proj_ctf.ravel(), bins=50, alpha=0.5, label='CTF', color='green', density=True)
    ax.hist(proj_noisy.ravel(), bins=50, alpha=0.5, label='Noisy', color='red', density=True)
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Density')
    ax.set_title('Value Distribution')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Radial profile of projection
    ax = plt.subplot(3, 4, 11)
    y, x = np.ogrid[-center:center, -center:center]
    r = np.sqrt(x**2 + y**2)
    r_bins = np.arange(0, 150, 5)
    
    for name, proj, color in [('CTF', proj_ctf, 'green'), 
                               ('Noisy', proj_noisy, 'red')]:
        profile = []
        for r_val in r_bins[:-1]:
            mask_ring = (r >= r_val) & (r < r_val + 5)
            if mask_ring.sum() > 0:
                profile.append(np.abs(proj)[mask_ring].mean())
            else:
                profile.append(0)
        ax.plot(r_bins[:-1], profile, label=name, color=color, linewidth=2)
    
    ax.axvline(128, color='gray', linestyle='--', alpha=0.5, label='Particle radius')
    ax.set_xlabel('Radius (px)')
    ax.set_ylabel('Mean |Intensity|')
    ax.set_title('Radial Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Empty subplot
    ax = plt.subplot(3, 4, 12)
    ax.axis('off')
    
    plt.suptitle(f'EMD-8194 Dataset Overview\n'
                 f'Image: {IMAGE_SIZE}×{IMAGE_SIZE}px | Pixel: {PIXEL_SIZE:.3f}Å | '
                 f'Particle: ~149Å (~256px) | B-factor: {B_FACTOR}Å²', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    vis_path = OUTPUT_DIR / "dataset_overview.png"
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {vis_path}")


def create_readme():
    """创建 README"""
    readme_path = OUTPUT_DIR / "README.txt"
    with open(readme_path, 'w') as f:
        f.write(f"""EMD-8194 Dataset
=================

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Model: EMD-8194

Model Information:
------------------
- Original: 234×234×234 voxels @ 0.637 Å/pixel
- Physical size: 149.0 Å
- After downsample(256): 256×256×256 @ 0.582 Å/pixel

Parameters:
-----------
- Image size: {IMAGE_SIZE}×{IMAGE_SIZE} px
- Pixel size: {PIXEL_SIZE:.3f} Å
- Number of projections: {N_PROJECTIONS}
- Defocus range: 1.0-2.0 μm
- SNR: {SNR}
- B-factor envelope: {B_FACTOR} Å²
- Voltage: {VOLTAGE} kV
- Cs: {CS} mm
- Amplitude contrast: {AC}

Particle Size:
--------------
- Particle diameter in image: ~256 px (~149 Å)
- Particle radius: ~128 px (~74 Å)
- Image padding: {(IMAGE_SIZE - 256) // 2}px on each side

Radius Scan (4 values):
-----------------------
{RADII}

Coverage relative to particle (256px diameter):
- R=90:  210px effective = 82% coverage
- R=110: 250px effective = 98% coverage (close to full)
- R=130: 290px effective = 113% coverage (includes delocalization)
- R=150: 330px effective = 129% coverage

Generated Files:
----------------
Base projections:
  - ctf_projections.mrcs    (CTF with B-factor envelope)
  - noisy_projections.mrcs  (CTF + noise)

Masked projections (for each radius R):
  - r{{R}}_noise_masked.mrcs
  - r{{R}}_noise_mask_sample.mrc

Metadata:
  - particles.star

Apodization:
------------
- Edge width: {APODIZATION_WIDTH}px
- Function: 0.5 * (1 + cos(π*(r-R)/edge_width))
- Outside fill: Gaussian noise (same variance as signal)

cryoSPARC Import:
-----------------
1. Import Particle Stack
2. Particle meta: particles.star
3. Particle data: Choose appropriate .mrcs file
4. Set pixel size: {PIXEL_SIZE:.3f} Å

Visualization:
--------------
- dataset_overview.png
""")
    print(f"README: {readme_path}")


def main():
    start_time = time.time()
    
    print("=" * 70)
    print("EMD-8194 Dataset Generation")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)
    
    # 1. Generate base data
    base_data = generate_base_data()
    
    # 2. Apply masks
    apply_apodized_masks(base_data)
    
    # 3. Export STAR
    export_star(base_data)
    
    # 4. Create visualization
    create_visualization(base_data)
    
    # 5. Create README
    create_readme()
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("ALL GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.mrcs")):
        print(f"  - {f.name} ({f.stat().st_size/1e9:.2f} GB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
