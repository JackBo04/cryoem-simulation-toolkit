"""
Generate Base Projections (Clean and CTF-only)
==============================================
生成基础投影数据：
1. clean_projections.mrcs - 无CTF的干净投影
2. ctf_projections.mrcs - 有CTF但无噪声
3. noisy_projections.mrcs - 有CTF和噪声

用于参考和对比分析
"""

import numpy as np
import mrcfile
from pathlib import Path
from scipy.fft import fft2, ifft2, fftfreq
from aspire.volume import Volume
from aspire.source import Simulation
import time

OUTPUT_DIR = Path("/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/cryoem_aspire/cryosparc_10k_corrected")
MAP_FILE = "/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/emd_54951.map"

# 参数
N_PROJECTIONS = 10000
IMAGE_SIZE = 360
PIXEL_SIZE = 0.74
VOLTAGE = 300
CS = 2.7
AC = 0.1
SNR = 0.05
SEED = 42
B_FACTOR = 60


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


def main():
    start_time = time.time()
    
    print("=" * 70)
    print("Generate Base Projections (Clean, CTF, Noisy)")
    print("=" * 70)
    print(f"Number: {N_PROJECTIONS}")
    print(f"Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Pixel Size: {PIXEL_SIZE} Å")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)
    
    # 1. Load volume
    print("\n[1/6] Loading volume...")
    t0 = time.time()
    vol = Volume.load(MAP_FILE, dtype=np.float32)
    vol_down = vol.downsample(256)
    print(f"      Volume shape after downsample: {vol_down.shape}")
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
    
    # 3. Generate clean projections (256x256)
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
    
    # Pad to 360x360
    print("\n[4/6] Padding projections to 360x360...")
    t0 = time.time()
    pad_size = (IMAGE_SIZE - 256) // 2
    proj_clean = np.pad(proj_256, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 
                        mode='constant', constant_values=0)
    print(f"      Padded shape: {proj_clean.shape}")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # Save clean projections
    print("\n  Saving clean_projections.mrcs...")
    clean_path = OUTPUT_DIR / "clean_projections.mrcs"
    with mrcfile.new(str(clean_path), overwrite=True) as mrc:
        mrc.set_data(proj_clean.astype(np.float32))
        mrc.header.cella.x = IMAGE_SIZE * PIXEL_SIZE
        mrc.header.cella.y = IMAGE_SIZE * PIXEL_SIZE
        mrc.header.cella.z = N_PROJECTIONS * PIXEL_SIZE
    print(f"      Saved: {clean_path} ({clean_path.stat().st_size/1e9:.2f} GB)")
    
    # 5. Generate defocus values
    print("\n[5/6] Generating defocus values...")
    t0 = time.time()
    rng = np.random.default_rng(SEED + 1)
    defocus = rng.uniform(1.0e4, 2.0e4, N_PROJECTIONS)
    defocus_v = defocus * (0.95 + 0.05 * rng.random(N_PROJECTIONS))
    defocus_angle = rng.uniform(0, 180, N_PROJECTIONS)
    print(f"      Defocus range: {defocus.min()/1e4:.2f}-{defocus.max()/1e4:.2f} μm")
    print(f"      Time: {time.time()-t0:.2f}s")
    
    # 6. Apply CTF
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
    
    # Save CTF projections
    print("\n  Saving ctf_projections.mrcs...")
    ctf_path = OUTPUT_DIR / "ctf_projections.mrcs"
    with mrcfile.new(str(ctf_path), overwrite=True) as mrc:
        mrc.set_data(proj_ctf.astype(np.float32))
        mrc.header.cella.x = IMAGE_SIZE * PIXEL_SIZE
        mrc.header.cella.y = IMAGE_SIZE * PIXEL_SIZE
        mrc.header.cella.z = N_PROJECTIONS * PIXEL_SIZE
    print(f"      Saved: {ctf_path} ({ctf_path.stat().st_size/1e9:.2f} GB)")
    
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
    
    # Save noisy projections
    print("\n  Saving noisy_projections.mrcs...")
    noisy_path = OUTPUT_DIR / "noisy_projections.mrcs"
    with mrcfile.new(str(noisy_path), overwrite=True) as mrc:
        mrc.set_data(proj_noisy.astype(np.float32))
        mrc.header.cella.x = IMAGE_SIZE * PIXEL_SIZE
        mrc.header.cella.y = IMAGE_SIZE * PIXEL_SIZE
        mrc.header.cella.z = N_PROJECTIONS * PIXEL_SIZE
    print(f"      Saved: {noisy_path} ({noisy_path.stat().st_size/1e9:.2f} GB)")
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("BASE PROJECTIONS GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print("\nGenerated files:")
    print(f"  1. clean_projections.mrcs - No CTF, no noise")
    print(f"  2. ctf_projections.mrcs   - With CTF, no noise")
    print(f"  3. noisy_projections.mrcs - With CTF and noise")
    print("=" * 70)


if __name__ == "__main__":
    main()
