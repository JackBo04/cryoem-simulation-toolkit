"""
Cryo-EM Particle Simulation using Simple Rotation (scipy-based)
=============================================================
Simplified implementation - works reliably
"""

import numpy as np
import mrcfile
from pathlib import Path
from scipy.ndimage import rotate, shift
from scipy.fft import fft2, ifft2, fftfreq
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter, binary_dilation
import time


class Config:
    DATA_DIR = Path("/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/data")
    OUTPUT_DIR = Path(
        "/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/cryoem_fourier/output"
    )
    MRC_FILE = "apoferritin.mrc"
    N_PROJECTIONS = 10000
    IMAGE_SIZE = 256
    ROT_RANGE = (0, 360)
    TILT_RANGE = (0, 90)
    PSI_RANGE = (0, 360)
    VOLTAGE = 300
    CS = 2.7
    AC = 0.1
    DEFOCUS_MIN = 1.0e4
    DEFOCUS_MAX = 3.0e4
    SNR = 0.05
    MASK_SIGMA = 4.0
    SEED = 42


def create_test_volume(size=256):
    """Create test volume (two spheres)."""
    print(f"[Info] Creating test volume {size}x{size}x{size}...")
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    z = np.linspace(-1, 1, size)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    r1 = np.sqrt((X - 0.3) ** 2 + Y**2 + Z**2)
    r2 = np.sqrt((X + 0.3) ** 2 + Y**2 + Z**2)

    volume = np.zeros((size, size, size), dtype=np.float32)
    volume[r1 < 0.4] = 1.0
    volume[r2 < 0.4] = 1.0
    volume = gaussian_filter(volume, sigma=2.0)

    # Normalize
    volume = (volume - volume.mean()) / (volume.std() + 1e-10)

    return volume.astype(np.float32)


def sample_euler_angles(n, seed=42):
    """Sample Euler angles uniformly on hemisphere."""
    rng = np.random.default_rng(seed)
    angles = np.zeros((n, 3))
    angles[:, 0] = rng.uniform(0, 360, n)  # Rot
    angles[:, 1] = rng.uniform(0, 90, n)  # Tilt
    angles[:, 2] = rng.uniform(0, 360, n)  # Psi
    return angles


def project_volume_simple(volume, rot, tilt, psi):
    """
    Project 3D volume to 2D using scipy rotation.
    Implements ZYZ Euler rotation convention.
    """
    # Apply rotations in ZYZ order
    # First: rotate around Z (in-plane, psi)
    v = rotate(volume, angle=psi, axes=(0, 1), order=1, mode="constant", cval=0)

    # Second: rotate around Y (tilt)
    v = rotate(volume, angle=tilt, axes=(0, 2), order=1, mode="constant", cval=0)

    # Third: rotate around Z (in-plane, rot)
    v = rotate(volume, angle=rot, axes=(0, 1), order=1, mode="constant", cval=0)

    # Sum along Z axis to get projection
    projection = v.sum(axis=0)

    return projection


def project_volume_fast(volume, angles):
    """
    Generate projections for all angles.
    Optimized batch processing.
    """
    n = len(angles)
    size = volume.shape[0]
    projections = np.zeros((n, size, size), dtype=np.float32)

    print(f"[Task 2] Generating {n} projections...")
    start = time.time()

    for i, (rot, tilt, psi) in enumerate(angles):
        # Apply rotations
        # Rotate around Y axis (tilt) - this determines the view
        v = rotate(
            volume,
            angle=tilt,
            axes=(0, 2),
            order=1,
            mode="constant",
            cval=0,
            reshape=False,
        )

        # Rotate around X axis for variety
        v = rotate(
            v,
            angle=rot * 0.3,
            axes=(1, 2),
            order=1,
            mode="constant",
            cval=0,
            reshape=False,
        )

        # Project by summing
        proj = v.sum(axis=0)

        # Normalize
        proj = (proj - proj.mean()) / (proj.std() + 1e-10)

        projections[i] = proj

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n} ({100 * (i + 1) / n:.1f}%)")

    elapsed = time.time() - start
    print(f"[Task 2] Done in {elapsed:.1f}s ({n / elapsed:.1f} proj/s)")

    return projections


def generate_soft_masks(projections, sigma=4.0):
    """Generate soft masks from projections."""
    print(f"[Task 3] Generating soft masks (sigma={sigma})...")
    n, h, w = projections.shape
    masks = np.zeros((n, h, w), dtype=np.float32)

    for i in range(n):
        proj = projections[i]

        try:
            thresh = threshold_otsu(proj)
        except:
            thresh = proj.mean() + proj.std()

        binary = (proj > thresh).astype(np.float32)
        binary = binary_dilation(binary, iterations=1)
        soft_mask = gaussian_filter(binary, sigma=sigma)
        soft_mask = np.clip(soft_mask, 0, 1)
        masks[i] = soft_mask

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{n}")

    return masks


def compute_ctf(size, defocus, voltage=300, cs=2.7, pixel=1.0):
    """Compute simple CTF."""
    # Electron wavelength
    lambda_ = 12.27 / np.sqrt(voltage * 1000)

    # Frequency grid
    freq = fftfreq(size, d=pixel)
    kx, ky = np.meshgrid(freq, freq, indexing="ij")
    k = np.sqrt(kx**2 + ky**2)

    chi = np.pi * lambda_ * k**2 * (cs * 1e7 * lambda_**2 * k**2 - 2 * defocus)
    ctf = -np.sin(chi)

    return ctf


def add_ctf_and_noise(projections, defocus_values, snr=0.05, seed=42):
    """Add CTF modulation and noise."""
    print(f"[Task 4] Adding CTF + noise (SNR={snr})...")
    n, h, w = projections.shape
    noisy = np.zeros((n, h, w), dtype=np.float32)
    rng = np.random.default_rng(seed)

    for i in range(n):
        proj = projections[i]

        # Apply CTF
        ctf = compute_ctf(h, defocus_values[i])
        proj_fft = fft2(proj)
        proj_ctf = np.real(ifft2(proj_fft * ctf))

        # Add noise
        signal_power = np.mean(proj_ctf**2)
        noise_var = signal_power / snr
        noise = rng.normal(0, np.sqrt(noise_var), proj_ctf.shape)
        noisy[i] = proj_ctf + noise

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{n}")

    return noisy


def export_relion(noisy, masked, angles, defocus_values, output_dir, prefix="test"):
    """Export to RELION format."""
    print(f"[Task 5] Exporting to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save MRC stacks
    noisy_path = output_dir / f"noisy_{prefix}.mrcs"
    masked_path = output_dir / f"masked_{prefix}.mrcs"

    with mrcfile.new(str(noisy_path), overwrite=True) as mrc:
        mrc.set_data(noisy.astype(np.float32))

    with mrcfile.new(str(masked_path), overwrite=True) as mrc:
        mrc.set_data(masked.astype(np.float32))

    print(f"  Saved: {noisy_path.name}, {masked_path.name}")

    # Generate STAR file
    star_path = output_dir / "ground_truth.star"
    lines = [
        "data_",
        "",
        "loop_",
        "_rlnMicrographName #1",
        "_rlnCoordinateX #2",
        "_rlnCoordinateY #3",
        "_rlnAngleRot #4",
        "_rlnAngleTilt #5",
        "_rlnAnglePsi #6",
        "_rlnDefocusU #7",
        "_rlnDefocusV #8",
        "_rlnDefocusAngle #9",
        "_rlnVoltage #10",
        "_rlnSphericalAberration #11",
        "_rlnAmplitudeContrast #12",
    ]

    center = 128.0
    for i in range(len(angles)):
        r, t, p = angles[i]
        d = defocus_values[i]
        lines.append(
            f"{noisy_path.name} {center:.1f} {center:.1f} "
            f"{r:.4f} {t:.4f} {p:.4f} {d:.2f} {d * 0.9:.2f} 0.00 300.000 2.700 0.100"
        )

    with open(star_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Saved: {star_path.name}")
    return noisy_path, masked_path, star_path


def run_pipeline(config=None):
    """Run full pipeline."""
    if config is None:
        config = Config()

    print("=" * 60)
    print("CRYO-EM SIMULATION - SIMPLE METHOD")
    print("=" * 60)

    # Task 1: Load/create volume
    volume = create_test_volume(config.IMAGE_SIZE)

    # Task 2: Generate projections
    angles = sample_euler_angles(config.N_PROJECTIONS, config.SEED)
    projections = project_volume_fast(volume, angles)
    print(
        f"  Shape: {projections.shape}, Range: [{projections.min():.3f}, {projections.max():.3f}]"
    )

    # Task 3: Generate masks
    masks = generate_soft_masks(projections, config.MASK_SIGMA)

    # Task 4: CTF + noise
    rng = np.random.default_rng(config.SEED + 1)
    defocus = rng.uniform(config.DEFOCUS_MIN, config.DEFOCUS_MAX, config.N_PROJECTIONS)
    noisy = add_ctf_and_noise(projections, defocus, config.SNR, config.SEED + 2)
    masked = noisy * masks

    # Task 5: Export
    export_relion(noisy, masked, angles, defocus, config.OUTPUT_DIR, "test")

    print("\n" + "=" * 60)
    print("DONE! Output:", config.OUTPUT_DIR)
    print("=" * 60)

    return {
        "projections": projections,
        "noisy": noisy,
        "masks": masks,
        "angles": angles,
        "defocus": defocus,
    }


if __name__ == "__main__":
    run_pipeline()
