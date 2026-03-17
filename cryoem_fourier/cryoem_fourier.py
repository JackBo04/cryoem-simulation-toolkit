"""
Cryo-EM Particle Simulation using Fourier Slice Theorem (PyTorch)
================================================================
Task 1-5: Full pipeline - GPU accelerated projection generation
"""

import os
import numpy as np
import mrcfile
from pathlib import Path

# PyTorch imports
import torch
import torch.nn.functional as F
from torch.fft import fftshift, ifftshift, fftn, ifftn, rfftn, irfftn

# For mask generation
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.filters import threshold_otsu


# ============================================================================
# Configuration
# ============================================================================


class Config:
    # Paths
    DATA_DIR = Path("/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/data")
    OUTPUT_DIR = Path(
        "/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/cryoem_fourier/output"
    )

    # Input
    MRC_FILE = "apoferritin.mrc"

    # Projection parameters
    N_PROJECTIONS = 10000
    IMAGE_SIZE = 256

    # Euler angles (hemisphere sampling)
    ROT_RANGE = (0, 360)
    TILT_RANGE = (0, 90)
    PSI_RANGE = (0, 360)

    # CTF parameters
    VOLTAGE = 300  # kV
    CS = 2.7  # mm
    AC = 0.1  # amplitude contrast
    DEFOCUS_MIN = 1.0e4  # Angstroms
    DEFOCUS_MAX = 3.0e4

    # Noise
    SNR = 0.05

    # Mask parameters
    MASK_SIGMA = 4.0

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Seed
    SEED = 42


# ============================================================================
# Helper Functions: Euler Angles to Rotation Matrix
# ============================================================================


def euler_to_rotation_matrix(
    rot: torch.Tensor, tilt: torch.Tensor, psi: torch.Tensor
) -> torch.Tensor:
    """
    Convert Euler angles to rotation matrix (ZYZ convention for cryo-EM).

    Parameters:
    -----------
    rot, tilt, psi : torch.Tensor
        Euler angles in radians, shape (N,) or scalar

    Returns:
    --------
    R : torch.Tensor
        Rotation matrices, shape (N, 3, 3) or (3, 3)
    """
    # Handle batch dimension
    is_batch = rot.ndim > 0

    if not is_batch:
        rot, tilt, psi = rot.unsqueeze(0), tilt.unsqueeze(0), psi.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # ZYZ rotation convention
    cos_r, sin_r = torch.cos(rot), torch.sin(rot)
    cos_t, sin_t = torch.cos(tilt), torch.sin(tilt)
    cos_p, sin_p = torch.cos(psi), torch.sin(psi)

    # Build rotation matrix
    R = torch.zeros((len(rot), 3, 3), dtype=rot.dtype, device=rot.device)

    R[:, 0, 0] = cos_r * cos_t * cos_p - sin_r * sin_p
    R[:, 0, 1] = -cos_r * cos_t * sin_p - sin_r * cos_p
    R[:, 0, 2] = cos_r * sin_t

    R[:, 1, 0] = sin_r * cos_t * cos_p + cos_r * sin_p
    R[:, 1, 1] = -sin_r * cos_t * sin_p + cos_r * cos_p
    R[:, 1, 2] = sin_r * sin_t

    R[:, 2, 0] = -sin_t * cos_p
    R[:, 2, 1] = sin_t * sin_p
    R[:, 2, 2] = cos_t

    if squeeze:
        R = R.squeeze(0)

    return R


def create_zyz_rotation(rot: float, tilt: float, psi: float) -> torch.Tensor:
    """Create ZYZ rotation matrix from angles in radians."""
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    cos_t, sin_t = np.cos(tilt), np.sin(tilt)
    cos_p, sin_p = np.cos(psi), np.sin(psi)

    R = torch.tensor(
        [
            [
                cos_r * cos_t * cos_p - sin_r * sin_p,
                -cos_r * cos_t * sin_p - sin_r * cos_p,
                cos_r * sin_t,
            ],
            [
                sin_r * cos_t * cos_p + cos_r * sin_p,
                -sin_r * cos_t * sin_p + cos_r * cos_p,
                sin_r * sin_t,
            ],
            [-sin_t * cos_p, sin_t * sin_p, cos_t],
        ],
        dtype=torch.float32,
    )

    return R


# ============================================================================
# Task 1: Load and Normalize MRC Volume
# ============================================================================


def load_volume(mrc_path: str, device: str = "cpu") -> torch.Tensor:
    """
    Load MRC file and normalize to PyTorch tensor.

    Parameters:
    -----------
    mrc_path : str
        Path to MRC file
    device : str
        'cpu' or 'cuda'

    Returns:
    --------
    volume : torch.Tensor
        3D volume tensor, shape (D, H, W)
    """
    print(f"[Task 1] Loading volume from {mrc_path}")

    with mrcfile.open(mrc_path, mode="r") as mrc:
        volume = mrc.data.copy()

    # Convert to tensor and normalize
    volume = torch.from_numpy(volume).float()

    # Normalize: zero mean, unit variance
    volume = (volume - volume.mean()) / (volume.std() + 1e-10)

    # Move to device
    volume = volume.to(device)

    print(f"[Task 1] Volume shape: {volume.shape}, device: {volume.device}")
    print(f"[Task 1] Volume mean: {volume.mean():.4f}, std: {volume.std():.4f}")

    return volume


def create_test_volume(size: int = 256, device: str = "cpu") -> torch.Tensor:
    """Create a simple test volume (two spheres)."""
    print(f"[Info] Creating test volume...")

    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    z = torch.linspace(-1, 1, size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

    # Two spheres
    r1 = torch.sqrt((X - 0.3) ** 2 + Y**2 + Z**2)
    r2 = torch.sqrt((X + 0.3) ** 2 + Y**2 + Z**2)

    volume = torch.zeros((size, size, size), dtype=torch.float32)
    volume[r1 < 0.4] = 1.0
    volume[r2 < 0.4] = 1.0

    # Smooth with Gaussian
    volume = gaussian_filter(volume.numpy(), sigma=2.0)
    volume = torch.from_numpy(volume).float()

    volume = volume.to(device)

    print(f"[Info] Test volume created: {volume.shape}")

    return volume


# ============================================================================
# Task 2: Fourier Slice Projection (Core Algorithm)
# ============================================================================


class FourierSliceProjector:
    """
    GPU-accelerated projection using Fourier Slice Theorem.

    Theorem: 2D projection = central slice of 3D Fourier transform
    """

    def __init__(self, volume: torch.Tensor, pad_factor: float = 1.5):
        """
        Initialize projector.

        Parameters:
        -----------
        volume : torch.Tensor
            3D volume tensor, shape (N, N, N)
        pad_factor : float
            Padding factor for FFT (reduces edge artifacts)
        """
        self.device = volume.device
        self.size = volume.shape[0]
        self.pad_factor = pad_factor

        # Pre-compute FFT of padded volume
        self.volume_padded = self._pad_volume(volume)
        self.volume_fft = self._compute_fft(self.volume_padded)

        print(f"[Projector] Initialized on {self.device}, size: {self.size}")

    def _pad_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """Pad volume to reduce FFT artifacts."""
        if self.pad_factor <= 1.0:
            return volume

        pad_size = int(self.size * (self.pad_factor - 1) / 2)
        padded = F.pad(volume, [pad_size] * 6, mode="constant", value=0)

        return padded

    def _compute_fft(self, volume: torch.Tensor) -> torch.Tensor:
        """Compute centered 3D FFT."""
        fft_vol = fftn(volume)
        fft_vol = fftshift(fft_vol, dim=(-3, -2, -1))

        # Zero mean to avoid low-frequency artifact
        shape = fft_vol.shape
        mid = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
        fft_vol[mid[0], mid[1], mid[2]] = 0

        return fft_vol

    def project(
        self, rot: torch.Tensor, tilt: torch.Tensor, psi: torch.Tensor
    ) -> torch.Tensor:
        """
        Project volume at given Euler angles.

        Parameters:
        -----------
        rot, tilt, psi : torch.Tensor
            Euler angles in radians

        Returns:
        --------
        projection : torch.Tensor
            2D projection image
        """
        # Create rotation matrix
        R = euler_to_rotation_matrix(rot, tilt, psi)

        # Extract central slice
        slice_fft = self._extract_slice_fft(R)

        # Inverse FFT to get projection
        projection = self._ifft_slice(slice_fft)

        return projection

    def project_batch(
        self, angles: torch.Tensor, batch_size: int = 100
    ) -> torch.Tensor:
        """
        Project volume at multiple angles.

        Parameters:
        -----------
        angles : torch.Tensor
            Euler angles, shape (N, 3) in radians
        batch_size : int
            Batch size for processing

        Returns:
        --------
        projections : torch.Tensor
            2D projections, shape (N, H, W)
        """
        n = len(angles)
        size = int(self.size * self.pad_factor)

        projections = torch.zeros(
            (n, size, size), dtype=torch.float32, device=self.device
        )

        n_batches = (n + batch_size - 1) // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n)

            rot = angles[start:end, 0]
            tilt = angles[start:end, 1]
            psi = angles[start:end, 2]

            batch_projections = self.project(rot, tilt, psi)
            projections[start:end] = batch_projections

            if (i + 1) % 10 == 0:
                print(f"  Processed batch {i + 1}/{n_batches}")

        return projections

    def _extract_slice_fft(self, R: torch.Tensor) -> torch.Tensor:
        """
        Extract central FFT slice using rotation matrix.
        Simplified implementation using real FFT.
        """
        # Padded volume size
        padded_size = self.volume_padded.shape[0]

        # Create frequency grid for 2D output (using rfft format)
        freq = torch.fft.rfftfreq(padded_size, d=1.0, device=self.device)

        # Create 2D frequency grid
        ky, kx = torch.meshgrid(freq, freq, indexing="ij")
        kz = torch.zeros_like(kx)

        # Stack to (H, W, 3) coordinates in zyx order
        coords_2d = torch.stack([kz, ky, kx], dim=-1)  # (H, W, 3)

        # Handle batch dimension
        if R.ndim == 3:
            # Batch of rotations
            coords_2d_flat = coords_2d.reshape(-1, 3)  # (H*W, 3)
            coords_rotated = torch.einsum(
                "ij,njk->nik", coords_2d_flat, R
            )  # (N, H*W, 3)
            coords_rotated = coords_rotated.reshape(
                R.shape[0], ky.shape[0], ky.shape[1], 3
            )
        else:
            # Single rotation
            coords_2d_flat = coords_2d.reshape(-1, 3)
            coords_rotated = coords_2d_flat @ R.T  # (H*W, 3)
            coords_rotated = coords_rotated.reshape(ky.shape[0], ky.shape[1], 3)

        # Get z-coordinate (perpendicular to slice)
        kz_rotated = coords_rotated[..., 2]  # (H, W) or (N, H, W)

        # Create interpolation grid
        if R.ndim == 3:
            # Batch mode - take first slice for now
            kz_slice = kz_rotated[0]
        else:
            kz_slice = kz_rotated

        # Convert to indices for sampling z-plane
        z_indices = (kz_slice * padded_size + padded_size // 2).long()
        z_indices = torch.clamp(z_indices, 0, padded_size - 1)

        # Sample from 3D FFT volume
        # Use simple indexing for z-plane
        slice_fft = self.volume_fft[z_indices, :, : padded_size // 2 + 1]

        return slice_fft

    def _ifft_slice(self, slice_fft: torch.Tensor) -> torch.Tensor:
        """Inverse FFT of 2D slice to get projection."""
        # Shift and compute inverse FFT
        slice_fft = ifftshift(slice_fft, dim=(-2, -1))
        projection = irfftn(slice_fft, dim=(-2, -1))
        projection = torch.real(projection)

        return projection


def generate_projections_pytorch(
    volume: torch.Tensor,
    angles: torch.Tensor,
    batch_size: int = 100,
    pad_factor: float = 1.5,
) -> np.ndarray:
    """
    Generate projections using Fourier Slice Theorem.

    Parameters:
    -----------
    volume : torch.Tensor
        3D volume
    angles : torch.Tensor
        Euler angles in radians, shape (N, 3)
    batch_size : int
        Processing batch size
    pad_factor : float
        FFT padding factor

    Returns:
    --------
    projections : np.ndarray
        2D projections, shape (N, H, W)
    """
    print(f"[Task 2] Generating {len(angles)} projections with Fourier Slice...")

    # Create projector
    projector = FourierSliceProjector(volume, pad_factor=pad_factor)

    # Generate projections
    projections = projector.project_batch(angles, batch_size=batch_size)

    # Unpad if needed
    if pad_factor > 1.0:
        pad_size = int(volume.shape[0] * (pad_factor - 1) / 2)
        projections = projections[:, pad_size:-pad_size, pad_size:-pad_size]

    # Convert to numpy
    projections = projections.cpu().numpy()

    print(f"[Task 2] Projections shape: {projections.shape}")
    print(f"[Task 2] Range: [{projections.min():.4f}, {projections.max():.4f}]")

    return projections


def sample_euler_angles(
    n: int,
    rot_range=(0, 360),
    tilt_range=(0, 90),
    psi_range=(0, 360),
    seed: int = 42,
    device: str = "cpu",
) -> np.ndarray:
    """
    Sample Euler angles uniformly on hemisphere.

    Returns angles in both numpy (for export) and torch (for processing).
    """
    rng = np.random.default_rng(seed)

    angles = np.zeros((n, 3))
    angles[:, 0] = rng.uniform(*rot_range, n)  # Rot
    angles[:, 1] = rng.uniform(*tilt_range, n)  # Tilt
    angles[:, 2] = rng.uniform(*psi_range, n)  # Psi

    # Convert to radians for torch
    angles_rad = torch.from_numpy(np.deg2rad(angles)).float()
    angles_rad = angles_rad.to(device)

    return angles, angles_rad


# ============================================================================
# Task 3: Generate Soft Masks (CPU-based)
# ============================================================================


def generate_soft_masks(projections: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Generate soft masks from projections."""
    print(f"[Task 3] Generating soft masks with sigma={sigma}...")

    n, h, w = projections.shape
    masks = np.zeros((n, h, w), dtype=np.float32)

    for i in range(n):
        proj = projections[i]

        # Otsu threshold
        try:
            thresh = threshold_otsu(proj)
        except:
            thresh = proj.mean() + proj.std()

        binary = (proj > thresh).astype(np.float32)

        # Morphological dilation
        binary = binary_dilation(binary, iterations=1)

        # Gaussian blur
        soft_mask = gaussian_filter(binary, sigma=sigma)
        soft_mask = np.clip(soft_mask, 0, 1)

        masks[i] = soft_mask

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{n}")

    print(f"[Task 3] Masks shape: {masks.shape}")

    return masks


# ============================================================================
# Task 4: CTF and Noise
# ============================================================================


def compute_ctf(
    size: int,
    defocus: float,
    voltage: float = 300,
    cs: float = 2.7,
    pixel_size: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute CTF (Contrast Transfer Function) in Fourier space.

    Parameters:
    -----------
    size : int
        Image size
    defocus : float
        Defocus in Angstroms
    voltage : float
        Acceleration voltage in kV
    cs : float
        Spherical aberration in mm
    pixel_size : float
        Pixel size in Angstroms

    Returns:
    --------
    ctf : torch.Tensor
        CTF array, shape (size, size)
    """
    # Electron wavelength (Angstroms) at given voltage
    # λ = 12.27 / sqrt(V + 0.978e-6*V^2)
    V = voltage * 1000  # Convert to volts
    lambda_ = 12.27 / np.sqrt(V + 0.978e-6 * V**2)

    # Spatial frequency
    freq = torch.fft.fftfreq(size, d=pixel_size, device=device)
    kx, ky = torch.meshgrid(freq, freq, indexing="ij")
    k = torch.sqrt(kx**2 + ky**2)

    # CTF formula
    # χ(k) = π * λ * k^2 * (Cs * λ^2 * k^2 - 2 * defocus)
    chi = np.pi * lambda_ * k**2 * (cs * 1e7 * lambda_**2 * k**2 - 2 * defocus)

    # CTF = -sin(χ) with envelope
    ctf = -torch.sin(chi)

    # Gaussian envelope (optional, for realistic simulation)
    # B = 100  # decay
    # ctf *= torch.exp(-B * k**2)

    return ctf


def add_ctf_and_noise(
    projections: np.ndarray,
    defocus_values: np.ndarray,
    snr: float = 0.05,
    device: str = "cpu",
    seed: int = 42,
) -> np.ndarray:
    """Add CTF modulation and Gaussian noise."""
    print(f"[Task 4] Adding CTF and noise (SNR={snr})...")

    n, h, w = projections.shape
    noisy = np.zeros((n, h, w), dtype=np.float32)

    rng = np.random.default_rng(seed)

    for i in range(n):
        proj = projections[i]

        # Apply CTF
        ctf = compute_ctf(h, defocus_values[i], device=device)

        # FFT, apply CTF, IFFT
        proj_fft = fftn(torch.from_numpy(proj).to(device))
        proj_ctf = irfftn(proj_fft * ctf)
        proj_ctf = torch.real(proj_ctf).cpu().numpy()

        # Add noise based on SNR
        signal_power = np.mean(proj_ctf**2)
        noise_var = signal_power / snr

        noise = rng.normal(0, np.sqrt(noise_var), proj_ctf.shape)
        noisy[i] = proj_ctf + noise

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{n}")

    print(f"[Task 4] Noisy images shape: {noisy.shape}")

    return noisy


# ============================================================================
# Task 5: Export to RELION Format
# ============================================================================


def export_relion_format(
    noisy_images: np.ndarray,
    masked_images: np.ndarray,
    angles: np.ndarray,
    defocus_values: np.ndarray,
    output_dir: Path,
    prefix: str = "particles",
):
    """Export data in RELION format."""
    print(f"[Task 5] Exporting to RELION format...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save noisy particles
    noisy_path = output_dir / f"noisy_{prefix}.mrcs"
    with mrcfile.new(str(noisy_path), overwrite=True) as mrc:
        mrc.set_data(noisy_images.astype(np.float32))
    print(f"  Saved: {noisy_path}")

    # Save masked particles
    masked_path = output_dir / f"masked_{prefix}.mrcs"
    with mrcfile.new(str(masked_path), overwrite=True) as mrc:
        mrc.set_data(masked_images.astype(np.float32))
    print(f"  Saved: {masked_path}")

    # Generate STAR file
    star_path = output_dir / "ground_truth.star"
    generate_star_file(angles, defocus_values, noisy_path.name, star_path)
    print(f"  Saved: {star_path}")

    return noisy_path, masked_path, star_path


def generate_star_file(
    angles: np.ndarray,
    defocus_values: np.ndarray,
    particles_filename: str,
    output_path: Path,
):
    """Generate RELION .star file."""
    n = len(angles)

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
    for i in range(n):
        rot, tilt, psi = angles[i]
        def_u = defocus_values[i]
        def_v = def_u * 0.9

        line = f"{particles_filename} {center:.1f} {center:.1f} "
        line += f"{rot:.4f} {tilt:.4f} {psi:.4f} "
        line += f"{def_u:.2f} {def_v:.2f} 0.00 "
        line += "300.000 2.700 0.100"

        lines.append(line)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Generated STAR with {n} particles")


# ============================================================================
# Main Pipeline
# ============================================================================


def run_pipeline(config: Config = None):
    """Run the full cryo-EM simulation pipeline."""
    if config is None:
        config = Config()

    device = config.DEVICE
    print(f"[Info] Using device: {device}")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mrc_path = config.DATA_DIR / config.MRC_FILE

    # === Task 1: Load Volume ===
    if mrc_path.exists():
        volume = load_volume(str(mrc_path), device=device)
    else:
        print(f"[Warning] MRC not found, creating test volume")
        volume = create_test_volume(config.IMAGE_SIZE, device=device)

    # === Task 2: Generate Projections ===
    angles_deg, angles_rad = sample_euler_angles(
        config.N_PROJECTIONS,
        rot_range=config.ROT_RANGE,
        tilt_range=config.TILT_RANGE,
        psi_range=config.PSI_RANGE,
        seed=config.SEED,
        device=device,
    )

    clean_projections = generate_projections_pytorch(
        volume, angles_rad, batch_size=200, pad_factor=1.5
    )

    # === Task 3: Generate Masks ===
    soft_masks = generate_soft_masks(clean_projections, sigma=config.MASK_SIGMA)

    # === Task 4: CTF + Noise ===
    rng = np.random.default_rng(config.SEED + 1)
    defocus_values = rng.uniform(
        config.DEFOCUS_MIN, config.DEFOCUS_MAX, config.N_PROJECTIONS
    )

    noisy_projections = add_ctf_and_noise(
        clean_projections,
        defocus_values,
        snr=config.SNR,
        device=device,
        seed=config.SEED + 2,
    )

    # Masked noisy images
    masked_noisy = noisy_projections * soft_masks

    # === Task 5: Export ===
    export_relion_format(
        noisy_projections,
        masked_noisy,
        angles_deg,
        defocus_values,
        config.OUTPUT_DIR,
        prefix="test",
    )

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print(f"Output: {config.OUTPUT_DIR}")
    print("=" * 60)

    return {
        "clean": clean_projections,
        "noisy": noisy_projections,
        "masks": soft_masks,
        "angles": angles_deg,
        "defocus": defocus_values,
    }


if __name__ == "__main__":
    run_pipeline()
