"""
CTF Envelope Check - Simple Version
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftfreq

# Parameters
IMAGE_SIZE = 256
PIXEL_SIZE = 1.06
VOLTAGE = 300
CS = 2.7
DEFOCUS = 15000
AC = 0.07

# Calculate wavelength
lambda_e = 12.27 / np.sqrt(VOLTAGE * 1000 + 0.9784 * VOLTAGE**2)
print(f"Wavelength: {lambda_e:.6f} Å")

# Frequency grid
freq = fftfreq(IMAGE_SIZE, d=PIXEL_SIZE)
kx, ky = np.meshgrid(freq, freq, indexing='ij')
k = np.sqrt(kx**2 + ky**2)
k[0, 0] = 1e-10

# CTF calculation
chi = np.pi * lambda_e * k**2 * (CS * 1e7 * lambda_e**2 * k**2 - 2 * DEFOCUS)
ctf = -np.sqrt(1 - AC**2) * np.sin(chi) - AC * np.cos(chi)

# Create figure
fig = plt.figure(figsize=(16, 10))

# Row 1: 2D CTF plots
ax1 = fig.add_subplot(2, 4, 1)
im1 = ax1.imshow(np.fft.fftshift(ctf), cmap='RdBu_r', vmin=-1, vmax=1)
ax1.set_title('CTF (NO Envelope)', fontsize=12, color='red')
ax1.axis('off')
plt.colorbar(im1, ax=ax1, fraction=0.046)

B_values = [30, 60, 100]
colors = ['blue', 'green', 'purple']

for i, (B, color) in enumerate(zip(B_values, colors)):
    envelope = np.exp(-B * k**2 / 4)
    ctf_env = ctf * envelope
    
    ax = fig.add_subplot(2, 4, i+2)
    im = ax.imshow(np.fft.fftshift(ctf_env), cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f'CTF (B={B} Å²)', fontsize=12, color=color)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

# Row 2: 1D profiles
# fftfreq返回: [0, 1/(N*d), ..., -1/(N*d)]
# 需要正确排序: 先正频率，后负频率
# 使用fftshift重新排序
freq_shifted = np.fft.fftshift(freq)
center_idx = IMAGE_SIZE // 2
# 取从中心到正频率的部分
k_1d = freq_shifted[center_idx:]  # 0 到 0.5/PIXEL_SIZE

ctf_shifted = np.fft.fftshift(ctf)
ctf_1d = ctf_shifted[center_idx, center_idx:]  # 中心行，从中心到右边缘

ax = fig.add_subplot(2, 4, 5)
ax.plot(k_1d, ctf_1d, 'r-', linewidth=2)
ax.set_xlabel('Spatial Frequency (1/Å)')
ax.set_ylabel('CTF Value')
ax.set_title('1D Profile (No Envelope)', fontsize=11)
ax.set_xlim([0, 0.5])
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

for i, (B, color) in enumerate(zip(B_values, colors)):
    envelope = np.exp(-B * k**2 / 4)
    ctf_env = ctf * envelope
    ctf_env_1d = np.fft.fftshift(ctf_env)[center_idx, center_idx:]
    
    ax = fig.add_subplot(2, 4, i+6)
    ax.plot(k_1d, ctf_1d, 'r--', linewidth=1, alpha=0.5, label='No env')
    ax.plot(k_1d, ctf_env_1d, color=color, linewidth=2, label=f'B={B}')
    ax.set_xlabel('Spatial Frequency (1/Å)')
    ax.set_ylabel('CTF Value')
    ax.set_title(f'1D Profile (B={B})', fontsize=11)
    ax.set_xlim([0, 0.5])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/ctf_simple_comparison.png', dpi=200, bbox_inches='tight')
print("Saved: ctf_simple_comparison.png")
plt.close()

# Print parameters
print("\n" + "="*60)
print("PARAMETERS")
print("="*60)
print(f"Voltage:    {VOLTAGE} kV")
print(f"Cs:         {CS} mm")
print(f"Pixel:      {PIXEL_SIZE} Å")
print(f"Defocus:    {DEFOCUS} Å ({DEFOCUS/10000:.2f} μm)")
print(f"Amplitude:  {AC}")
print(f"Wavelength: {lambda_e:.4f} Å")
print("="*60)

print("\nEnvelope values:")
for k_test in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    print(f"  k={k_test:.2f} Å⁻¹:", end="")
    for B in [30, 60, 100]:
        env = np.exp(-B * k_test**2 / 4)
        print(f"  E(B={B})={env:.3f}", end="")
    print()
