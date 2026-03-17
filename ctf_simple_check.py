"""
简化版CTF对比检查 - 确认参数正确性
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftfreq, fft2, ifft2

# 参数
IMAGE_SIZE = 256
PIXEL_SIZE = 1.06  # Å
VOLTAGE = 300  # kV
CS = 2.7  # mm
DEFOCUS = 15000  # Å (1.5 μm)
AC = 0.07  # 振幅对比度

# 计算电子波长 (Å)
lambda_e = 12.27 / np.sqrt(VOLTAGE * 1000 + 0.9784 * VOLTAGE**2)
print(f"电子波长 λ = {lambda_e:.6f} Å")

# 频率坐标
freq = fftfreq(IMAGE_SIZE, d=PIXEL_SIZE)
kx, ky = np.meshgrid(freq, freq, indexing='ij')
k = np.sqrt(kx**2 + ky**2)
k[0, 0] = 1e-10

# CTF 相位
chi = np.pi * lambda_e * k**2 * (CS * 1e7 * lambda_e**2 * k**2 - 2 * DEFOCUS)

# 完整CTF (含振幅对比度)
ctf_no_env = -np.sqrt(1 - AC**2) * np.sin(chi) - AC * np.cos(chi)

# 不同B-factor
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 原始CTF (无envelope)
ax = axes[0, 0]
im = ax.imshow(np.fft.fftshift(ctf_no_env), cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_title('NO Envelope\n(Current Implementation)', fontsize=11)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

# 1D radial profile (简单版本)
center = IMAGE_SIZE // 2
# 取中心十字截面
ctf_1d_noenv = np.fft.fftshift(ctf_no_env)[center, center:]
k_1d = freq[center:]  # 从中心到边缘的频率

ax = axes[1, 0]
ax.plot(k_1d, ctf_1d_noenv, 'r-', linewidth=1.5)
ax.set_xlabel('Spatial Frequency (1/Å)')
ax.set_ylabel('CTF Value')
ax.set_title('1D Profile (No Envelope)')
ax.set_xlim([0, 0.5])
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# 不同B-factor对比
B_factors = [30, 60, 100]
colors = ['blue', 'green', 'purple']

for idx, (B, color) in enumerate(zip(B_factors, colors)):
    envelope = np.exp(-B * k**2 / 4)
    ctf_env = ctf_no_env * envelope
    
    # 2D plot
    ax = axes[0, idx+1] if idx < 2 else axes[0, 2]
    if idx < 2:
        im = ax.imshow(np.fft.fftshift(ctf_env), cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'Envelope B={B} Å²', fontsize=11, color=color)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        # 显示envelope本身
        im = ax.imshow(np.fft.fftshift(envelope), cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Envelope Function\n(B={B} Å² shown)', fontsize=11, color=color)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 1D plot
    ax = axes[1, idx+1]
    ctf_1d_env = np.fft.fftshift(ctf_env)[center, center:]
    ax.plot(k_1d, ctf_1d_noenv, 'r--', linewidth=1, alpha=0.5, label='No env')
    ax.plot(k_1d, ctf_1d_env, color=color, linewidth=2, label=f'B={B}')
    ax.set_xlabel('Spatial Frequency (1/Å)')
    ax.set_ylabel('CTF Value')
    ax.set_title(f'1D Profile (B={B})')
    ax.set_xlim([0, 0.5])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/ctf_simple_comparison.png', dpi=200, bbox_inches='tight')
print("简单对比图已保存到: ctf_simple_comparison.png")
plt.close()

# 打印关键参数
print("\n" + "="*60)
print("KEY PARAMETERS VERIFICATION")
print("="*60)
print(f"Voltage:              {VOLTAGE} kV")
print(f"Cs (Spherical Aberration): {CS} mm")
print(f"Pixel Size:           {PIXEL_SIZE} Å")
print(f"Defocus:              {DEFOCUS} Å = {DEFOCUS/10000:.2f} μm")
print(f"Amplitude Contrast:   {AC}")
print(f"Electron Wavelength:  {lambda_e:.4f} Å")
print("="*60)
print(f"\nCTF Formula Used:")
print(f"  chi = πλk²(Cs·λ²·k² - 2Δ)")
print(f"  CTF = -sqrt(1-AC²)·sin(chi) - AC·cos(chi)")
print(f"\nEnvelope Formula:")
print(f"  E(k) = exp(-B·k²/4)")
print(f"\nRecommended B-factor range: 30-100 Å²")
print("="*60)

# 验证计算
print(f"\nVerification at specific frequencies:")
for k_test in [0.05, 0.1, 0.2, 0.3]:
    chi_test = np.pi * lambda_e * k_test**2 * (CS * 1e7 * lambda_e**2 * k_test**2 - 2 * DEFOCUS)
    ctf_test = -np.sqrt(1 - AC**2) * np.sin(chi_test) - AC * np.cos(chi_test)
    env_30 = np.exp(-30 * k_test**2 / 4)
    env_60 = np.exp(-60 * k_test**2 / 4)
    env_100 = np.exp(-100 * k_test**2 / 4)
    print(f"  k={k_test:.2f} Å⁻¹: CTF={ctf_test:+.3f}, Env(B=30)={env_30:.3f}, Env(B=60)={env_60:.3f}, Env(B=100)={env_100:.3f}")
