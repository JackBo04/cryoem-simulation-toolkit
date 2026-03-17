"""
详细对比：有/无 Envelope 的 CTF 差异
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
lambda_e = 12.27 / np.sqrt(VOLTAGE * 1000 + 0.9784 * VOLTAGE**2)  # 更精确的公式
print(f"电子波长 λ = {lambda_e:.6f} Å")

# 频率坐标
freq = fftfreq(IMAGE_SIZE, d=PIXEL_SIZE)
kx, ky = np.meshgrid(freq, freq, indexing='ij')
k = np.sqrt(kx**2 + ky**2)

# 避免除零
k[0, 0] = 1e-10

# CTF 相位
chi = np.pi * lambda_e * k**2 * (CS * 1e7 * lambda_e**2 * k**2 - 2 * DEFOCUS)

# 完整CTF (含振幅对比度)
ctf_no_env = -np.sqrt(1 - AC**2) * np.sin(chi) - AC * np.cos(chi)

# 添加 Envelope (B-factor)
B_factor = 60  # Å²，典型值 50-100
envelope = np.exp(-B_factor * k**2 / 4)
ctf_with_env = ctf_no_env * envelope

print(f"\nB-factor = {B_factor} Å²")
print(f"Envelope at k=0.1: {np.exp(-B_factor * 0.1**2 / 4):.4f}")
print(f"Envelope at k=0.2: {np.exp(-B_factor * 0.2**2 / 4):.4f}")
print(f"Envelope at k=0.3: {np.exp(-B_factor * 0.3**2 / 4):.4f}")

# 创建对比图
fig = plt.figure(figsize=(16, 12))

# 1. CTF 2D 图对比
ax1 = plt.subplot(3, 3, 1)
im1 = ax1.imshow(ctf_no_env, cmap='RdBu_r', vmin=-1, vmax=1)
ax1.set_title('WITHOUT Envelope\n(你的当前实现)', fontsize=12, color='red')
ax1.axis('off')
plt.colorbar(im1, ax=ax1, fraction=0.046)

ax2 = plt.subplot(3, 3, 2)
im2 = ax2.imshow(ctf_with_env, cmap='RdBu_r', vmin=-1, vmax=1)
ax2.set_title('WITH Envelope (B=60Å²)\n(物理正确)', fontsize=12, color='green')
ax2.axis('off')
plt.colorbar(im2, ax=ax2, fraction=0.046)

ax3 = plt.subplot(3, 3, 3)
im3 = ax3.imshow(envelope, cmap='viridis', vmin=0, vmax=1)
ax3.set_title('Envelope Function\nE(k)=exp(-Bk²/4)', fontsize=12)
ax3.axis('off')
plt.colorbar(im3, ax=ax3, fraction=0.046)

# 2. 径向平均对比
from scipy.ndimage import map_coordinates

def radial_average(image, center=None):
    """计算图像的径向平均"""
    if center is None:
        center = np.array(image.shape) // 2
    y, x = np.indices(image.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    
    # 转换为空间频率 (1/Å)
    r_max = int(np.sqrt(2) * min(center)) + 1  # 最大可能的半径
    
    tbin = np.bincount(r.ravel(), image.ravel(), minlength=r_max)
    nr = np.bincount(r.ravel(), minlength=r_max)
    radialprofile = tbin / (nr + 1e-10)
    
    # 一维的频率值
    max_freq = 0.5 / PIXEL_SIZE  # 奈奎斯特频率
    k_values = np.arange(r_max) * (max_freq / r_max)
    return k_values[:r_max], radialprofile[:r_max]

# 重新排列CTF使得低频在中心
ctf_no_env_centered = np.fft.fftshift(ctf_no_env)
ctf_with_env_centered = np.fft.fftshift(ctf_with_env)

k_radial, ctf_no_env_radial = radial_average(ctf_no_env_centered)
_, ctf_with_env_radial = radial_average(ctf_with_env_centered)

ax4 = plt.subplot(3, 3, 4)
ax4.plot(k_radial, ctf_no_env_radial, 'r-', linewidth=1.5, label='Without Envelope')
ax4.set_xlabel('Spatial Frequency (1/Å)', fontsize=10)
ax4.set_ylabel('CTF Value', fontsize=10)
ax4.set_title('Radial Profile (No Envelope)', fontsize=11)
ax4.set_xlim([0, 0.5])
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)

ax5 = plt.subplot(3, 3, 5)
ax5.plot(k_radial, ctf_with_env_radial, 'g-', linewidth=1.5, label='With Envelope')
ax5.set_xlabel('Spatial Frequency (1/Å)', fontsize=10)
ax5.set_ylabel('CTF Value', fontsize=10)
ax5.set_title('Radial Profile (With Envelope)', fontsize=11)
ax5.set_xlim([0, 0.5])
ax5.grid(True, alpha=0.3)
ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)

ax6 = plt.subplot(3, 3, 6)
ax6.semilogy(k_radial, np.exp(-B_factor * k_radial**2 / 4), 'b-', linewidth=2)
ax6.set_xlabel('Spatial Frequency (1/Å)', fontsize=10)
ax6.set_ylabel('Envelope Value', fontsize=10)
ax6.set_title(f'Envelope Decay (B={B_factor}Å²)', fontsize=11)
ax6.set_xlim([0, 0.5])
ax6.grid(True, alpha=0.3)

# 3. 局部放大对比 - 低频区域
ax7 = plt.subplot(3, 3, 7)
center = IMAGE_SIZE // 2
zoom = 60  # 显示中心 120x120 区域
im7 = ax7.imshow(ctf_no_env_centered[center-zoom:center+zoom, center-zoom:center+zoom], 
                  cmap='RdBu_r', vmin=-1, vmax=1)
ax7.set_title('Zoom: Low Frequency\n(No Envelope)', fontsize=11)
ax7.axis('off')
plt.colorbar(im7, ax=ax7, fraction=0.046)

ax8 = plt.subplot(3, 3, 8)
im8 = ax8.imshow(ctf_with_env_centered[center-zoom:center+zoom, center-zoom:center+zoom], 
                  cmap='RdBu_r', vmin=-1, vmax=1)
ax8.set_title('Zoom: Low Frequency\n(With Envelope)', fontsize=11)
ax8.axis('off')
plt.colorbar(im8, ax=ax8, fraction=0.046)

# 高频区域
ax9 = plt.subplot(3, 3, 9)
# 显示角落的高频
im9 = ax9.imshow(ctf_no_env_centered[-40:, -40:], cmap='RdBu_r', vmin=-1, vmax=1)
ax9.set_title('Zoom: High Frequency\n(No Envelope = messy)', fontsize=11, color='red')
ax9.axis('off')
plt.colorbar(im9, ax=ax9, fraction=0.046)

plt.tight_layout()
plt.savefig('/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/ctf_detailed_comparison.png', dpi=200, bbox_inches='tight')
print("\n详细对比图已保存到: ctf_detailed_comparison.png")
plt.close()

# 第二张图：一维CTF曲线详细对比
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# 对比1: 完整范围
ax = axes[0, 0]
ax.plot(k_radial, ctf_no_env_radial, 'r-', linewidth=1.5, alpha=0.8, label='Without Envelope')
ax.plot(k_radial, ctf_with_env_radial, 'g-', linewidth=2, label='With Envelope (B=60Å²)')
ax.set_xlabel('Spatial Frequency (1/Å)', fontsize=11)
ax.set_ylabel('CTF Value', fontsize=11)
ax.set_title('CTF Comparison (Full Range)', fontsize=12)
ax.set_xlim([0, 0.5])
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# 对比2: 低频率区域放大
ax = axes[0, 1]
ax.plot(k_radial, ctf_no_env_radial, 'r-', linewidth=2, alpha=0.8, label='Without Envelope')
ax.plot(k_radial, ctf_with_env_radial, 'g-', linewidth=2.5, label='With Envelope (B=60Å²)')
ax.set_xlabel('Spatial Frequency (1/Å)', fontsize=11)
ax.set_ylabel('CTF Value', fontsize=11)
ax.set_title('CTF Comparison (Low Freq: 0-0.2 Å⁻¹)', fontsize=12)
ax.set_xlim([0, 0.2])
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# 对比3: 高频率区域放大
ax = axes[1, 0]
ax.plot(k_radial, ctf_no_env_radial, 'r-', linewidth=2, alpha=0.8, label='Without Envelope')
ax.plot(k_radial, ctf_with_env_radial, 'g-', linewidth=2.5, label='With Envelope (B=60Å²)')
ax.set_xlabel('Spatial Frequency (1/Å)', fontsize=11)
ax.set_ylabel('CTF Value', fontsize=11)
ax.set_title('CTF Comparison (High Freq: 0.2-0.5 Å⁻¹)', fontsize=12)
ax.set_xlim([0.2, 0.5])
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# 对比4: Envelope 单独显示
ax = axes[1, 1]
envelope_radial = np.exp(-B_factor * k_radial**2 / 4)
ax.fill_between(k_radial, envelope_radial, alpha=0.3, color='blue')
ax.plot(k_radial, envelope_radial, 'b-', linewidth=2, label=f'Envelope (B={B_factor}Å²)')
ax.set_xlabel('Spatial Frequency (1/Å)', fontsize=11)
ax.set_ylabel('Envelope Amplitude', fontsize=11)
ax.set_title('Envelope Function: E(k) = exp(-Bk²/4)', fontsize=12)
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 1.05])
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
# 标注关键点
for k_val in [0.1, 0.2, 0.3]:
    env_val = np.exp(-B_factor * k_val**2 / 4)
    ax.axvline(x=k_val, color='gray', linestyle='--', alpha=0.5)
    ax.annotate(f'k={k_val}\nE={env_val:.2f}', xy=(k_val, env_val), 
                xytext=(k_val+0.02, env_val+0.15 if env_val > 0.5 else env_val-0.15),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig('/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/ctf_1d_comparison.png', dpi=200, bbox_inches='tight')
print("一维CTF对比图已保存到: ctf_1d_comparison.png")
plt.close()

# 第三张图：不同B-factor的对比
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
B_values = [30, 60, 100, 150]
colors = ['purple', 'blue', 'green', 'orange']
for B, color in zip(B_values, colors):
    env = np.exp(-B * k_radial**2 / 4)
    ctf_B = ctf_no_env_radial * env
    ax1.plot(k_radial, ctf_B, color=color, linewidth=1.5, label=f'B={B}Å²')
ax1.set_xlabel('Spatial Frequency (1/Å)', fontsize=11)
ax1.set_ylabel('CTF Value', fontsize=11)
ax1.set_title('CTF with Different B-factors', fontsize=12)
ax1.set_xlim([0, 0.5])
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

ax2 = axes[1]
for B, color in zip(B_values, colors):
    env = np.exp(-B * k_radial**2 / 4)
    ax2.plot(k_radial, env, color=color, linewidth=2, label=f'B={B}Å²')
ax2.set_xlabel('Spatial Frequency (1/Å)', fontsize=11)
ax2.set_ylabel('Envelope Value', fontsize=11)
ax2.set_title('Envelope Functions', fontsize=12)
ax2.set_xlim([0, 0.5])
ax2.set_ylim([0, 1.05])
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/data/zouhuangbo/cryo-EM/nrips/mask_test/ctf_bfactor_comparison.png', dpi=200, bbox_inches='tight')
print("B-factor对比图已保存到: ctf_bfactor_comparison.png")
plt.close()

print("\n" + "="*60)
print("关键参数确认:")
print("="*60)
print(f"电压 (Voltage): {VOLTAGE} kV")
print(f"球差 (Cs): {CS} mm")
print(f"像素大小 (Pixel size): {PIXEL_SIZE} Å")
print(f"欠焦量 (Defocus): {DEFOCUS} Å = {DEFOCUS/10000:.2f} μm")
print(f"振幅对比度 (Amplitude Contrast): {AC}")
print(f"电子波长 (λ): {lambda_e:.4f} Å")
print(f"B-factor: {B_factor} Å² (典型范围: 30-100 Å²)")
print("="*60)
