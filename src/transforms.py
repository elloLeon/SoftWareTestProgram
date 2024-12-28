import cv2
import numpy as np

def add_noise(images, noise_level=0.1, dtype=np.float32):
    """
    为输入图像添加随机噪声。

    参数:
        images (numpy.ndarray): 输入图像数组，形状为 (N, C, H, W)。
        noise_level (float): 噪声强度，默认值为 0.1。
        dtype (numpy.dtype): 输出数据的类型，默认值为 np.float32。

    返回:
        numpy.ndarray: 添加噪声后的图像数组。
    """
    noisy_images = []
    for img in images:
        noise = np.random.normal(0, noise_level, img.shape)
        noisy_image = np.clip(img + noise, 0, 1)  # 确保像素值在 [0, 1] 范围内
        noisy_images.append(noisy_image)
    return np.array(noisy_images, dtype=dtype)
def add_blur(images, kernel_size=5):
    blurred_images = [cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) for img in images]
    return np.array(blurred_images)
