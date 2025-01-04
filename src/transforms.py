import cv2
import numpy as np


def add_noise(images, noise_level=0.1, dtype=np.float32):
    """
    为输入图像添加随机噪声。
    """
    noisy_images = []
    for img in images:
        noise = np.random.normal(0, noise_level, img.shape)
        noisy_image = np.clip(img + noise, 0, 1)  # 确保像素值在 [0, 1] 范围内
        noisy_images.append(noisy_image)
    return np.array(noisy_images, dtype=dtype)


def add_blur(images, kernel_size=5, method="gaussian"):
    """
    为输入图像添加模糊效果。

    参数:
        images (numpy.ndarray): 输入图像数组。
        kernel_size (int): 模糊核大小，默认为 5。
        method (str): 模糊方法，支持 "gaussian" 和 "average"。

    返回:
        numpy.ndarray: 添加模糊效果后的图像数组。
    """
    blurred_images = []
    for img in images:
        if method == "gaussian":
            blurred_image = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif method == "average":
            blurred_image = cv2.blur(img, (kernel_size, kernel_size))
        else:
            raise ValueError("Unsupported blur method. Use 'gaussian' or 'average'.")
        blurred_images.append(blurred_image)
    return np.array(blurred_images)


def add_flip(images, flip_code=1):
    """
    为输入图像添加翻转（水平、垂直或同时）。

    参数:
        images (numpy.ndarray): 输入图像数组。
        flip_code (int): 翻转类型，1 表示水平翻转，0 表示垂直翻转，-1 表示同时翻转。

    返回:
        numpy.ndarray: 翻转后的图像数组。
    """
    flipped_images = [cv2.flip(img, flip_code) for img in images]
    return np.array(flipped_images)


def add_rotation(images, angle=15):
    """
    为输入图像添加旋转变换。

    参数:
        images (numpy.ndarray): 输入图像数组。
        angle (int): 旋转角度，默认为 15 度。

    返回:
        numpy.ndarray: 旋转后的图像数组。
    """
    rotated_images = []
    for img in images:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(img, matrix, (w, h))
        rotated_images.append(rotated_image)
    return np.array(rotated_images)
