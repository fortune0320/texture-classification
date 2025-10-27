import numpy as np
from src.dataset_processor import add_salt_pepper_noise


def test_salt_pepper_noise():
    # 创建测试图像（全灰）
    image = np.ones((100, 100), dtype=np.uint8) * 128

    # 添加噪声
    noisy_image = add_salt_pepper_noise(image, 0.1, 0.1)

    # 验证噪声添加
    assert np.sum(noisy_image == 255) > 0  # 盐噪声
    assert np.sum(noisy_image == 0) > 0  # 椒噪声
    assert np.sum(noisy_image == 128) > 0  # 原始像素保留

    # 验证噪声比例大致符合预期
    total_pixels = 100 * 100
    salt_ratio = np.sum(noisy_image == 255) / total_pixels
    pepper_ratio = np.sum(noisy_image == 0) / total_pixels

    assert 0.08 <= salt_ratio <= 0.12  # 允许一定误差范围
    assert 0.08 <= pepper_ratio <= 0.12
