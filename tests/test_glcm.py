import numpy as np
from src.glcm_compatible import manual_glcm, calculate_contrast, calculate_energy


def test_manual_glcm():
    # 创建简单测试图像（2x2图像，0度方向有2对像素：(0,1)和(2,3)）
    image = np.array([[0, 1], [2, 3]], dtype=np.uint8)

    # 测试0度方向（水平方向，每行相邻像素对）
    glcm_0 = manual_glcm(image, distance=1, angle=0)

    # 修复：考虑归一化，总像素对数量为2，所以每个非零值应为 1/2 = 0.5
    assert np.isclose(glcm_0[0, 1], 0.5)  # (0,1)对的归一化值
    assert np.isclose(glcm_0[2, 3], 0.5)  # (2,3)对的归一化值
    assert np.sum(glcm_0) == 1.0  # 验证归一化（总和为1）

    # 测试90度方向（垂直方向，每列相邻像素对）
    glcm_90 = manual_glcm(image, distance=1, angle=90)
    assert np.isclose(glcm_90[0, 2], 0.5)  # (0,2)对的归一化值
    assert np.isclose(glcm_90[1, 3], 0.5)  # (1,3)对的归一化值
    assert np.sum(glcm_90) == 1.0  # 验证归一化


def test_glcm_metrics():
    # 创建测试用的GLCM矩阵
    glcm = np.zeros((4, 4), dtype=np.float64)
    glcm[0, 0] = 0.5
    glcm[0, 1] = 0.25
    glcm[1, 0] = 0.25

    # 测试能量
    energy = calculate_energy(glcm)
    assert np.isclose(energy, 0.5**2 + 0.25**2 + 0.25**2)

    # 测试对比度
    contrast = calculate_contrast(glcm)
    expected = 0.5 * (0 - 0) ** 2 + 0.25 * (0 - 1) ** 2 + 0.25 * (1 - 0) ** 2
    assert np.isclose(contrast, expected)
