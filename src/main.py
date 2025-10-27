"""主程序入口"""

import sys
import os

# 将项目根目录（/app）添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from src.dataset_processor import main as process_data
from src.glcm_compatible import main as extract_features_and_train
from src.analysis import main as run_analysis


def main():
    print("=== 开始纹理分类流程 ===")

    # 1. 处理数据集
    print("\n1. 处理原始图像数据...")
    process_data()

    # 2. 提取特征并训练模型
    print("\n2. 提取GLCM特征并训练模型...")
    extract_features_and_train()

    # 3. 运行特征重要性分析
    print("\n3. 分析特征重要性...")
    run_analysis()

    print("\n=== 所有流程完成 ===")


if __name__ == "__main__":
    main()
