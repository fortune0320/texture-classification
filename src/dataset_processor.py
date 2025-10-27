import os
import cv2
import numpy as np
from PIL import Image
import random
import shutil


def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = np.copy(image)
    salt_mask = np.random.random(image.shape) < salt_prob
    noisy_image[salt_mask] = 255
    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy_image[pepper_mask] = 0
    return noisy_image


def process_texture_images():
    base_dir = "E:/Keep Learning part.2/semester 1/Machine Learning/Brodatz_Project"
    raw_data_dir = os.path.join(base_dir, "raw_data")
    processed_dir = os.path.join(base_dir, "processed_data")

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print("创建主目录: " + processed_dir)

    texture_classes = ["D1", "D2", "D3", "D4", "D5", "D6"]

    for class_name in texture_classes:
        print("\n处理类别: " + class_name)

        train_dir = os.path.join(processed_dir, class_name + "_train")
        test_dir = os.path.join(processed_dir, class_name + "_test")

        for dir_path in [train_dir, test_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print("创建目录: " + dir_path)

        class_dir = os.path.join(raw_data_dir, class_name)
        if os.path.exists(class_dir):
            image_files = [f for f in os.listdir(class_dir) if f.endswith(".tif")]
            print("找到 %d 张TIF图片" % len(image_files))

            if len(image_files) >= 16:
                random.shuffle(image_files)
                train_files = image_files[:10]
                test_files = image_files[10:16]

                train_count = 0
                for i, filename in enumerate(train_files):
                    image_path = os.path.join(class_dir, filename)
                    try:
                        image = Image.open(image_path)
                        if image.mode != "L":
                            image = image.convert("L")
                        image = np.array(image)

                        output_path = os.path.join(train_dir, class_name + "_%d.png" % (i+1))
                        cv2.imwrite(output_path, image)
                        train_count += 1

                    except Exception as e:
                        print("处理训练图像 %s 时出错: %s" % (filename, e))

                test_count = 0
                for i, filename in enumerate(test_files):
                    image_path = os.path.join(class_dir, filename)
                    try:
                        image = Image.open(image_path)
                        if image.mode != "L":
                            image = image.convert("L")
                        image = np.array(image)

                        noisy_image = add_salt_pepper_noise(
                            image, salt_prob=0.002, pepper_prob=0.002
                        )

                        output_path = os.path.join(test_dir, class_name + "_%d.png" % (i+1))
                        cv2.imwrite(output_path, noisy_image)
                        test_count += 1

                        print("  添加噪声: %s_%d.png" % (class_name, i + 1))

                    except Exception as e:
                        print("处理测试图像 %s 时出错: %s" % (filename, e))

                print("  成功处理: %d张训练 + %d张测试" % (train_count, test_count))

            else:
                print("警告: %s 类别的图像数量不足 (需要至少16张，当前%d张)" % (class_name, len(image_files)))
        else:
            print("警告: 找不到目录 %s" % class_dir)

    print("\n图像处理完成！已为测试集添加噪声。")


def verify_folder_structure():
    base_dir = os.environ.get("BASE_DIR", "data")
    processed_dir = os.path.join(base_dir, "processed_data")

    if os.path.exists(processed_dir):
        print("\n=== 文件夹结构验证 ===")
        print("主目录: " + processed_dir)

        texture_classes = ["D1", "D2", "D3", "D4", "D5", "D6"]

        for class_name in texture_classes:
            train_dir = os.path.join(processed_dir, class_name + "_train")
            test_dir = os.path.join(processed_dir, class_name + "_test")

            train_exists = os.path.exists(train_dir)
            test_exists = os.path.exists(test_dir)

            train_files = []
            test_files = []

            if train_exists:
                train_files = [f for f in os.listdir(train_dir) if f.endswith(".png")]

            if test_exists:
                test_files = [f for f in os.listdir(test_dir) if f.endswith(".png")]

            print("%s:" % class_name)
            print("  Train目录: %s (%d张PNG图片)" % ("存在" if train_exists else "缺失", len(train_files)))
            print("  Test目录:  %s (%d张PNG图片)" % ("存在" if test_exists else "缺失", len(test_files)))


def main():
    print("开始处理原始数据集")

    processed_dir = "E:/Keep Learning part.2/semester 1/Machine Learning/Brodatz_Project/processed_data"
    if os.path.exists(processed_dir):
        for item in os.listdir(processed_dir):
            item_path = os.path.join(processed_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print("已清空processed_data目录")

    process_texture_images()
    verify_folder_structure()
    print("\n处理完成！")


if __name__ == "__main__":
    main()
