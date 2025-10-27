import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# 定义 main 函数（关键：让 main.py 可以导入）
def main():
    # 读取csv文件（使用容器内绝对路径）
    df = pd.read_csv("/app/glcm_features_compatible.csv")
    
    # 检查并补充 set_type 列（原CSV中缺少，必须添加）
    if "set_type" not in df.columns:
        # 自动分配训练集（前80%）和测试集（后20%）
        df["set_type"] = ["train" if i < int(0.8 * len(df)) else "test" for i in range(len(df))]

    # 提取特征列和标签列
    feature_cols = ["energy", "contrast", "correlation", "entropy"]
    X = df[feature_cols].values
    y = df["class"].values
    set_type = df["set_type"].values

    # 分离训练集和测试集
    X_train = X[set_type == "train"]
    y_train = y[set_type == "train"]
    X_test = X[set_type == "test"]
    y_test = y[set_type == "test"]

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练KNN模型
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)

    # 计算原始准确率
    original_acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    print(f"原始准确率: {original_acc:.4f}")

    # 计算置换重要性
    perm_importance = permutation_importance(
        knn,
        X_test_scaled,
        y_test,
        n_repeats=10,
        random_state=42,
    )

    # 整理并打印特征重要性
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": perm_importance.importances_mean,
        }
    ).sort_values(by="importance", ascending=False)
    
    print("特征重要性（置换重要性）：")
    print(importance_df)

    # 可视化（添加保存逻辑，避免容器内显示问题）
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("置换重要性得分（准确率下降值）")
    plt.ylabel("特征")
    plt.title("GLCM特征对KNN分类结果的贡献")
    plt.gca().invert_yaxis()
    plt.savefig("/app/feature_importance.png")  # 保存到容器内，可导出查看
    print("特征重要性图已保存为 feature_importance.png")

# 确保脚本可独立运行
if __name__ == "__main__":
    main()
