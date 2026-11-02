import os
import pandas as pd
from itertools import combinations
from shutil import copy2  # 用于复制图片（可以换成 os.rename 以移动）

# === 配置 ===
attr_csv_path = r"D:\celebA\list_attr_celeba.csv"
img_dir = r"D:\celebA\img_align_celeba\img_align_celeba"  # 图像目录路径
output_dir = r"F:\Paper ViT\datasets\CelebA\CelebA"  # 输出分类目录
num_attrs_to_combine = 2  # 组合属性数，比如 2 表示 Black_Hair + Bangs

# === 加载属性CSV ===
df = pd.read_csv(attr_csv_path)
df.set_index("image_id", inplace=True)
df.replace(-1, 0, inplace=True)  # 把 -1 转成 0（无属性）

# 属性名列表
attr_names = df.columns.tolist()

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 遍历每张图像
for image_id, row in df.iterrows():
    # 找出该图像具备的属性（值为 1）
    present_attrs = [attr for attr in attr_names if row[attr] == 1]

    # 从这些属性中选择组合（如每张图组合2个属性）
    if len(present_attrs) >= num_attrs_to_combine:
        combs = list(combinations(present_attrs, num_attrs_to_combine))

        for comb in combs:
            class_name = "+".join(comb)
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            src_img_path = os.path.join(img_dir, image_id)
            dst_img_path = os.path.join(class_dir, image_id)

            # 复制图像到对应类别目录（注意：如使用移动可改成 os.rename）
            if os.path.exists(src_img_path):
                copy2(src_img_path, dst_img_path)

print("分类完成 ✅")
