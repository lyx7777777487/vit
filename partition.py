import os
import shutil

# 读取身份标识文件
identity_file = 'd:\\celebA\\identity_CelebA.txt'
with open(identity_file, 'r') as f:
    lines = f.readlines()

# 创建目标文件夹
output_dir = 'd:\\celebA\\train_by_identity'
os.makedirs(output_dir, exist_ok=True)

# 组织图片
for line in lines:
    # 解析每一行
    img_name, identity_id = line.strip().split()

    # 构建源文件和目标文件路径
    src_path = os.path.join('d:\\celebA\\train', img_name)
    dst_dir = os.path.join(output_dir, f'id_{identity_id}')

    # 如果源文件存在
    if os.path.exists(src_path):
        # 创建身份ID对应的文件夹
        os.makedirs(dst_dir, exist_ok=True)

        # 复制图片到对应文件夹
        dst_path = os.path.join(dst_dir, img_name)
        shutil.copy2(src_path, dst_path)

print('整理完成！')