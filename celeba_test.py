import tensorflow as tf
import random
from vit_keras import vit
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle

# 设置随机种子确保结果可复现
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


# 使用tf.function加速预处理
@tf.function
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    # 添加更多的图像预处理和增强
    img = tf.image.resize(img, [224, 224])

    # 标准化图像 - 使用ImageNet的均值和标准差
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    img = (img - mean) / std

    img = tf.expand_dims(img, axis=0)
    return img


def preprocess(paths):
    path_img1, path_img2 = paths
    img1 = preprocess_image(path_img1)
    img2 = preprocess_image(path_img2)
    return [img1, img2]


# 优化余弦相似度计算
def compute_score(embeddings1, embeddings2):
    # 确保embeddings是一维的
    if len(embeddings1.shape) > 1:
        embeddings1 = embeddings1.flatten()
    if len(embeddings2.shape) > 1:
        embeddings2 = embeddings2.flatten()

    # 使用L2归一化提高相似度计算的稳定性
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2)

    cosine_distance = cosine(embeddings1, embeddings2)
    score = 1 - cosine_distance
    return score


def compute_roc(results_dictionary, fig_name, positive_label=1):
    vit_results = []
    resnet_results = []
    vgg_results = []
    gt_results = []

    for pair_example in results_dictionary.keys():
        vit_results.append(results_dictionary[pair_example]['vit'])
        resnet_results.append(results_dictionary[pair_example]['resnet'])
        vgg_results.append(results_dictionary[pair_example]['vgg'])
        gt_results.append(results_dictionary[pair_example]['GT'])

    # Figures
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    lw = 3

    """ ViT_B32 """
    # Data
    fpr_vit, tpr_vit, thresholds_vit = roc_curve(gt_results, vit_results, pos_label=positive_label)
    auc_vit = auc(fpr_vit, tpr_vit)
    fnr_vit = 1 - tpr_vit
    eer_vit = fpr_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))]
    eer_vit_threshold = thresholds_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))]

    # Plot
    ax.plot(fpr_vit, tpr_vit, linestyle='-', lw=lw, color='blue',
            label='ViT_B32 (EER=%s, AUC=%s)' % ('{0:.2f}'.format(eer_vit), '{0:.3f}'.format(auc_vit)))
    ax.scatter(eer_vit, tpr_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))], color='blue', linewidths=8, zorder=10)

    # CSV
    vit_pd = pd.DataFrame({'FPR_ViT': fpr_vit, 'TPR_ViT': tpr_vit})
    vit_pd['EER_ViT'] = pd.DataFrame([eer_vit, tpr_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))]])
    os.makedirs("f:\\Paper ViT\\saved_results\\Tests\\CelebA\\tmp", exist_ok=True)
    vit_pd.to_csv("f:\\Paper ViT\\tmp\\ViT_B32_ROC.csv", header=True, index=False)

    """ ResNet_50 """
    # Data
    fpr_resnet, tpr_resnet, thresholds_resnet = roc_curve(gt_results, resnet_results, pos_label=positive_label)
    auc_resnet = auc(fpr_resnet, tpr_resnet)
    fnr_resnet = 1 - tpr_resnet
    eer_resnet = fpr_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))]
    eer_resnet_threshold = thresholds_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))]

    # Plot
    ax.plot(fpr_resnet, tpr_resnet, linestyle='-', lw=lw, color='orange',
            label='ResNet_50 (EER=%s, AUC=%s)' % ('{0:.2f}'.format(eer_resnet), '{0:.3f}'.format(auc_resnet)))
    ax.scatter(eer_resnet, tpr_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))], color='orange', linewidths=8,
               zorder=10)

    # CSV
    resnet_pd = pd.DataFrame({'FPR_RESNET': fpr_resnet, 'TPR_RESNET': tpr_resnet})
    resnet_pd['EER_RESNET'] = pd.DataFrame([eer_resnet, tpr_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))]])
    resnet_pd.to_csv("f:\\Paper ViT\\tmp\\ResNet_50_ROC.csv", header=True, index=False)

    """ VGG_16 """
    # Data
    fpr_vgg, tpr_vgg, thresholds_vgg = roc_curve(gt_results, vgg_results, pos_label=positive_label)
    auc_vgg = auc(fpr_vgg, tpr_vgg)
    fnr_vgg = 1 - tpr_vgg
    eer_vgg = fpr_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))]
    eer_vgg_threshold = thresholds_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))]

    # Plot
    ax.plot(fpr_vgg, tpr_vgg, linestyle='-', lw=lw, color='green',
            label='VGG_16 (EER=%s, AUC=%s)' % ('{0:.2f}'.format(eer_vgg), '{0:.3f}'.format(auc_vgg)))
    ax.scatter(eer_vgg, tpr_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))], color='green', linewidths=8, zorder=10)

    # CSV
    vgg_pd = pd.DataFrame({'FPR_VGG': fpr_vgg, 'TPR_VGG': tpr_vgg})
    vgg_pd['EER_VGG'] = pd.DataFrame([eer_vgg, tpr_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))]])
    vgg_pd.to_csv("f:\\Paper ViT\\tmp\\VGG_16_ROC.csv", header=True, index=False)

    ax.set_title('Receiver Operating Characteristics (ROC)', fontsize=15)
    ax.set_xlabel('FPR', fontsize=15)
    ax.set_ylabel('TPR', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.legend(loc='lower right', prop={"size": 11})

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.savefig(f"f:\\Paper ViT\\tmp\\{fig_name}.png", bbox_inches='tight')

    ax.set_xlim([0.0, 0.3])
    ax.set_ylim([0.7, 1.0])
    plt.savefig(f"f:\\Paper ViT\\tmp\\{fig_name}_zoom.png", bbox_inches='tight')

    # 返回AUC值用于比较
    return auc_vit, auc_resnet, auc_vgg


def create_pairs_dataset(base_path, num_pairs=10000):
    """直接从图像文件夹创建配对数据集"""
    # 获取所有图像文件
    all_images = os.listdir(base_path)
    print(f"[INFO] 总共找到 {len(all_images)} 张图像")

    # 创建配对
    pairs = {}
    pair_id = 0

    # 生成正样本对（相同人的不同图像）
    print("[INFO] 生成正样本对...")
    num_positive = num_pairs // 2
    while pair_id < num_positive:
        # 随机选择两张图像
        img1, img2 = random.sample(all_images, 2)
        # 获取身份标识（假设图像名称格式为：000001.jpg）
        id1 = int(img1.split('.')[0])
        id2 = int(img2.split('.')[0])

        # 如果是同一个人的图像（假设每个人有5张图像）
        if id1 // 5 == id2 // 5 and img1 != img2:
            pairs[pair_id] = [
                os.path.join(base_path, img1),
                os.path.join(base_path, img2),
                1  # 正样本标签
            ]
            pair_id += 1

    # 生成负样本对（不同人的图像）
    print("[INFO] 生成负样本对...")
    while pair_id < num_pairs:
        img1, img2 = random.sample(all_images, 2)
        id1 = int(img1.split('.')[0])
        id2 = int(img2.split('.')[0])

        # 如果是不同人的图像
        if id1 // 5 != id2 // 5:
            pairs[pair_id] = [
                os.path.join(base_path, img1),
                os.path.join(base_path, img2),
                0  # 负样本标签
            ]
            pair_id += 1

    print(f"[INFO] 成功创建 {len(pairs)} 个图像对")
    return pairs


# 1. 修改 create_pairs_dataset 调用部分
base_path = 'F:\\Paper ViT\\datasets\\CelebA\\img_align_celeba'
# 可以指定生成的配对数量，默认10000对
pairs = create_pairs_dataset(base_path, num_pairs=10000)

# 添加路径检查
print("\n检查路径...")
invalid_paths = []
for key, val in pairs.items():
    if not os.path.exists(val[0]) or not os.path.exists(val[1]):
        print(f"警告: 图像对 {key} 的路径有问题:")
        print(f"图像1存在: {os.path.exists(val[0])} - {val[0]}")
        print(f"图像2存在: {os.path.exists(val[1])} - {val[1]}")
        invalid_paths.append(key)

# 移除无效路径
for key in invalid_paths:
    del pairs[key]

print(f"[INFO] 有效图像对: {len(pairs)}")

"""
LOAD MODELS
"""

# 使用混合精度训练提高性能
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("[INFO] 使用混合精度: float16")
except:
    print("[INFO] 当前TensorFlow版本不支持混合精度")

image_size = 224
num_classes = 10177  # CelebA数据集的身份数


# 定义模型加载函数
def load_model(model_type, image_size, num_classes, weights_path):
    print(f"[INFO] 加载 {model_type} 模型...")

    if model_type == "vit":
        base_model = vit.vit_b32(
            image_size=image_size,
            pretrained=True,
            include_top=False,
            pretrained_top=False,
        )
        y = tf.keras.layers.Dense(num_classes, activation='softmax')(base_model.output)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=y)

    elif model_type == "resnet":
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
            pooling=None,
        )
        y = tf.keras.layers.GlobalAvgPool2D()(base_model.output)
        y = tf.keras.layers.Dense(units=num_classes, activation='softmax',
                                  kernel_initializer=tf.keras.initializers.GlorotUniform())(y)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=y, name='ResNet50')

    elif model_type == "vgg":
        base_model = tf.keras.applications.VGG16(
            include_top=True,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
            pooling=None,
        )
        y = base_model.layers[-2].output
        y = tf.keras.layers.Dense(units=num_classes, activation='softmax',
                                  kernel_initializer=tf.keras.initializers.GlorotUniform())(y)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=y, name='VGG16')

    model.load_weights(weights_path).expect_partial()
    feature_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)

    return feature_model


# 加载所有模型
vit_model = load_model("vit", image_size, num_classes, "./CelebA_results/vitB32/checkpoint")
resnet50_model = load_model("resnet", image_size, num_classes, "./CelebA_results/resnet50/checkpoint")
vgg16_model = load_model("vgg", image_size, num_classes, "./saved_results/Models/VGG_16/checkpoint")


# 使用tf.function加速模型推理
@tf.function
def predict_model(model, img):
    return model(img, training=False)


"""
PREPROCESS IMAGE PAIRS AND COMPUTE SCORE MODELS
"""

# 创建结果目录
os.makedirs("f:\\Paper ViT\\saved_results\\Tests\\CelebA", exist_ok=True)
os.makedirs("f:\\Paper ViT\\tmp", exist_ok=True)

# 尝试加载已有结果
try:
    with open("f:\\Paper ViT\\tmp\\celeba_results.pickle", 'rb') as f:
        print("[INFO] 加载已有结果...")
        results = pickle.load(f)
        print(f"[INFO] 已加载 {len(results)} 个结果")
except:
    print("[INFO] 未找到已有结果，将重新计算")
    results = {}

# 处理图像对
print("\n[INFO] 处理图像对...")
for key, val in tqdm(pairs.items(), desc="处理进度"):
    # 跳过已处理的图像对
    if key in results:
        continue

    try:
        img1_, img2_ = preprocess(val[:-1])

        # 使用tf.function加速的模型推理
        embeddings1_vit = predict_model(vit_model, img1_).numpy()
        embeddings2_vit = predict_model(vit_model, img2_).numpy()
        score_vit = compute_score(embeddings1_vit, embeddings2_vit)

        embeddings1_resnet = predict_model(resnet50_model, img1_).numpy()
        embeddings2_resnet = predict_model(resnet50_model, img2_).numpy()
        score_resnet = compute_score(embeddings1_resnet, embeddings2_resnet)

        embeddings1_vgg16 = predict_model(vgg16_model, img1_).numpy()
        embeddings2_vgg16 = predict_model(vgg16_model, img2_).numpy()
        score_vgg16 = compute_score(embeddings1_vgg16, embeddings2_vgg16)

        results[key] = {'vit': score_vit, 'resnet': score_resnet, 'vgg': score_vgg16, 'GT': val[-1]}

        # 每10个样本打印一次进度
        if key % 10 == 0:
            print(f"\n[INFO] 处理进度: {key}/{len(pairs)}")
            print(f"[INFO] 当前对: {os.path.basename(val[0])} - {os.path.basename(val[1])}")
            print(f"[INFO] 真实标签: {val[-1]}")
            print(f"[INFO] ViT分数: {score_vit:.4f}")
            print(f"[INFO] ResNet分数: {score_resnet:.4f}")
            print(f"[INFO] VGG分数: {score_vgg16:.4f}")

    except Exception as e:
        print(f"处理对 {key} 时出错: {e}")

# 保存结果到pickle文件
with open("f:\\Paper ViT\\tmp\\celeba_results.pickle", 'wb') as f:
    pickle.dump(results, f)

# 打印结果摘要
print(f"\n[INFO] 结果摘要:")
for key, val in list(results.items())[:5]:  # 只打印前5个结果
    print(f"[INFO] 图像对 {key} -> \t真实标签: {val['GT']}")
    print(f"[INFO] \t ViT: \t\t{val['vit']:.4f}")
    print(f"[INFO] \t ResNet: \t{val['resnet']:.4f}")
    print(f"[INFO] \t VGG: \t\t{val['vgg']:.4f}")

# 计算ROC曲线
auc_vit, auc_resnet, auc_vgg = compute_roc(results, fig_name='ROC_CelebA_optimized', positive_label=1)

print(f"[INFO] 处理完成! 结果已保存到 f:\\Paper ViT\\tmp\\celeba_results.pickle")
print(f"[INFO] AUC值: ViT={auc_vit:.4f}, ResNet={auc_resnet:.4f}, VGG={auc_vgg:.4f}")