import tensorflow as tf
import random
from vit_keras import vit
from vit_attention.vit_SLA import create_vit_linear_attention
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle
import gc
import sys

# 设置随机种子确保结果可复现
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


# 配置GPU内存增长
def configure_memory():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[INFO] GPU内存动态增长已启用")
        except RuntimeError as e:
            print(f"[ERROR] 内存增长配置失败: {e}")


# 在程序开始时立即配置内存
configure_memory()


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
    vit_sla_results = []
    gt_results = []

    for pair_example in results_dictionary.keys():
        vit_results.append(results_dictionary[pair_example]['vit'])
        resnet_results.append(results_dictionary[pair_example]['resnet'])
        vgg_results.append(results_dictionary[pair_example]['vgg'])
        vit_sla_results.append(results_dictionary[pair_example]['vit_sla'])
        gt_results.append(results_dictionary[pair_example]['GT'])

    # Figures
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    lw = 3

    # ViT_B32
    fpr_vit, tpr_vit, thresholds_vit = roc_curve(gt_results, vit_results, pos_label=positive_label)
    auc_vit = auc(fpr_vit, tpr_vit)
    fnr_vit = 1 - tpr_vit
    eer_vit = fpr_vit[np.argmin(np.absolute(fnr_vit - fpr_vit))]
    ax.plot(fpr_vit, tpr_vit, linestyle='-', lw=lw, color='blue',
            label='ViT_B32 (EER=%s, AUC=%s)' % ('{0:.2f}'.format(eer_vit), '{0:.3f}'.format(auc_vit)))

    # ResNet_50
    fpr_resnet, tpr_resnet, thresholds_resnet = roc_curve(gt_results, resnet_results, pos_label=positive_label)
    auc_resnet = auc(fpr_resnet, tpr_resnet)
    fnr_resnet = 1 - tpr_resnet
    eer_resnet = fpr_resnet[np.argmin(np.absolute(fnr_resnet - fpr_resnet))]
    ax.plot(fpr_resnet, tpr_resnet, linestyle='-', lw=lw, color='orange',
            label='ResNet_50 (EER=%s, AUC=%s)' % ('{0:.2f}'.format(eer_resnet), '{0:.3f}'.format(auc_resnet)))

    # VGG_16
    fpr_vgg, tpr_vgg, thresholds_vgg = roc_curve(gt_results, vgg_results, pos_label=positive_label)
    auc_vgg = auc(fpr_vgg, tpr_vgg)
    fnr_vgg = 1 - tpr_vgg
    eer_vgg = fpr_vgg[np.argmin(np.absolute(fnr_vgg - fpr_vgg))]
    ax.plot(fpr_vgg, tpr_vgg, linestyle='-', lw=lw, color='green',
            label='VGG_16 (EER=%s, AUC=%s)' % ('{0:.2f}'.format(eer_vgg), '{0:.3f}'.format(auc_vgg)))

    # ViT-SLA
    fpr_vit_sla, tpr_vit_sla, thresholds_vit_sla = roc_curve(gt_results, vit_sla_results, pos_label=positive_label)
    auc_vit_sla = auc(fpr_vit_sla, tpr_vit_sla)
    fnr_vit_sla = 1 - tpr_vit_sla
    eer_vit_sla = fpr_vit_sla[np.argmin(np.absolute(fnr_vit_sla - fpr_vit_sla))]
    ax.plot(fpr_vit_sla, tpr_vit_sla, linestyle='-', lw=lw, color='red',
            label='ViT-SLA (EER=%s, AUC=%s)' % ('{0:.2f}'.format(eer_vit_sla), '{0:.3f}'.format(auc_vit_sla)))

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

    return auc_vit, auc_resnet, auc_vgg, auc_vit_sla


"""
CREATE DATASET WITH PAIRS
"""


def create_pairs_dataset(pairs_file, base_path):
    pairs = {}
    with open(pairs_file, 'r') as file:
        for idx, line in enumerate(file):
            line_aux = line.split('\n')[0].split('\t')
            if len(line_aux) < 4:
                line_aux.insert(2, line_aux[0])

            img_first = f"{line_aux[0]}_{''.join(str(i) for i in [0 for _ in range(4 - len(line_aux[1]))])}{line_aux[1]}.jpg"
            img_second = f"{line_aux[2]}_{''.join(str(i) for i in [0 for _ in range(4 - len(line_aux[3]))])}{line_aux[3]}.jpg"

            pairs[idx] = [
                f"{base_path}\\{line_aux[0]}\\{img_first}",
                f"{base_path}\\{line_aux[2]}\\{img_second}",
                1 if line_aux[0] == line_aux[2] else 0,
            ]
    return pairs


# 使用函数创建数据集
pairs_file = 'F:\\Paper ViT\\datasets\\LFW\\pairs.txt'
base_path = 'F:\\Paper ViT\\datasets\\LFW\\LFW'
pairs = create_pairs_dataset(pairs_file, base_path)

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
num_classes = 8631


# 定义模型加载函数
def load_model(model_type, image_size, num_classes, weights_path):
    print(f"[INFO] 加载 {model_type} 模型...")

    # 检查权重文件是否存在
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

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
            pooling='avg'  # 使用全局平均池化减少内存使用
        )
        y = tf.keras.layers.Dense(units=num_classes, activation='softmax',
                                  kernel_initializer=tf.keras.initializers.GlorotUniform())(base_model.output)
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
    elif model_type == "vit_sla":
        model = create_vit_linear_attention(
            image_size=image_size,
            patch_size=32,
            num_classes=num_classes,
            embed_dim=768,
            depth=8,
            num_heads=8,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.1
        )

    # 加载权重并创建特征模型
    model.load_weights(weights_path).expect_partial()
    feature_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)

    return feature_model


# 顺序加载模型并在加载之间清理内存
def load_all_models():
    models = {}
    model_configs = [
        ("vit", r"F:\Paper ViT\saved_results\Models\ViT_B32\checkpoint"),
        ("resnet", r"F:\Paper ViT\saved_results\Models\ResNet_50\checkpoint"),
        ("vgg", r"F:\Paper ViT\saved_results\Models\VGG_16\checkpoint"),
        ("vit_sla", r"F:\Paper ViT\train\vit\checkpoint")
    ]

    for model_type, weights_path in model_configs:
        try:
            # 清理现有模型和缓存数据
            tf.keras.backend.clear_session()
            gc.collect()

            # 加载模型
            models[model_type] = load_model(model_type, image_size, num_classes, weights_path)
            print(f"[INFO] 成功加载 {model_type} 模型")

        except Exception as e:
            print(f"[ERROR] 加载 {model_type} 模型失败: {e}")
            models[model_type] = None  # 设置为None，但继续尝试加载其他模型

    return models


try:
    # 加载所有模型
    models = load_all_models()

    # 更新模型引用
    vit_model = models.get('vit')
    resnet50_model = models.get('resnet')
    vgg16_model = models.get('vgg')
    vit_sla_model = models.get('vit_sla')

    # 验证所有模型是否成功加载
    if not all([vit_model, resnet50_model, vgg16_model, vit_sla_model]):
        print("[警告] 部分模型加载失败，将跳过这些模型的处理")

except Exception as e:
    print(f"[ERROR] 模型加载过程中发生错误: {e}")
    sys.exit(1)  # 如果发生严重错误，终止程序


# 使用tf.function加速模型推理
@tf.function
def predict_model(model, img):
    if model is None:
        return None  # 如果模型为None，直接返回None
    return model(img, training=False)


"""
PREPROCESS IMAGE PAIRS AND COMPUTE SCORE MODELS
"""

# 创建结果目录
os.makedirs("f:\\Paper ViT\\tmp", exist_ok=True)

# 尝试加载已有结果
try:
    with open("2_dataset_test/LFW/1/results.pickle", 'rb') as f:
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
        result_entry = {'GT': val[-1]}

        # 对每个模型分别处理，避免同时占用内存
        if vit_model is not None:
            embeddings1_vit = predict_model(vit_model, img1_).numpy()
            embeddings2_vit = predict_model(vit_model, img2_).numpy()
            result_entry['vit'] = compute_score(embeddings1_vit, embeddings2_vit)
        else:
            result_entry['vit'] = 0.0  # 如果模型为None，设置默认分数

        if resnet50_model is not None:
            embeddings1_resnet = predict_model(resnet50_model, img1_).numpy()
            embeddings2_resnet = predict_model(resnet50_model, img2_).numpy()
            result_entry['resnet'] = compute_score(embeddings1_resnet, embeddings2_resnet)
        else:
            result_entry['resnet'] = 0.0

        if vgg16_model is not None:
            embeddings1_vgg16 = predict_model(vgg16_model, img1_).numpy()
            embeddings2_vgg16 = predict_model(vgg16_model, img2_).numpy()
            result_entry['vgg'] = compute_score(embeddings1_vgg16, embeddings2_vgg16)
        else:
            result_entry['vgg'] = 0.0

        if vit_sla_model is not None:
            embeddings1_vit_sla = predict_model(vit_sla_model, img1_).numpy()
            embeddings2_vit_sla = predict_model(vit_sla_model, img2_).numpy()
            result_entry['vit_sla'] = compute_score(embeddings1_vit_sla, embeddings2_vit_sla)
        else:
            result_entry['vit_sla'] = 0.0

        results[key] = result_entry

        # 每10个样本打印一次进度
        if key % 10 == 0:
            print(f"\n[INFO] 处理进度: {key}/{len(pairs)}")
            print(f"[INFO] 当前对: {os.path.basename(val[0])} - {os.path.basename(val[1])}")
            print(f"[INFO] 真实标签: {val[-1]}")
            for model_name in ['vit', 'resnet', 'vgg', 'vit_sla']:
                if model_name in result_entry:
                    print(f"[INFO] {model_name.upper()}分数: {result_entry[model_name]:.4f}")

    except Exception as e:
        print(f"处理对 {key} 时出错: {e}")

# 保存结果到pickle文件
with open("2_dataset_test/LFW/1/results.pickle", 'wb') as f:
    pickle.dump(results, f)

# 打印结果摘要
print(f"\n[INFO] 结果摘要:")
for key, val in list(results.items())[:5]:  # 只打印前5个结果
    print(f"[INFO] 图像对 {key} -> \t真实标签: {val['GT']}")
    for model_name in ['vit', 'resnet', 'vgg', 'vit_sla']:
        if model_name in val:
            print(f"[INFO] \t {model_name.upper()}: \t\t{val[model_name]:.4f}")

# 计算ROC曲线
try:
    auc_vit, auc_resnet, auc_vgg, auc_vit_sla = compute_roc(results, fig_name='ROC_optimized', positive_label=1)
    print(f"[INFO] 处理完成! 结果已保存到 f:\\Paper ViT\\tmp\\results.pickle")
    print(f"[INFO] AUC值: ViT={auc_vit:.4f}, ResNet={auc_resnet:.4f}, VGG={auc_vgg:.4f}, ViT-SLA={auc_vit_sla:.4f}")
except Exception as e:
    print(f"[ERROR] 计算ROC曲线时出错: {e}")