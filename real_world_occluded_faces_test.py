import os
import pickle
import numpy as np
import pandas as pd
import gc # 导入垃圾回收模块
from vit_attention.vit_SLA import create_vit_linear_attention

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部日志, 1=INFO不显示, 2=INFO和WARNING不显示, 3=只显示ERROR
import tensorflow as tf
import matplotlib.pyplot as plt
from vit_keras import vit
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score
from tqdm import tqdm



# 设置 GPU 内存按需增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # 内存增长必须在 GPU 初始化之前设置
        print(e)



def remove_empty_directories():
    neutral_items = os.listdir(NEUTRAL_DIR)
    masked_items = os.listdir(MASKED_DIR)
    sunglasses_items = os.listdir(SUNGLASSES_DIR)

    for item in neutral_items:
        if not os.listdir(f"{NEUTRAL_DIR}/{item}"):
            print(f'Neutral folder ->\t{item} has no images, removing directory')
            os.rmdir(f'{NEUTRAL_DIR}/{item}')
    for item in masked_items:
        if not os.listdir(f"{MASKED_DIR}/{item}"):
            print(f'Masked folder ->\t{item} has no images, removing directory')
            os.rmdir(f'{MASKED_DIR}/{item}')
    for item in sunglasses_items:
        if not os.listdir(f"{SUNGLASSES_DIR}/{item}"):
            print(f'Sunglasses folder ->\t{item} has no images, removing directory')
            os.rmdir(f'{SUNGLASSES_DIR}/{item}')


def get_unique_names():
    neutral_items = os.listdir(NEUTRAL_DIR)
    masked_items = list(map(lambda x: x.split('_wearing_mask')[0], os.listdir(MASKED_DIR)))
    sunglasses_items = list(map(lambda x: x.split('_wearing_sunglasses')[0], os.listdir(SUNGLASSES_DIR)))
    return sorted(set(masked_items + neutral_items + sunglasses_items))


def get_histogram():
    data = {key: {'Neutral': None, 'Masked': None, 'Sunglasses': None} for key in UNIQUE_NAMES}

    neutral_folders = os.listdir(NEUTRAL_DIR)
    masked_folders = os.listdir(MASKED_DIR)
    sunglasses_folders = os.listdir(SUNGLASSES_DIR)

    for folder in neutral_folders:
        name = folder
        data[name]['Neutral'] = len(os.listdir(f"{NEUTRAL_DIR}/{folder}"))
    for folder in masked_folders:
        name = folder.split('_wearing_mask')[0]
        data[name]['Masked'] = len(os.listdir(f"{MASKED_DIR}/{folder}"))
    for folder in sunglasses_folders:
        name = folder.split('_wearing_sunglasses')[0]
        data[name]['Sunglasses'] = len(os.listdir(f"{SUNGLASSES_DIR}/{folder}"))

    data_aux = data.copy()
    for name in data.keys():
        if any(val is None for val in data[name].values()):
            data_aux.pop(name)

            # 使用 shutil.rmtree 替代 rm -r 命令
            import shutil
            try:
                if os.path.exists(f"{NEUTRAL_DIR}/{name}"):
                    shutil.rmtree(f"{NEUTRAL_DIR}/{name}")
            except Exception as e:
                print(f"删除目录错误 {NEUTRAL_DIR}/{name}: {e}")
            try:
                if os.path.exists(f"{MASKED_DIR}/{name}_wearing_mask"):
                    shutil.rmtree(f"{MASKED_DIR}/{name}_wearing_mask")
            except Exception as e:
                print(f"删除目录错误 {MASKED_DIR}/{name}_wearing_mask: {e}")
            try:
                if os.path.exists(f"{SUNGLASSES_DIR}/{name}_wearing_sunglasses"):
                    shutil.rmtree(f"{SUNGLASSES_DIR}/{name}_wearing_sunglasses")
            except Exception as e:
                print(f"删除目录错误 {SUNGLASSES_DIR}/{name}_wearing_sunglasses: {e}")
    data = data_aux.copy()
    return data


def plot_histogram(data_histogram):
    """绘制数据集直方图"""
    print("生成数据集直方图...")

    # 准备数据
    names = list(data_histogram.keys())
    neutral_counts = [data_histogram[name]['Neutral'] for name in names]
    masked_counts = [data_histogram[name]['Masked'] for name in names]
    sunglasses_counts = [data_histogram[name]['Sunglasses'] for name in names]

    # 设置图表
    plt.figure(figsize=(20, 10))

    # 设置柱状图位置
    x = np.arange(len(names))
    width = 0.25

    # 绘制柱状图
    plt.bar(x - width, neutral_counts, width, label='Neutral', color='#3498db')
    plt.bar(x, masked_counts, width, label='Masked', color='#e67e22')
    plt.bar(x + width, sunglasses_counts, width, label='Sunglasses', color='#95a5a6')

    # 设置图表标题和标签
    plt.title('Histogram', fontsize=24)
    plt.ylabel('Number of Images', fontsize=14)
    plt.xticks(x, names, rotation=90, fontsize=8)
    plt.yticks(fontsize=12)

    # 设置网格线和图例
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    os.makedirs(r'F:\Paper ViT\saved_results\Tests\RealWorldOccludedFaces', exist_ok=True)
    plt.savefig(r'F:\Paper ViT\saved_results\Tests\RealWorldOccludedFaces\histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("直方图已保存到RealWorldOccludedFaces/histogram.png")


def get_data():
    data = {key: {'Neutral': {}, 'Masked': {}, 'Sunglasses': {}} for key in UNIQUE_NAMES}

    neutral_folders = os.listdir(NEUTRAL_DIR)
    masked_folders = os.listdir(MASKED_DIR)
    sunglasses_folders = os.listdir(SUNGLASSES_DIR)

    for folder in neutral_folders:
        for file in sorted(os.listdir(f"{NEUTRAL_DIR}/{folder}")):
            person_name = folder
            file_name = file.split('.jpg')[0]
            file_path = os.path.join(f"{NEUTRAL_DIR}/{folder}/{file}")
            data[person_name]['Neutral'][file_name] = {
                'file': file_path,
                'embeddings': {
                    'vit': None,
                    'resnet': None,
                    'vgg': None,
                    'vit_sla':None,
                }
            }
    for folder in masked_folders:
        for file in sorted(os.listdir(f"{MASKED_DIR}/{folder}")):
            person_name = folder.split('_wearing_mask')[0]
            file_name = file.split('.jpg')[0]
            file_path = os.path.join(f"{MASKED_DIR}/{folder}/{file}")
            data[person_name]['Masked'][file_name] = {
                'file': file_path,
                'embeddings': {
                    'vit': None,
                    'resnet': None,
                    'vgg': None,
                    'vit_sla': None,
                }
            }
    for folder in sunglasses_folders:
        for file in sorted(os.listdir(f"{SUNGLASSES_DIR}/{folder}")):
            person_name = folder.split('_wearing_sunglasses')[0]
            file_name = file.split('.jpg')[0]
            file_path = os.path.join(f"{SUNGLASSES_DIR}/{folder}/{file}")
            data[person_name]['Sunglasses'][file_name] = {
                'file': file_path,
                'embeddings': {
                    'vit': None,
                    'resnet': None,
                    'vgg': None,
                    'vit_sla': None,  # 添加vit_sla嵌入向量

                }
            }
    return data


def remove_empty_files():
    """删除数据集中的空文件"""
    print("检查并删除空文件...")

    # 检查中性图像
    for person in os.listdir(NEUTRAL_DIR):
        person_dir = os.path.join(NEUTRAL_DIR, person)
        if os.path.isdir(person_dir):
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                if os.path.getsize(img_path) == 0:
                    print(f"删除空文件: {img_path}")
                    os.remove(img_path)

    # 检查戴口罩图像
    for person in os.listdir(MASKED_DIR):
        person_dir = os.path.join(MASKED_DIR, person)
        if os.path.isdir(person_dir):
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                if os.path.getsize(img_path) == 0:
                    print(f"删除空文件: {img_path}")
                    os.remove(img_path)

    # 检查戴墨镜图像
    for person in os.listdir(SUNGLASSES_DIR):
        person_dir = os.path.join(SUNGLASSES_DIR, person)
        if os.path.isdir(person_dir):
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                if os.path.getsize(img_path) == 0:
                    print(f"删除空文件: {img_path}")
                    os.remove(img_path)


def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise ValueError(f"图像文件不存在: {image_path}")

    # 检查文件大小
    if os.path.getsize(image_path) == 0:
        print(f"警告: 图像文件为空: {image_path}")
        # 返回一个空白图像而不是抛出错误
        blank_image = tf.zeros([224, 224, 3], dtype=tf.float32)
        blank_image = tf.expand_dims(blank_image, 0)  # 添加批次维度
        return blank_image

    img_ = tf.io.read_file(image_path)

    if tf.size(img_) == 0:  # 检查文件是否为空
        print(f"警告: 图像文件读取为空: {image_path}")
        # 返回一个空白图像而不是抛出错误
        blank_image = tf.zeros([224, 224, 3], dtype=tf.float32)
        blank_image = tf.expand_dims(blank_image, 0)  # 添加批次维度
        return blank_image

    try:
        img_ = tf.image.decode_jpeg(img_, channels=3)
        img_ = tf.image.resize(img_, [224, 224])
        img_ = img_ / 255.0  # 归一化到 [0, 1]
        img_ = tf.expand_dims(img_, 0)  # 添加批次维度
        return img_
    except Exception as e:
        print(f"处理图像时出错 {image_path}: {e}")
        # 返回一个空白图像而不是抛出错误
        blank_image = tf.zeros([224, 224, 3], dtype=tf.float32)
        blank_image = tf.expand_dims(blank_image, 0)  # 添加批次维度
        return blank_image


def compute_score(embeddings1, embeddings2):
    cosine_distance = cosine(embeddings1, embeddings2)
    score = 1 - cosine_distance
    return score


def plot_and_csv(models, ground_truth_, categories, positive_label=1):
    results_dir = os.path.join(r'F:\Paper ViT\saved_results\Tests\RealWorldOccludedFaces', f"CATEGORIES-{'_'.join(categories)}")
    os.makedirs(results_dir, exist_ok=True)

    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    for model in models.keys():
        model_name_ = models[model]['name']
        model_color_ = models[model]['color']
        model_scores_ = models[model]['scores']

        # Data
        fpr, tpr, thresholds = roc_curve(ground_truth_, model_scores_, pos_label=positive_label)
        auc_result = auc(fpr, tpr)
        fnr = 1 - tpr
        eer = fpr[np.argmin(np.absolute(fnr - fpr))]
        eer_threshold = thresholds[np.argmin(np.absolute(fnr - fpr))]

        # 优化1: 使用二分搜索策略寻找最佳F1分数
        # 首先找到分数的范围
        score_min, score_max = min(model_scores_), max(model_scores_)

        # 创建一个更小的阈值集合，但确保覆盖关键区域
        # 使用对数空间采样，在低分数区域采样更密集
        num_thresholds = 100  # 减少阈值数量但保持合理精度

        # 使用分位数采样，确保关键区域有足够的采样点
        quantiles = np.linspace(0, 1, num_thresholds)
        sampled_thresholds = np.quantile(model_scores_, quantiles)

        # 确保包含EER阈值
        if eer_threshold not in sampled_thresholds:
            sampled_thresholds = np.append(sampled_thresholds, eer_threshold)
            sampled_thresholds.sort()

        print(f"处理 {model_name_} 的 {len(sampled_thresholds)} 个阈值（从原始的 {len(thresholds)} 个中采样）")

        # 优化2: 向量化计算
        max_f1 = 0
        max_f1_recall = 0
        max_f1_precision = 0

        # 使用NumPy向量化操作加速计算
        ground_truth_array = np.array(ground_truth_)
        model_scores_array = np.array(model_scores_)

        for thresh in tqdm(sampled_thresholds, desc=f"处理 {model_name_} 阈值"):
            # 向量化预测
            predictions = (model_scores_array >= thresh).astype(int)

            # 计算混淆矩阵元素
            tp = np.sum((predictions == 1) & (ground_truth_array == 1))
            fp = np.sum((predictions == 1) & (ground_truth_array == 0))
            fn = np.sum((predictions == 0) & (ground_truth_array == 1))

            # 计算指标
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > max_f1:
                max_f1 = f1
                max_f1_precision = precision
                max_f1_recall = recall

        # Plot
        ax.plot(fpr, tpr, linestyle='-', lw=3, color=model_color_,
                label=f'{model_name_} (EER={eer:.2f}, AUC={auc_result:.3f}, R={max_f1_recall:.3f}, P={max_f1_precision:.3f}, F={max_f1:.3f})')
        ax.scatter(eer, tpr[np.argmin(np.absolute(fnr - fpr))], color=model_color_, linewidths=8, zorder=10)

        # CSV
        result_pd = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        result_pd['EER'] = pd.DataFrame([eer, tpr[np.argmin(np.absolute(fnr - fpr))]])
        result_pd.to_csv(f"{results_dir}/{model_name_}_ROC.csv", header=True, index=False)


    ax.set_title('Receiver Operating Characteristics (ROC)', fontsize=15)
    ax.set_xlabel('FPR', fontsize=15)
    ax.set_ylabel('TPR', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.legend(loc='lower right', prop={"size": 11})

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.savefig(f"{results_dir}/ROC.png", bbox_inches='tight')

    ax.set_xlim([0.0, 0.3])
    ax.set_ylim([0.7, 1.0])
    plt.savefig(f"{results_dir}/ROC_zoom.png", bbox_inches='tight')


def compute_roc(scores_dict, categories):
    print(f"开始计算ROC曲线，类别: {categories}")

    # 检查scores_dict是否为空
    if not scores_dict:
        print("错误: 分数字典为空!")
        return

    vit_scores = []
    resnet_scores = []
    vgg_scores = []
    vit_sla_scores = []  # 添加vit_sla分数列表
    ground_truth = []

    # 添加计数器来跟踪处理的数据
    processed_items = 0

    for neutral_person_ in scores_dict.keys():
        for neutral_file_ in scores_dict[neutral_person_].keys():
            for probe_person_, values in scores_dict[neutral_person_][neutral_file_].items():
                category_match = False

                for category_ in categories:
                    if category_ in probe_person_:
                        category_match = True
                        vit_scores.append(values['scores']['vit'])
                        resnet_scores.append(values['scores']['resnet'])
                        vgg_scores.append(values['scores']['vgg'])
                        vit_sla_scores.append(values['scores']['vit_sla'])  # 添加vit_sla分数
                        ground_truth.append(
                            1 if neutral_person_ == values['person'] else 0)  # 1 if same person, 0 if different
                        processed_items += 1

                        # 每处理1000个项目打印一次进度
                        if processed_items % 1000 == 0:
                            print(f"已处理 {processed_items} 个比较项")

    print(f"总共处理了 {processed_items} 个比较项")
    print(
        f"收集的分数: ViT={len(vit_scores)}, ResNet={len(resnet_scores)}, VGG={len(vgg_scores)}, ViT_SLA={len(vit_sla_scores)}")
    print(f"真值标签数量: {len(ground_truth)}")

    # 检查是否有足够的数据进行分析
    if len(ground_truth) == 0 or len(vit_scores) == 0:
        print("错误: 没有收集到足够的数据来生成ROC曲线!")
        return

    models_scores = {
        'vit': {'name': 'ViT_SLA', 'color': 'blue', 'scores': vit_scores},
        'resnet': {'name': 'ViT_B32', 'color': 'orange', 'scores': resnet_scores},
        'vgg': {'name': 'ResNet_50', 'color': 'green', 'scores': vgg_scores},
        'vit_sla': {'name': 'VGG_16', 'color': 'red', 'scores': vit_sla_scores},  # 添加vit_sla模型
    }

    print("开始绘制ROC曲线...")
    try:
        plot_and_csv(models_scores, ground_truth, categories)
        print(
            f"ROC曲线已保存到 F:\\Paper ViT\\saved_results\\Tests\\RealWorldOccludedFaces\\CATEGORIES-{'_'.join(categories)}")
    except Exception as e:
        print(f"绘制ROC曲线时出错: {e}")


"""
CREATE DATASET
"""


BASE_DIR = r"F:\RealWorldOccludedFaces-main\RealWorldOccludedFaces-main\images"
NEUTRAL_DIR = os.path.join(BASE_DIR, "neutral")
MASKED_DIR = os.path.join(BASE_DIR, "masked")
SUNGLASSES_DIR = os.path.join(BASE_DIR, "sunglasses")

remove_empty_directories()
remove_empty_files()

UNIQUE_NAMES = get_unique_names()
DATA_HISTOGRAM = get_histogram()
DATA = get_data()
plot_histogram(DATA_HISTOGRAM)


def load_and_compute_embeddings(model_name):
    """加载指定模型并计算嵌入向量"""
    print(f"\n正在加载 {model_name} 模型并计算嵌入向量...")

    IMAGE_SIZE = 224
    NUM_CLASSES = 8631

    if model_name == "vit_sla":
        # 加载 ViT_SLA 模型
        model = create_vit_linear_attention(
            image_size=IMAGE_SIZE,
            patch_size=32,
            num_classes=NUM_CLASSES,
            embed_dim=768,
            depth=8,
            num_heads=8,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.1
        )
        model.load_weights(r"F:\Paper ViT\tmp1\checkpoint").expect_partial()
        model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)

    elif model_name == "vit":
        # 加载 ViT_B32 模型
        model = vit.vit_b32(
            image_size=IMAGE_SIZE,
            pretrained=True,
            include_top=False,
            pretrained_top=False,
        )
        y = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(model.output)
        model = tf.keras.models.Model(inputs=model.input, outputs=y)
        model.load_weights(r"F:\Paper ViT\saved_results\Models\ViT_B32\checkpoint").expect_partial()
        model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)

    elif model_name == "resnet":
        # 加载 ResNet_50 模型
        model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            pooling=None,
        )
        Y = tf.keras.layers.GlobalAvgPool2D()(model.output)
        Y = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax',
                                  kernel_initializer=tf.keras.initializers.GlorotUniform())(Y)
        model = tf.keras.models.Model(inputs=model.input, outputs=Y, name='ResNet50')
        model.load_weights(r"F:\Paper ViT\saved_results\Models\ResNet_50\checkpoint").expect_partial()
        model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)

    elif model_name == "vgg":
        # 加载 VGG_16 模型
        model = tf.keras.applications.VGG16(
            include_top=True,
            weights="imagenet",
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            pooling=None,
        )
        Y = model.layers[-2].output
        Y = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax',
                                  kernel_initializer=tf.keras.initializers.GlorotUniform)(Y)
        model = tf.keras.models.Model(inputs=model.input, outputs=Y, name='VGG16')
        model.load_weights(r"F:\Paper ViT\saved_results\Models\VGG_16\checkpoint").expect_partial()
        model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)

    # 计算嵌入向量
    for person_ in tqdm(DATA.keys(), desc=f"计算 {model_name} 嵌入向量"):
        # 计算中性图像的嵌入向量
        for file_ in DATA[person_]['Neutral'].keys():
            img_path = DATA[person_]['Neutral'][file_]['file']
            img = preprocess_image(img_path)
            embedding = model.predict(img, verbose=0)
            DATA[person_]['Neutral'][file_]['embeddings'][model_name] = embedding[0]

        # 计算戴口罩图像的嵌入向量
        for file_ in DATA[person_]['Masked'].keys():
            img_path = DATA[person_]['Masked'][file_]['file']
            img = preprocess_image(img_path)
            embedding = model.predict(img, verbose=0)
            DATA[person_]['Masked'][file_]['embeddings'][model_name] = embedding[0]

        # 计算戴墨镜图像的嵌入向量
        for file_ in DATA[person_]['Sunglasses'].keys():
            img_path = DATA[person_]['Sunglasses'][file_]['file']
            img = preprocess_image(img_path)
            embedding = model.predict(img, verbose=0)
            DATA[person_]['Sunglasses'][file_]['embeddings'][model_name] = embedding[0]

    # 释放模型内存
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    print(f"{model_name} 模型的嵌入向量计算完成，已释放内存")


# 替换原有的"LOAD MODELS"和"PREPROCESS IMAGES AND COMPUTE EMBEDDINGS"部分
"""
LOAD MODELS AND COMPUTE EMBEDDINGS (SEQUENTIAL PROCESSING)
"""
# 顺序加载模型并计算嵌入向量
for model_name in ["vit", "resnet", "vgg", "vit_sla"]:
    load_and_compute_embeddings(model_name)

"""
COMPUTE SCORES
"""

print("\n计算相似度分数...")

# 创建一个字典来存储分数
scores_dict = {}

# 对于每个人的中性图像
for neutral_person in tqdm(DATA.keys(), desc="计算分数"):
    scores_dict[neutral_person] = {}

    # 对于该人的每张中性图像
    for neutral_file in DATA[neutral_person]['Neutral'].keys():
        scores_dict[neutral_person][neutral_file] = {}

        # 对于每个人的戴口罩和戴墨镜图像
        for probe_person in DATA.keys():
            # 处理戴口罩图像
            for masked_file in DATA[probe_person]['Masked'].keys():
                scores_dict[neutral_person][neutral_file][f"{probe_person}_Masked_{masked_file}"] = {
                    'person': probe_person,
                    'scores': {
                        'vit': compute_score(
                            DATA[neutral_person]['Neutral'][neutral_file]['embeddings']['vit'],
                            DATA[probe_person]['Masked'][masked_file]['embeddings']['vit']
                        ),
                        'resnet': compute_score(
                            DATA[neutral_person]['Neutral'][neutral_file]['embeddings']['resnet'],
                            DATA[probe_person]['Masked'][masked_file]['embeddings']['resnet']
                        ),
                        'vgg': compute_score(
                            DATA[neutral_person]['Neutral'][neutral_file]['embeddings']['vgg'],
                            DATA[probe_person]['Masked'][masked_file]['embeddings']['vgg']
                        ),
                        'vit_sla': compute_score(
                            DATA[neutral_person]['Neutral'][neutral_file]['embeddings']['vit_sla'],
                            DATA[probe_person]['Masked'][masked_file]['embeddings']['vit_sla']
                        )
                    }
                }

            # 处理戴墨镜图像
            for sunglasses_file in DATA[probe_person]['Sunglasses'].keys():
                scores_dict[neutral_person][neutral_file][f"{probe_person}_Sunglasses_{sunglasses_file}"] = {
                    'person': probe_person,
                    'scores': {
                        'vit': compute_score(
                            DATA[neutral_person]['Neutral'][neutral_file]['embeddings']['vit'],
                            DATA[probe_person]['Sunglasses'][sunglasses_file]['embeddings']['vit']
                        ),
                        'resnet': compute_score(
                            DATA[neutral_person]['Neutral'][neutral_file]['embeddings']['resnet'],
                            DATA[probe_person]['Sunglasses'][sunglasses_file]['embeddings']['resnet']
                        ),
                        'vgg': compute_score(
                            DATA[neutral_person]['Neutral'][neutral_file]['embeddings']['vgg'],
                            DATA[probe_person]['Sunglasses'][sunglasses_file]['embeddings']['vgg']
                        ),
                        'vit_sla': compute_score(
                            DATA[neutral_person]['Neutral'][neutral_file]['embeddings']['vit_sla'],
                            DATA[probe_person]['Sunglasses'][sunglasses_file]['embeddings']['vit_sla']
                        )
                    }
                }

# 计算ROC曲线
compute_roc(scores_dict, ['Masked'])
compute_roc(scores_dict, ['Sunglasses'])
compute_roc(scores_dict, ['Masked', 'Sunglasses'])

print("\n所有处理完成!")

