import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from vit_keras import vit
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score
from tqdm import tqdm
from vit_attention.vit_SLA import create_vit_linear_attention

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

    # Remove any person which does not have images for all categories
    data_aux = data.copy()
    for name in data.keys():
        if any(val is None for val in data[name].values()):
            data_aux.pop(name)

            # Also remove directories
            try:
                os.system(f'rm -r {NEUTRAL_DIR}/{name}')
            except FileNotFoundError: pass
            try:
                os.system(f'rm -r {MASKED_DIR}/{name}_wearing_mask')
            except FileNotFoundError: pass
            try:
                os.system(f'rm -r {SUNGLASSES_DIR}/{name}_wearing_sunglasses')
            except FileNotFoundError: pass
    data = data_aux.copy()
    return data


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
                    'vit_sla': None,

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
                    'vit_sla': None,
                }
            }
    return data


def preprocess_image(img_path):
    img_ = tf.io.read_file(img_path)
    img_ = tf.image.decode_jpeg(img_, channels=3)
    img_ = tf.image.convert_image_dtype(img_, dtype=tf.float32)
    img_ = tf.image.resize(img_, [224, 224])
    img_ = tf.expand_dims(img_, axis=0)
    return img_


def compute_score(embeddings1, embeddings2):
    cosine_distance = cosine(embeddings1, embeddings2)
    score = 1 - cosine_distance
    return score


def plot_and_csv(models, ground_truth_, categories, positive_label=1):
    results_dir = os.path.join(r'F:\Paper ViT\saved_results\Tests\RealWorldOccludedFaces', f"CATEGORIES-{'_'.join(categories)}")
    try:
        os.mkdir(results_dir)
    except FileExistsError:
        pass

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

        # Find the maximum F1 score and corresponding threshold
        max_f1 = 0
        max_f1_recall = 0
        max_f1_precision = 0

        for thresh in tqdm(thresholds, desc="Processing thresholds"):
            binarized_results = [1 if score >= thresh else 0 for score in model_scores_]
            current_fscore = f1_score(ground_truth_, binarized_results)
            if current_fscore > max_f1:
                max_f1 = current_fscore
                max_f1_recall = recall_score(ground_truth_, binarized_results)
                max_f1_precision = precision_score(ground_truth_, binarized_results)

        # Plot
        ax.plot(fpr, tpr, linestyle='-', lw=3, color=model_color_, label=f'{model_name_} (EER={eer:.2f}, AUC={auc_result:.3f}, R={max_f1_recall:.3f}, P={max_f1_precision:.3f}, F={max_f1:.3f})')
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
    vit_scores = []
    resnet_scores = []
    vit_sla_scores = []
    ground_truth = []

    for neutral_person_ in scores_dict.keys():
        for neutral_file_ in scores_dict[neutral_person_].keys():
            for probe_person_, values in scores_dict[neutral_person_][neutral_file_].items():

                for category_ in categories:
                    if category_ in probe_person_:
                        vit_scores.append(values['scores']['vit'])
                        resnet_scores.append(values['scores']['resnet'])
                        vit_sla_scores.append(values['scores']['vit_sla'])


                        ground_truth.append(1 if neutral_person_ == values['person'] else 0)  # 1 if same person, 0 if different

    models_scores = {
        'vit':          {'name': 'ViT_B32',         'color': 'blue',    'scores': vit_scores},
        'resnet':       {'name': 'ResNet_50',       'color': 'orange',  'scores': resnet_scores},
        'vit_sla':      {'name': 'ViT_SLA',          'color': 'green',   'scores': vit_sla_scores},

    }
    plot_and_csv(models_scores, ground_truth, categories)


"""
CREATE DATASET
"""


BASE_DIR = r"F:\RealWorldOccludedFaces-main\RealWorldOccludedFaces-main\images"
NEUTRAL_DIR = f"{BASE_DIR}/neutral"
MASKED_DIR = f"{BASE_DIR}/masked"
SUNGLASSES_DIR = f"{BASE_DIR}/sunglasses"

remove_empty_directories()

UNIQUE_NAMES = get_unique_names()
DATA_HISTOGRAM = get_histogram()
DATA = get_data()


"""
LOAD MODELS
"""


IMAGE_SIZE = 224
NUM_CLASSES = 8631

""" ViT_B32 """
vit_model = vit.vit_b32(
    image_size=IMAGE_SIZE,
    pretrained=True,
    include_top=False,
    pretrained_top=False,
)
y = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(vit_model.output)
vit_model = tf.keras.models.Model(inputs=vit_model.input, outputs=y)

vit_model.load_weights(r"F:\Paper ViT\3_model_train\vit\checkpoint").expect_partial()   # suppresses warnings
vit_model = tf.keras.models.Model(inputs=vit_model.input, outputs=vit_model.layers[-2].output)
vit_model.summary()

""" ResNet_50 """
resnet50_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    pooling=None,
)
Y = tf.keras.layers.GlobalAvgPool2D()(resnet50_model.output)
Y = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform())(Y)
resnet50_model = tf.keras.models.Model(inputs=resnet50_model.input, outputs=Y, name='ResNet50')

resnet50_model.load_weights(r"F:\Paper ViT\3_model_train\resnet\checkpoint").expect_partial()   # suppresses warnings
resnet50_model = tf.keras.models.Model(inputs=resnet50_model.input, outputs=resnet50_model.layers[-2].output)
resnet50_model.summary()

""" ViT_SLA """
vit_sla_model = create_vit_linear_attention(
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

vit_sla_model.load_weights(r"F:\Paper ViT\vit_attention\LFW_results\vit_linear2\checkpoint").expect_partial()   # suppresses warnings
vit_sla_model = tf.keras.models.Model(inputs=vit_sla_model.input, outputs=vit_sla_model.layers[-2].output)
vit_sla_model.summary()


"""
PREPROCESS IMAGES AND COMPUTE EMBEDDINGS
"""


try:
    with open(r'F:\Paper ViT\saved_results\Tests\RealWorldOccludedFaces\embeddings.pickle', 'rb') as embeddings_file:
        DATA = pickle.load(embeddings_file)
except FileNotFoundError:
    for person in DATA.keys():
        for category in DATA[person].keys():
            for element in DATA[person][category].keys():
                img = preprocess_image(DATA[person][category][element]['file'])

                embeddings_vit = vit_model(img).numpy()
                embeddings_resnet = resnet50_model(img).numpy()
                embeddings_vit_sla = vit_sla_model(img).numpy()


                DATA[person][category][element]['embeddings']['vit'] = embeddings_vit
                DATA[person][category][element]['embeddings']['resnet'] = embeddings_resnet
                DATA[person][category][element]['embeddings']['vit_sla'] = embeddings_vit_sla


    with open(r'F:\Paper ViT\saved_results\Tests\RealWorldOccludedFaces\embeddings.pickle', 'wb') as embeddings_file:
        pickle.dump(DATA, embeddings_file)


"""
MATCH IMAGES TO OBTAIN MATCHING SCORES
"""


try:
    with open(r'F:\Paper ViT\saved_results\Tests\RealWorldOccludedFaces\scores.pickle', 'rb') as scores_file:
        SCORES = pickle.load(scores_file)
except FileNotFoundError:
    SCORES = {person: {} for person in DATA.keys()}
    compare_with = [
        'masked',
        'sunglasses',
    ]

    for neutral_person in SCORES.keys():
        for neutral_file in sorted(os.listdir(f'{NEUTRAL_DIR}/{neutral_person}')):
            neutral_id = f"{neutral_person}_{neutral_file.split('.jpg')[0]}"
            SCORES[neutral_person][neutral_id] = {}

            if 'masked' in compare_with:
                for masked_person in sorted(os.listdir(MASKED_DIR)):
                    for masked_file in sorted(os.listdir(f'{MASKED_DIR}/{masked_person}')):
                        masked_id = f"{masked_person}_{masked_file.split('.jpg')[0]}"

                        SCORES[neutral_person][neutral_id][masked_id] = {
                            'person': masked_person.split('_wearing_mask')[0],
                            'scores': {
                                'vit': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['vit'],
                                    DATA[masked_person.split('_wearing_mask')[0]]['Masked'][masked_file.split('.jpg')[0]]['embeddings']['vit']
                                ),
                                'resnet': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['resnet'],
                                    DATA[masked_person.split('_wearing_mask')[0]]['Masked'][masked_file.split('.jpg')[0]]['embeddings']['resnet']
                                ),
                                'vit_sla': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['vit_sla'],
                                    DATA[masked_person.split('_wearing_mask')[0]]['Masked'][masked_file.split('.jpg')[0]]['embeddings']['vit_sla']
                                ),
                            }
                        }

            if 'sunglasses' in compare_with:
                for sunglasses_person in sorted(os.listdir(SUNGLASSES_DIR)):
                    for sunglasses_file in sorted(os.listdir(f'{SUNGLASSES_DIR}/{sunglasses_person}')):
                        sunglasses_id = f"{sunglasses_person}_{sunglasses_file.split('.jpg')[0]}"

                        SCORES[neutral_person][neutral_id][sunglasses_id] = {
                            'person': sunglasses_person.split('_wearing_sunglasses')[0],
                            'scores': {
                                'vit': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['vit'],
                                    DATA[sunglasses_person.split('_wearing_sunglasses')[0]]['Sunglasses'][sunglasses_file.split('.jpg')[0]]['embeddings']['vit']
                                ),
                                'resnet': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['resnet'],
                                    DATA[sunglasses_person.split('_wearing_sunglasses')[0]]['Sunglasses'][sunglasses_file.split('.jpg')[0]]['embeddings']['resnet']
                                ),
                                'vit_sla': compute_score(
                                    DATA[neutral_person]['Neutral'][neutral_file.split('.jpg')[0]]['embeddings']['vit_sla'],
                                    DATA[sunglasses_person.split('_wearing_sunglasses')[0]]['Sunglasses'][sunglasses_file.split('.jpg')[0]]['embeddings']['vit_sla']
                                ),
                            }
                        }

    with open(r'F:\Paper ViT\saved_results\Tests\RealWorldOccludedFaces\scores.pickle', 'wb') as scores_file:
        pickle.dump(SCORES, scores_file)


"""
COMPUTE ROC CURVES
"""


compute_roc(
    SCORES,
    categories=[
        'mask',
        'sunglasses',
    ]
)
