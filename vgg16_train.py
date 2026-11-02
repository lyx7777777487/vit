import datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from data_generator import create_data_generators

# ✅ 动态显存增长设置
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ✅ 混合精度策略
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

"""
HYPERPARAMETERS
"""
strategy = tf.distribute.MirroredStrategy()

image_size = 224
batch_size = 2 * strategy.num_replicas_in_sync  # ✅ 降低 batch_size
num_epochs = 25
learning_rate = 0.0001
num_classes = 8631

"""
DATASET
"""
train_gen, val_gen, test_gen = create_data_generators(target_size=image_size, batch_size=batch_size)

"""
MODEL
"""
with strategy.scope():
    base_model = tf.keras.applications.VGG16(
        include_top=False,  # ✅ 去掉默认全连接层，节省参数
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax',
              kernel_initializer=GlorotUniform(),
              dtype='float32')(x)  # ✅ 强制输出层为 float32
    vgg_model = Model(inputs=base_model.input, outputs=x)

vgg_model.summary()

"""
MODEL COMPILE
"""
with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    vgg_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top-5-accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top-10-accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=100, name='top-100-accuracy'),
        ]
    )

"""
CALLBACKS
"""
checkpoint_filepath = "./tmp/checkpoint"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='max',
)

csv_logger = tf.keras.callbacks.CSVLogger(
    "./tmp/training_log.csv",
    separator=',',
    append=True,
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,
)

# ✅ 关闭 write_graph 和 histogram_freq，节省资源
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir="./tmp/logs" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S"),
    histogram_freq=0,
    write_graph=False,
    update_freq='epoch',
)

"""
LOAD WEIGHTS (可选)
"""
# Load pre-trained model weights before training
best_weights = r"F:\Paper ViT\LFW_results\vgg16/checkpoint"
vgg_model.load_weights(best_weights)


"""
TRAIN
"""
history = vgg_model.fit(
    train_gen,
    epochs=num_epochs,
    validation_data=val_gen,
    callbacks=[checkpoint_callback, csv_logger, early_stopping, tb_callback]
)

"""
EVALUATE
"""
vgg_model.load_weights(checkpoint_filepath)
loss, accuracy, top5, top10, top100 = vgg_model.evaluate(test_gen)
print(f"Test Accuracy: {round(accuracy*100, 2)}%")
print(f"Top-5 Accuracy: {round(top5*100, 2)}%")
print(f"Top-10 Accuracy: {round(top10*100, 2)}%")
print(f"Top-100 Accuracy: {round(top100*100, 2)}%")

"""
HISTORY FIGURES
"""

# PLOTS
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./tmp/model accuracy.png')
plt.close()
# summarize history for Top 5 accuracy
plt.plot(history.history['top-5-accuracy'])
plt.plot(history.history['val_top-5-accuracy'])
plt.title('model top 5 accuracy')
plt.ylabel('top 5 accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./tmp/model top 5 accuracy.png')
plt.close()
# summarize history for Top 10 accuracy
plt.plot(history.history['top-10-accuracy'])
plt.plot(history.history['val_top-10-accuracy'])
plt.title('model top 10 accuracy')
plt.ylabel('top 10 accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./tmp/model top 10 accuracy.png')
plt.close()
# summarize history for Top 100 accuracy
plt.plot(history.history['top-100-accuracy'])
plt.plot(history.history['val_top-100-accuracy'])
plt.title('model top 100 accuracy')
plt.ylabel('top 100 accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./tmp/model top 100 accuracy.png')
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./tmp/model loss.png')
plt.close()
