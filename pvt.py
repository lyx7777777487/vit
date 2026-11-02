import datetime
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from data_generator import create_data_generators

"""
PVT MODEL IMPLEMENTATION
"""


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, sr_ratio=1, qkv_bias=False, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = tf.keras.layers.Dense(embed_dim, use_bias=qkv_bias)
        self.kv = tf.keras.layers.Dense(embed_dim * 2, use_bias=qkv_bias)
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(embed_dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = tf.keras.layers.Conv2D(embed_dim, kernel_size=sr_ratio, strides=sr_ratio, padding='valid')
            self.norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, x, H, W, training=None):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # Query projection
        q = self.q(x)  # (B, N, C)
        q = tf.reshape(q, (B, N, self.num_heads, self.head_dim))
        q = tf.transpose(q, (0, 2, 1, 3))  # (B, num_heads, N, head_dim)

        # Key-Value projection with spatial reduction
        if self.sr_ratio > 1:
            x_reshaped = tf.reshape(x, (B, H, W, C))
            x_sr = self.sr(x_reshaped)  # Spatial reduction
            x_sr = tf.reshape(x_sr, (B, -1, C))
            x_sr = self.norm(x_sr)
            kv = self.kv(x_sr)
        else:
            kv = self.kv(x)

        kv = tf.reshape(kv, (B, -1, 2, self.num_heads, self.head_dim))
        kv = tf.transpose(kv, (2, 0, 3, 1, 4))  # (2, B, num_heads, N, head_dim)
        k, v = kv[0], kv[1]

        # Attention computation
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = tf.matmul(attn, v)  # (B, num_heads, N, head_dim)
        x = tf.transpose(x, (0, 2, 1, 3))  # (B, N, num_heads, head_dim)
        x = tf.reshape(x, (B, N, C))

        x = self.proj(x)
        x = self.proj_drop(x, training=training)

        return x


class MLP(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features, out_features=None, drop=0., **kwargs):
        super().__init__(**kwargs)
        out_features = out_features or in_features  # 确保输出维度正确
        self.fc1 = tf.keras.layers.Dense(hidden_features, activation='gelu')
        self.fc2 = tf.keras.layers.Dense(out_features)
        self.drop = tf.keras.layers.Dropout(drop)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., sr_ratio=1, qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., **kwargs):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, sr_ratio=sr_ratio,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        # 修复：明确指定in_features和out_features
        self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_hidden_dim, out_features=embed_dim, drop=drop)

        # DropPath implementation
        self.drop_path_rate = drop_path
        if drop_path > 0.:
            self.drop_path = tf.keras.layers.Dropout(drop_path, noise_shape=(None, 1, 1))
        else:
            self.drop_path = lambda x, training=None: x

    def call(self, x, H, W, training=None):
        # Self-attention with residual connection
        attn_out = self.attn(self.norm1(x), H, W, training=training)
        x = x + self.drop_path(attn_out, training=training)

        # MLP with residual connection
        mlp_out = self.mlp(self.norm2(x), training=training)
        x = x + self.drop_path(mlp_out, training=training)

        return x


class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.H = img_size // patch_size
        self.W = img_size // patch_size
        self.num_patches = self.H * self.W

        # 卷积层用于patch embedding
        self.proj = tf.keras.layers.Conv2D(
            embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid',
            name=f'patch_embed_conv_{embed_dim}'
        )

    def build(self, input_shape):
        super().build(input_shape)
        # 在build阶段初始化LayerNormalization，确保输入形状已知
        self.norm = tf.keras.layers.LayerNormalization(
            axis=-1,
            name=f'patch_embed_norm_{self.embed_dim}'
        )

    def call(self, x, training=None):
        B = tf.shape(x)[0]

        # Patch embedding through convolution
        x = self.proj(x)  # (B, H//patch_size, W//patch_size, embed_dim)
        H_new, W_new = tf.shape(x)[1], tf.shape(x)[2]

        # Flatten spatial dimensions
        x = tf.reshape(x, (B, H_new * W_new, self.embed_dim))  # (B, num_patches, embed_dim)

        # Apply normalization
        x = self.norm(x)

        return x, (H_new, W_new)


def create_pvt_small(num_classes=1000, img_size=224):
    """Create PVT-Small model"""
    # Model configuration
    embed_dims = [64, 128, 320, 512]
    num_heads = [1, 2, 5, 8]
    mlp_ratios = [8, 8, 4, 4]
    depths = [3, 4, 6, 3]
    sr_ratios = [8, 4, 2, 1]
    drop_rate = 0.0
    attn_drop_rate = 0.0
    drop_path_rate = 0.1

    # Create functional model instead of using Variables directly
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3), name='input_images')
    x = inputs

    # Stage 1
    patch_embed1 = PatchEmbed(
        img_size=img_size,
        patch_size=4,
        in_chans=3,
        embed_dim=embed_dims[0]
    )
    x, (H, W) = patch_embed1(x)

    # 使用层而不是直接的Variable来添加位置编码
    class PositionalEmbedding(tf.keras.layers.Layer):
        def __init__(self, num_patches, embed_dim, **kwargs):
            super().__init__(**kwargs)
            self.num_patches = num_patches
            self.embed_dim = embed_dim

        def build(self, input_shape):
            self.pos_embed = self.add_weight(
                name='pos_embed',
                shape=(1, self.num_patches, self.embed_dim),
                initializer='truncated_normal',
                trainable=True
            )
            super().build(input_shape)

        def call(self, x):
            return x + self.pos_embed

    # Position embedding for stage 1
    pos_embed1 = PositionalEmbedding((img_size // 4) * (img_size // 4), embed_dims[0])
    x = pos_embed1(x)
    x = tf.keras.layers.Dropout(drop_rate, name='dropout_stage1')(x)

    # Transformer blocks for stage 1
    dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
    cur = 0
    for i in range(depths[0]):
        block = TransformerBlock(
            embed_dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
            sr_ratio=sr_ratios[0], drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i], name=f'transformer_block_stage1_{i}'
        )
        x = block(x, H, W)
    cur += depths[0]

    # Reshape for next stage
    x = tf.reshape(x, (-1, H, W, embed_dims[0]))

    # Stage 2
    patch_embed2 = PatchEmbed(
        img_size=img_size // 4,
        patch_size=2,
        in_chans=embed_dims[0],
        embed_dim=embed_dims[1]
    )
    x, (H, W) = patch_embed2(x)

    pos_embed2 = PositionalEmbedding((img_size // 8) * (img_size // 8), embed_dims[1])
    x = pos_embed2(x)
    x = tf.keras.layers.Dropout(drop_rate, name='dropout_stage2')(x)

    for i in range(depths[1]):
        block = TransformerBlock(
            embed_dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
            sr_ratio=sr_ratios[1], drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i], name=f'transformer_block_stage2_{i}'
        )
        x = block(x, H, W)
    cur += depths[1]

    x = tf.reshape(x, (-1, H, W, embed_dims[1]))

    # Stage 3
    patch_embed3 = PatchEmbed(
        img_size=img_size // 8,
        patch_size=2,
        in_chans=embed_dims[1],
        embed_dim=embed_dims[2]
    )
    x, (H, W) = patch_embed3(x)

    pos_embed3 = PositionalEmbedding((img_size // 16) * (img_size // 16), embed_dims[2])
    x = pos_embed3(x)
    x = tf.keras.layers.Dropout(drop_rate, name='dropout_stage3')(x)

    for i in range(depths[2]):
        block = TransformerBlock(
            embed_dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
            sr_ratio=sr_ratios[2], drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i], name=f'transformer_block_stage3_{i}'
        )
        x = block(x, H, W)
    cur += depths[2]

    x = tf.reshape(x, (-1, H, W, embed_dims[2]))

    # Stage 4
    patch_embed4 = PatchEmbed(
        img_size=img_size // 16,
        patch_size=2,
        in_chans=embed_dims[2],
        embed_dim=embed_dims[3]
    )
    x, (H, W) = patch_embed4(x)

    pos_embed4 = PositionalEmbedding((img_size // 32) * (img_size // 32), embed_dims[3])
    x = pos_embed4(x)
    x = tf.keras.layers.Dropout(drop_rate, name='dropout_stage4')(x)

    for i in range(depths[3]):
        block = TransformerBlock(
            embed_dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
            sr_ratio=sr_ratios[3], drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i], name=f'transformer_block_stage4_{i}'
        )
        x = block(x, H, W)

    # Global average pooling and classification head
    x = tf.keras.layers.LayerNormalization(axis=-1, name='final_norm')(x)
    x = tf.reduce_mean(x, axis=1, name='global_avg_pool')  # Global average pooling

    # Classification head
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='classification_head')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='PVT-Small')
    return model


"""
HYPERPARAMETERS
"""

# Distribute training
strategy = tf.distribute.MirroredStrategy()

# Input
image_size = 224

# Hyper-parameters
batch_size = 4 * strategy.num_replicas_in_sync
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
    pvt_model = create_pvt_small(num_classes=num_classes, img_size=image_size)

pvt_model.summary()

"""
MODEL COMPILE
"""

with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    pvt_model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top-5-accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="top-10-accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=100, name="top-100-accuracy"),
        ]
    )

"""
CALLBACKS
"""

# checkpoint callback
checkpoint_filepath = r"F:\Paper ViT\3_model_train\pvt_celeba\checkpoint"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    save_freq='epoch',
)

# csv logger callback
csv_filepath = r"F:\Paper ViT\3_model_train\pvt_celeba\training_log.csv"
csv_logger = keras.callbacks.CSVLogger(
    csv_filepath,
    separator=',',
    append=True,
)

# early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=0,
    mode='auto',
)

# tensorboard callback
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=r"F:\Paper ViT\3_model_train\pvt_celeba\logs" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S"),
    histogram_freq=1,
    write_graph=True,
    update_freq='epoch',
)

"""
LOAD PRE-TRAINED MODEL WEIGHTS
"""

# Load pre-trained model weights before training (if available)
best_weights = r"F:\Paper ViT\saved_results\Models\ViT_B32\checkpoint"
pvt_model.load_weights(best_weights)


"""
TRAIN THE MODEL
"""

# Train the model
history = pvt_model.fit(
    train_gen,
    epochs=num_epochs,
    validation_data=val_gen,
    callbacks=[
        checkpoint_callback,
        csv_logger,
        early_stopping,
        tb_callback,
    ]
)

"""
EVALUATE THE MODEL
"""

# Load best weights seen during training
pvt_model.load_weights(checkpoint_filepath)

# Evaluate the model
loss, accuracy, top_five_accuracy, top_ten_accuracy, top_hundred_accuracy = pvt_model.evaluate(test_gen)
accuracy = round(accuracy * 100, 2)
top_five_accuracy = round(top_five_accuracy * 100, 2)
top_ten_accuracy = round(top_ten_accuracy * 100, 2)
top_hundred_accuracy = round(top_hundred_accuracy * 100, 2)
print(f"Accuracy on the test set: {accuracy}%.")
print(f"Top 5 Accuracy on the test set: {top_five_accuracy}%.")
print(f"Top 10 Accuracy on the test set: {top_ten_accuracy}%.")
print(f"Top 100 Accuracy on the test set: {top_hundred_accuracy}%.")

"""
HISTORY FIGURES
"""

# Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('PVT model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(r'F:\Paper ViT\3_model_train\pvt_celeba\model accuracy.png')
plt.close()

# Top 5 accuracy
plt.plot(history.history['top-5-accuracy'])
plt.plot(history.history['val_top-5-accuracy'])
plt.title('PVT model top 5 accuracy')
plt.ylabel('top 5 accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(r'F:\Paper ViT\3_model_train\pvt_celeba\model top 5 accuracy.png')
plt.close()

# Top 10 accuracy
plt.plot(history.history['top-10-accuracy'])
plt.plot(history.history['val_top-10-accuracy'])
plt.title('PVT model top 10 accuracy')
plt.ylabel('top 10 accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(r'F:\Paper ViT\3_model_train\pvt_celeba\model top 10 accuracy.png')
plt.close()

# Top 100 accuracy
plt.plot(history.history['top-100-accuracy'])
plt.plot(history.history['val_top-100-accuracy'])
plt.title('PVT model top 100 accuracy')
plt.ylabel('top 100 accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(r'F:\Paper ViT\3_model_train\pvt_celeba\model top 100 accuracy.png')
plt.close()

# Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('PVT model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(r'F:\Paper ViT\3_model_train\pvt_celeba\model loss.png')
plt.close()