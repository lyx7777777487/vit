import datetime
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from data_generator import create_data_generators
import keras

"""
PROGRESSIVE REPARAMETERIZED BATCHNORM (PRepBN)
Implementation to replace LayerNorm for inference latency reduction
"""


class PRepBN(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, momentum=0.9, center=True, scale=True, **kwargs):
        super(PRepBN, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.momentum = momentum
        self.center = center
        self.scale = scale
        self.training_mode = True  # Flag to switch between training (BN+LN) and inference (folded BN)

    def build(self, input_shape):
        dim = input_shape[-1]

        # BatchNorm parameters
        self.bn_moving_mean = self.add_weight(
            name="bn_moving_mean",
            shape=(dim,),
            initializer="zeros",
            trainable=False,
        )
        self.bn_moving_variance = self.add_weight(
            name="bn_moving_variance",
            shape=(dim,),
            initializer="ones",
            trainable=False,
        )

        # LayerNorm parameters
        if self.scale:
            self.ln_gamma = self.add_weight(
                name="ln_gamma",
                shape=(dim,),
                initializer="ones",
                trainable=True,
            )
        else:
            self.ln_gamma = None

        if self.center:
            self.ln_beta = self.add_weight(
                name="ln_beta",
                shape=(dim,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.ln_beta = None

        # Folded parameters for inference (initialized but will be calculated during reparameterization)
        self.folded_gamma = self.add_weight(
            name="folded_gamma",
            shape=(dim,),
            initializer="ones",
            trainable=False,
        )
        self.folded_beta = self.add_weight(
            name="folded_beta",
            shape=(dim,),
            initializer="zeros",
            trainable=False,
        )

        self.built = True

    def call(self, inputs, training=None):
        # Determine if we're in training mode based on the passed training flag or the layer's training attribute
        is_training = training if training is not None else self.training_mode

        if is_training:
            # During training - use combined BatchNorm + LayerNorm
            # First apply BatchNorm
            input_shape = tf.shape(inputs)
            batch_size = input_shape[0]
            sequence_length = input_shape[1]
            dim = input_shape[2]

            # Reshape to [batch_size*sequence_length, dim] for BatchNorm
            reshaped_inputs = tf.reshape(inputs, [-1, dim])

            # Calculate batch statistics
            batch_mean, batch_variance = tf.nn.moments(reshaped_inputs, axes=[0])

            # Update moving statistics
            self.bn_moving_mean.assign(
                self.bn_moving_mean * self.momentum + batch_mean * (1 - self.momentum)
            )
            self.bn_moving_variance.assign(
                self.bn_moving_variance * self.momentum + batch_variance * (1 - self.momentum)
            )

            # Normalize with batch statistics
            bn_output = (reshaped_inputs - batch_mean) / tf.sqrt(batch_variance + self.epsilon)
            bn_output = tf.reshape(bn_output, input_shape)  # Reshape back

            # Then apply LayerNorm
            ln_mean, ln_variance = tf.nn.moments(bn_output, axes=[-1], keepdims=True)
            ln_output = (bn_output - ln_mean) / tf.sqrt(ln_variance + self.epsilon)

            # Apply scale and shift
            if self.ln_gamma is not None:
                ln_output *= self.ln_gamma
            if self.ln_beta is not None:
                ln_output += self.ln_beta

            return ln_output
        else:
            # During inference - use folded parameters
            return inputs * self.folded_gamma + self.folded_beta

    def reparameterize(self):
        """
        Fold BatchNorm into LayerNorm parameters for inference
        This should be called after training and before saving inference model
        """
        # Calculate folded parameters
        # folded_gamma = ln_gamma / sqrt(bn_moving_variance + epsilon)
        # folded_beta = ln_beta - (ln_gamma * bn_moving_mean / sqrt(bn_moving_variance + epsilon))

        stddev = tf.sqrt(self.bn_moving_variance + self.epsilon)

        if self.ln_gamma is not None:
            self.folded_gamma.assign(self.ln_gamma / stddev)
        else:
            self.folded_gamma.assign(1.0 / stddev)

        if self.ln_beta is not None:
            if self.ln_gamma is not None:
                self.folded_beta.assign(self.ln_beta - (self.ln_gamma * self.bn_moving_mean / stddev))
            else:
                self.folded_beta.assign(self.ln_beta - (self.bn_moving_mean / stddev))
        else:
            if self.ln_gamma is not None:
                self.folded_beta.assign(-(self.ln_gamma * self.bn_moving_mean / stddev))
            else:
                self.folded_beta.assign(-(self.bn_moving_mean / stddev))

        # Set training mode to False to use folded parameters during inference
        self.training_mode = False

    def get_config(self):
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "center": self.center,
            "scale": self.scale,
        })
        return config


"""
CUSTOM SIMPLIFIED LINEAR ATTENTION LAYER
Converting PyTorch implementation to TensorFlow
"""


class SimplifiedLinearAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., focusing_factor=3, kernel_size=5, **kwargs):
        super(SimplifiedLinearAttention, self).__init__(**kwargs)

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.focusing_factor = focusing_factor

        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)

        self.dwc = tf.keras.layers.Conv1D(
            filters=self.head_dim,
            kernel_size=kernel_size,
            padding='same',
            groups=self.head_dim,
            use_bias=False
        )

        # 创建一个灵活的位置编码，根据输入序列长度调整
        self.positional_encoding = self.add_weight(
            name="positional_encoding",
            shape=(1, 1, self.dim),  # 简化为每个维度一个位置编码值
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

    def call(self, x, mask=None, training=None):
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]  # 动态获取序列长度

        # QKV 投影
        qkv = self.qkv(x)  # [B, N, 3*dim]
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 动态创建适合当前序列长度的位置编码
        k = tf.reshape(k, [B, self.num_heads, N, self.head_dim])

        # 创建适合当前序列长度的位置编码
        pos_encoding = tf.tile(self.positional_encoding, [1, N, 1])  # [1, N, dim]
        pos_encoding = tf.reshape(pos_encoding, [1, 1, N, self.dim])

        # 每个头部只获取对应的位置编码部分
        pos_encoding_per_head = tf.reshape(
            pos_encoding,
            [1, self.num_heads, N, self.head_dim]
        )

        # 应用位置编码
        k = k + pos_encoding_per_head

        # 应用ReLU激活
        q = tf.nn.relu(q)
        k = tf.nn.relu(k)

        # 线性注意力计算
        k_sum = tf.reduce_sum(k, axis=2, keepdims=True)  # [B, num_heads, 1, head_dim]
        z_inv = 1.0 / (tf.reduce_sum(q * k_sum, axis=-1, keepdims=True) + 1e-6)

        # 计算注意力
        kv = tf.matmul(k, v, transpose_a=True)  # [B, num_heads, head_dim, head_dim]
        attn_output = tf.matmul(q, kv)  # [B, num_heads, N, head_dim]
        attn_output = attn_output * z_inv

        # 局部特征增强
        v_local = tf.reshape(v, [B * self.num_heads, N, self.head_dim])
        v_local = self.dwc(v_local)
        v_local = tf.reshape(v_local, [B, self.num_heads, N, self.head_dim])

        # 合并特征
        x = attn_output + v_local

        # 重塑和投影
        x = tf.transpose(x, [0, 2, 1, 3])  # [B, N, num_heads, head_dim]
        x = tf.reshape(x, [B, N, self.dim])  # [B, N, dim]

        # 最终投影
        x = self.proj(x)
        x = self.proj_drop(x, training=training)

        return x


class LinearAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, window_size=(7, 7), mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.keras.activations.gelu,
                 norm_layer=PRepBN, **kwargs):  # Using PRepBN as default norm_layer
        super(LinearAttentionBlock, self).__init__(**kwargs)

        self.norm1 = norm_layer(epsilon=1e-6)
        self.attn = SimplifiedLinearAttention(
            dim=dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = tf.keras.layers.Dropout(drop_path) if drop_path > 0. else tf.keras.layers.Layer()

        self.norm2 = norm_layer(epsilon=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(hidden_dim=mlp_hidden_dim, out_dim=dim, drop=drop, act_layer=act_layer)

    def call(self, x, training=None):
        shortcut = x
        x = self.norm1(x, training=training)
        x = self.attn(x, training=training)
        x = shortcut + self.drop_path(x, training=training)

        shortcut = x
        x = self.norm2(x, training=training)
        x = self.mlp(x, training=training)
        x = shortcut + self.drop_path(x, training=training)

        return x


class MLP(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, out_dim, drop=0., act_layer=tf.keras.activations.gelu, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(hidden_dim)
        self.act = act_layer
        self.drop1 = tf.keras.layers.Dropout(drop)
        self.fc2 = tf.keras.layers.Dense(out_dim)
        self.drop2 = tf.keras.layers.Dropout(drop)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x


class ClassToken(tf.keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            name='cls_token'
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_tokens, inputs], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


class AddPositionEmbs(tf.keras.layers.Layer):
    def __init__(self, num_patches_plus_one, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches_plus_one = num_patches_plus_one
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            shape=(1, self.num_patches_plus_one, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name='pos_embed'
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.pos_embed

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches_plus_one": self.num_patches_plus_one,
            "embed_dim": self.embed_dim
        })
        return config


def create_vit_linear_attention(
        image_size=224,
        patch_size=16,
        num_classes=8631,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=PRepBN,  # Using PRepBN as default
        act_layer=tf.keras.activations.gelu,
        **kwargs
):
    num_patches = (image_size // patch_size) * (image_size // patch_size)
    window_size = (image_size // patch_size, image_size // patch_size)

    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))

    # Patch embedding
    x = tf.keras.layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patch_embed"
    )(inputs)

    # Reshape patches to sequence
    x = tf.keras.layers.Reshape((-1, embed_dim))(x)

    # Add cls token using the custom layer
    x = ClassToken(embed_dim=embed_dim, name="class_token_layer")(x)

    # Add position embedding using the custom layer
    x = AddPositionEmbs(num_patches_plus_one=num_patches + 1, embed_dim=embed_dim, name="pos_embed_layer")(x)

    # Dropout
    x = tf.keras.layers.Dropout(drop_rate)(x)

    # Attention blocks
    for i in range(depth):
        x = LinearAttentionBlock(
            dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate * i / depth,
            norm_layer=norm_layer,
            act_layer=act_layer,
            name=f"linear_attention_block_{i}"
        )(x)

    # Final norm layer
    x = norm_layer(epsilon=1e-6, name="final_norm")(x)

    # Extract CLS token
    x = tf.keras.layers.Lambda(lambda v: v[:, 0], name="extract_cls_token")(x)

    # Classification head
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="head")(x)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name="vit_linear_attention")


# Custom callback to perform reparameterization
class ReparameterizationCallback(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        """Reparameterize all PRepBN layers after training"""
        for layer in self.model.layers:
            # Recursively search for PRepBN layers
            self._reparameterize_layer(layer)
        print("Model reparameterization complete - BatchNorm folded into parameters for inference")

    def _reparameterize_layer(self, layer):
        """Helper function to recursively find and reparameterize PRepBN layers"""
        if isinstance(layer, PRepBN):
            layer.reparameterize()
        elif hasattr(layer, 'layers'):
            # For container layers like Model, Sequential
            for sublayer in layer.layers:
                self._reparameterize_layer(sublayer)


# 其余代码（训练、评估等）
if __name__ == "__main__":
    """
    HYPERPARAMETERS
    """

    # Distribute training
    strategy = tf.distribute.MirroredStrategy()

    # Input
    image_size = 224

    # Hyper-parameters
    batch_size = 8 * strategy.num_replicas_in_sync
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
        # Create custom ViT model with Linear Attention and PRepBN
        vit_model = create_vit_linear_attention(
            image_size=224,
            patch_size=32,
            num_classes=num_classes,
            embed_dim=768,
            depth=8,
            num_heads=8,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.1,
            norm_layer=PRepBN  # Using PRepBN instead of LayerNorm
        )

    vit_model.summary()

    """
    MODEL COMPILE
    """
    # 启用混合精度
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        vit_model.compile(
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
    checkpoint_filepath = r"F:\Paper ViT\vit_attention\LFW_results\1\checkpoint"
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
    csv_filepath = r"F:\Paper ViT\vit_attention\LFW_results\1\training_log.csv"
    csv_logger = keras.callbacks.CSVLogger(
        csv_filepath,
        separator=',',
        append=True,
    )

    # early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=15,
        verbose=0,
        mode='auto',
    )

    # tensorboard callback
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir="./tmp/logs" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S"),
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch',
    )

    # Add reparameterization callback
    reparam_callback = ReparameterizationCallback()

    """
    LOAD PRE-TRAINED MODEL WEIGHTS
    """

    # Load pre-trained model weights before training
    best_weights = r"F:\Paper ViT\3_model_train\vit\checkpoint"
    vit_model.load_weights(best_weights)

    """
    TRAIN THE MODEL
    """

    # Train the model
    history = vit_model.fit(
        train_gen,
        epochs=num_epochs,
        validation_data=val_gen,
        callbacks=[
            checkpoint_callback,
            csv_logger,
            early_stopping,
            tb_callback,
            reparam_callback,  # Add the reparameterization callback
        ]
    )

    """
    SAVE INFERENCE-OPTIMIZED MODEL
    """
    # Save a separate model for inference after reparameterization
    inference_model_path = r"F:\Paper ViT\vit_attention\LFW_results\1\inference_model"
    vit_model.save(inference_model_path)
    print(f"Saved inference-optimized model to {inference_model_path}")

    """
    EVALUATE THE MODEL
    """

    # Load best weights seen during training
    vit_model.load_weights(checkpoint_filepath)

    # Evaluate the model
    loss, accuracy, top_five_accuracy, top_ten_accuracy, top_hundred_accuracy = vit_model.evaluate(test_gen)
    accuracy = round(accuracy * 100, 2)
    top_five_accuracy = round(top_five_accuracy * 100, 2)
    top_ten_accuracy = round(top_ten_accuracy * 100, 2)
    top_hundred_accuracy = round(top_hundred_accuracy * 100, 2)
    print(f"Accuracy on the test set: {accuracy}%.")
    print(f"Top 5 Accuracy on the test set: {top_five_accuracy}%.")
    print(f"Top 10 Accuracy on the test set: {top_ten_accuracy}%.")
    print(f"Top 100 Accuracy on the test set: {top_hundred_accuracy}%.")

    """
    LATENCY BENCHMARKING
    """
    # Benchmark inference latency comparison between LayerNorm and PRepBN
    print("Benchmarking inference latency...")

    # Create dummy input
    dummy_input = tf.random.normal((1, image_size, image_size, 3))

    # Warm-up runs
    for _ in range(10):
        _ = vit_model(dummy_input, training=False)

    # Measure inference time
    import time

    num_runs = 100

    start_time = time.time()
    for _ in range(num_runs):
        _ = vit_model(dummy_input, training=False)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
    print(f"Average inference time: {avg_inference_time:.2f} ms")

    """
    HISTORY FIGURES
    """

    # Accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./tmp/model accuracy.png')
    plt.close()
    # Top 5 accuracy
    plt.plot(history.history['top-5-accuracy'])
    plt.plot(history.history['val_top-5-accuracy'])
    plt.title('model top 5 accuracy')
    plt.ylabel('top 5 accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./tmp/model top 5 accuracy.png')
    plt.close()
    # Top 10 accuracy
    plt.plot(history.history['top-10-accuracy'])
    plt.plot(history.history['val_top-10-accuracy'])
    plt.title('model top 10 accuracy')
    plt.ylabel('top 10 accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./tmp/model top 10 accuracy.png')
    plt.close()
    # Top 100 accuracy
    plt.plot(history.history['top-100-accuracy'])
    plt.plot(history.history['val_top-100-accuracy'])
    plt.title('model top 100 accuracy')
    plt.ylabel('top 100 accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./tmp/model top 100 accuracy.png')
    plt.close()
    # Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./tmp/model loss.png')
    plt.close()