# type: ignore
import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input
from scipy.ndimage import gaussian_filter
from PIL import Image

# ==========================================
# 1. 事前学習済みバックボーンによる特徴抽出
# ==========================================

def build_resnet_encoder(input_shape=(224, 224, 3)):
    """
    ImageNet事前学習済みResNet50を特徴抽出器として採用 [2]。
    """
    base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    out1 = base_model.get_layer("conv3_block4_out").output  # 28x28x512
    out2 = base_model.get_layer("conv4_block6_out").output  # 14x14x1024
    
    model = keras.Model(inputs=base_model.input, outputs=[out1, out2], name="ResNet50_Encoder")
    return model


def aggregate_features(feature_maps, target_size=(28, 28), patch_size=3):
    """
    複数スケールの特徴マップを同じ解像度にリサイズ・結合し、空間集約を行います [2]。
    """
    resized_maps = []
    for f_map in feature_maps:
        resized = tf.image.resize(f_map, target_size, method='bilinear')
        resized_maps.append(resized)
    
    concat_features = tf.concat(resized_maps, axis=-1)
    
    aggregated = tf.nn.avg_pool2d(
        concat_features, 
        ksize=patch_size, 
        strides=1, 
        padding='SAME'
    )
    return aggregated


# ==========================================
# 2. ReConPatch 構成要素
# ==========================================

def l2_distance(z):
    diff = tf.expand_dims(z, axis=1) - tf.expand_dims(z, axis=0)
    return tf.reduce_sum(diff ** 2, axis=-1)


class PairwiseSimilarity(layers.Layer):
    def __init__(self, sigma=1.0):
        super(PairwiseSimilarity, self).__init__()
        self.sigma = sigma

    def call(self, z):
        return tf.exp(-l2_distance(z) / self.sigma)


class ContextualSimilarity(layers.Layer):
    def __init__(self, k):
        super(ContextualSimilarity, self).__init__()
        self.k = k

    def call(self, z):
        distances = l2_distance(z)
        kth_nearest = -tf.math.top_k(-distances, k=self.k, sorted=True)[0][:, -1]
        mask = tf.cast(distances <= tf.expand_dims(kth_nearest, axis=-1), tf.float32)

        intersection = tf.matmul(mask, mask, transpose_b=True)
        norm = tf.reduce_sum(mask, axis=-1, keepdims=True)
        similarity_tilde = (intersection / norm) * mask

        k_half = max(1, self.k // 2)
        k_half_nearest = -tf.math.top_k(-distances, k=k_half, sorted=True)[0][:, -1]
        mask_half = tf.cast(distances <= tf.expand_dims(k_half_nearest, axis=-1), tf.float32)

        R = mask_half * tf.transpose(mask_half)
        sum_sim = tf.matmul(R, similarity_tilde)
        r_count = tf.reduce_sum(R, axis=-1, keepdims=True)
        similarity_hat = sum_sim / tf.maximum(r_count, 1e-9)

        return 0.5 * (similarity_hat + tf.transpose(similarity_hat))


class ReConPatchModel(keras.Model):
    def __init__(self, input_dim, embedding_dim, projection_dim, alpha, margin=1.0, gamma=0.9, k_neighbors=5):
        super(ReConPatchModel, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.gamma = gamma

        self.embedding = layers.Dense(embedding_dim)
        self.projection = layers.Dense(projection_dim)
        self.ema_embedding = layers.Dense(embedding_dim, trainable=False)
        self.ema_projection = layers.Dense(projection_dim, trainable=False)

        self.embedding.build((None, input_dim))
        self.projection.build((None, embedding_dim))
        self.ema_embedding.build((None, input_dim))
        self.ema_projection.build((None, embedding_dim))

        self.ema_embedding.set_weights(self.embedding.get_weights())
        self.ema_projection.set_weights(self.projection.get_weights())

        self.pairwise_similarity = PairwiseSimilarity(sigma=1.0)
        self.contextual_similarity = ContextualSimilarity(k=k_neighbors)

    def call(self, x):
        return self.embedding(x)

    def train_step(self, x):
        h_ema = self.ema_embedding(x)
        z_ema = self.ema_projection(h_ema)
        p_sim = self.pairwise_similarity(z_ema)
        c_sim = self.contextual_similarity(z_ema)
        w = self.alpha * p_sim + (1 - self.alpha) * c_sim

        with tf.GradientTape() as tape:
            h = self.embedding(x)
            z = self.projection(h)
            distances = tf.sqrt(l2_distance(z) + 1e-9)
            delta = distances / tf.reduce_mean(distances, axis=-1, keepdims=True)
            rc_loss = tf.reduce_sum(tf.reduce_mean(
                w * (delta ** 2) + (1 - w) * (tf.nn.relu(self.margin - delta) ** 2),
                axis=-1
            ))

        gradients = tape.gradient(rc_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.update_ema()
        return {"rc_loss": rc_loss}

    def update_ema(self):
        train_vars = self.embedding.variables + self.projection.variables
        ema_vars = self.ema_embedding.variables + self.ema_projection.variables
        for ema_var, train_var in zip(ema_vars, train_vars):
            ema_var.assign(self.gamma * ema_var + (1.0 - self.gamma) * train_var)


def greedy_k_center(features, coreset_ratio=0.01):
    """
    💡 進捗表示を追加したコアセットサンプリング [4]
    """
    n = features.shape[0]
    num_centers = max(1, int(n * coreset_ratio))
    centers = [np.random.randint(n)]
    
    # 進捗表示の間隔（10%刻み）
    log_interval = max(1, num_centers // 10)
    
    min_dists = np.sum((features - features[centers[0]])**2, axis=1)
    for i in range(1, num_centers):
        new_center = np.argmax(min_dists)
        centers.append(new_center)
        new_dists = np.sum((features - features[new_center])**2, axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        
        # コアセット選択の進捗を出力
        if i % log_interval == 0 or i == num_centers - 1:
            progress_pct = (i + 1) / num_centers * 100
            print(f"  [コアセット選択進捗] {i+1} / {num_centers} 点選択完了 ({progress_pct:.1f}%)")
            
    return features[centers]


def batch_euclidean_distance(X, Y):
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True).T
    XY = np.dot(X, Y.T)
    dists_sq = np.clip(X_sq - 2*XY + Y_sq, 0, None)
    return np.sqrt(dists_sq + 1e-9)


# ==========================================
# 3. 空間2D対応した ReConPatch 検出器
# ==========================================

class ReConPatchSpatialDetector:
    def __init__(self, input_dim, embedding_dim=512, projection_dim=128, 
                 alpha=0.5, margin=1.0, gamma=0.9, k_neighbors=5, coreset_ratio=0.01):
        self.coreset_ratio = coreset_ratio
        self.model = ReConPatchModel(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            alpha=alpha,
            margin=margin,
            gamma=gamma,
            k_neighbors=k_neighbors
        )
        self.memory_bank = None

    def fit(self, spatial_features, epochs=10, batch_size=64, learning_rate=1e-4):
        B, H, W, C = spatial_features.shape
        flat_features = tf.reshape(spatial_features, (-1, C))
        total_patches = flat_features.shape[0]
        
        # 1エポックあたりのバッチ数を計算
        num_batches = math.ceil(total_patches / batch_size)
        
        print(f"--- 訓練開始 (総パッチ数: {total_patches}, 総バッチ数/エポック: {num_batches}) ---")
        dataset = tf.data.Dataset.from_tensor_slices(flat_features).shuffle(2000).batch(batch_size)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer)

        for epoch in range(epochs):
            epoch_loss, steps = 0.0, 0
            for batch in dataset:
                metrics = self.model.train_step(batch)
                epoch_loss += metrics["rc_loss"].numpy()
                steps += 1
                
                # 💡 500バッチごと、またはエポックの最終バッチ時に経過を出力
                if steps % 500 == 0 or steps == num_batches:
                    current_loss = metrics["rc_loss"].numpy()
                    print(f"  [Epoch {epoch+1}/{epochs}] バッチ: {steps}/{num_batches} - 現在のバッチLoss: {current_loss:.4f}")
                    
            print(f"=> Epoch {epoch+1}/{epochs} 終了 - 平均Loss: {epoch_loss / steps:.4f}\n")

        print("特徴空間のマッピングおよびメモリバンク（コアセット）構築を開始します...")
        mapped_flat = self.model(flat_features).numpy()
        self.memory_bank = greedy_k_center(mapped_flat, coreset_ratio=self.coreset_ratio)
        print(f"メモリバンク構築完了 (登録代表点数: {self.memory_bank.shape[0]})")

    def predict_anomaly_map(self, test_spatial_features, spatial_shape):
        H, W = spatial_shape
        flat_test = tf.reshape(test_spatial_features, (-1, test_spatial_features.shape[-1]))
        mapped_test = self.model(flat_test).numpy()

        dists = batch_euclidean_distance(mapped_test, self.memory_bank)
        patch_scores = np.min(dists, axis=1)

        anomaly_map = patch_scores.reshape((H, W))
        return anomaly_map


# ==========================================
# 4. 画像読み込みと前処理
# ==========================================

def load_img_for_display(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    return np.array(img, dtype=np.float32) / 255.0


def preprocess_for_model(img_array):
    return preprocess_input(img_array * 255.0)


# ==========================================
# 5. メインパイプライン
# ==========================================

def run_pipeline(input_train_dir, input_test_dir, output_dir, image_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)

    extensions = ("**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.PNG", "**/*.JPG", "**/*.JPEG")
    
    train_paths = []
    for ext in extensions:
        train_paths.extend(glob.glob(os.path.join(input_train_dir, ext), recursive=True))
        
    test_paths = []
    for ext in extensions:
        test_paths.extend(glob.glob(os.path.join(input_test_dir, ext), recursive=True))

    train_paths = sorted(list(set(train_paths)))
    test_paths = sorted(list(set(test_paths)))

    if not train_paths or not test_paths:
        raise ValueError(f"画像ファイルが探索されませんでした。パスをご確認ください。")

    print(f"訓練用（正常）画像数: {len(train_paths)}")
    print(f"テスト用画像数: {len(test_paths)}")

    train_images_display = np.array([load_img_for_display(p, image_size) for p in train_paths])
    train_images_model = np.array([preprocess_for_model(img) for img in train_images_display])

    resnet_encoder = build_resnet_encoder(input_shape=(image_size[0], image_size[1], 3))
    
    print("訓練データのResNet50特徴量を抽出中...")
    raw_train_features = resnet_encoder.predict(train_images_model, batch_size=4, verbose=1) # Keras内蔵進捗バーを表示
    spatial_train_features = aggregate_features(raw_train_features, target_size=(28, 28), patch_size=3)
    
    input_dim = spatial_train_features.shape[-1]
    detector = ReConPatchSpatialDetector(
        input_dim=input_dim,
        embedding_dim=256,
        projection_dim=64,
        coreset_ratio=0.01
    )
    detector.fit(spatial_train_features, epochs=5, batch_size=64)

    # 💡 [進捗カウンタ] テスト画像推論の進行状況を見える化
    total_test = len(test_paths)
    print("\n--- テスト画像の異常スコアマップの生成と保存 ---")
    for idx, test_path in enumerate(test_paths):
        display_img = load_img_for_display(test_path, image_size)
        model_img = preprocess_for_model(display_img)
        input_batch = np.expand_dims(model_img, axis=0)
        
        raw_test_features = resnet_encoder.predict(input_batch, verbose=0)
        spatial_test_features = aggregate_features(raw_test_features, target_size=(28, 28), patch_size=3)
        
        anomaly_map_small = detector.predict_anomaly_map(spatial_test_features, (28, 28))
        
        anomaly_map_resized = Image.fromarray(anomaly_map_small).resize(image_size, Image.Resampling.BILINEAR)
        anomaly_map_resized = np.array(anomaly_map_resized)

        # ガウシアン平滑化
        anomaly_map_smoothed = gaussian_filter(anomaly_map_resized, sigma=4)

        # 可視化と保存
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(display_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(display_img)
        axes[1].imshow(anomaly_map_smoothed, cmap='jet', alpha=0.5)
        axes[1].set_title(f"Anomaly Heatmap (Max: {np.max(anomaly_map_smoothed):.2f})")
        axes[1].axis('off')

        rel_path = os.path.relpath(test_path, input_test_dir)
        safe_file_name = "anomaly_" + rel_path.replace(os.sep, "_")
        output_file_path = os.path.join(output_dir, safe_file_name)

        plt.savefig(output_file_path, bbox_inches='tight')
        plt.close()
        
        # 💡 [経過表示] 現在処理中の画像番号 / 全体数を表示
        print(f"  [{idx+1: >3} / {total_test}] 結果を保存しました: {output_file_path}")

    print("\nすべての推論処理と結果の保存が完了しました。")


# ==========================================
# 6. メイン実行部
# ==========================================

if __name__ == "__main__":
    INPUT_TRAIN_DIR = "/home/medicot/ReconPatch/bottle/train/good"
    INPUT_TEST_DIR = "/home/medicot/ReconPatch/bottle/test"
    OUTPUT_DIR = "/home/medicot/ReconPatch/bottle/output_results"

    try:
        run_pipeline(
            input_train_dir=INPUT_TRAIN_DIR,
            input_test_dir=INPUT_TEST_DIR,
            output_dir=OUTPUT_DIR
        )
    except Exception as e:
        print(f"\n[エラー] 実行中に問題が発生しました。画像パス等を確認してください。")
        print(f"エラー詳細: {e}")