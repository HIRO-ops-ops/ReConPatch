# type: ignore
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from PIL import Image

# ==========================================
# 1. U-Netエンコーダによるマルチスケール特徴抽出
# ==========================================

def build_unet_encoder(input_shape=(224, 224, 3)):
    """
    U-Netのエンコーダ（ダウンサンプリング層）を定義。
    低レベル・中レベル・高レベルのマルチスケール特徴マップを出力します。
    """
    inputs = layers.Input(shape=input_shape)
    
    # Block 1 (224x224)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1) # 112x112
    
    # Block 2 (112x112)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2) # 56x56
    
    # Block 3 (56x56)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3) # 28x28
    
    # 浅いレイヤー(c1), 中間(c2), 深いレイヤー(c3)の特徴を出力
    # ※実運用では学習済み重み（Segmentation Models等）をロードすることを推奨します
    model = keras.Model(inputs=inputs, outputs=[c1, c2, c3], name="UNet_Encoder")
    return model


def aggregate_features(feature_maps, target_size=(56, 56), patch_size=3):
    """
    論文3.1節に基づき、複数スケールの特徴マップを同じ解像度にリサイズ・結合し、
    指定パッチサイズで近傍平均プーリング（空間的な集約）を行います [2]。
    """
    resized_maps = []
    for f_map in feature_maps:
        # すべての特徴マップを指定サイズ（例: 56x56）にリサイズ
        resized = tf.image.resize(f_map, target_size, method='bilinear')
        resized_maps.append(resized)
    
    # チャンネル方向に結合
    concat_features = tf.concat(resized_maps, axis=-1) # (B, H, W, Total_C)
    
    # 近傍平均プーリングによる局所的な特徴の平滑化
    aggregated = tf.nn.avg_pool2d(
        concat_features, 
        ksize=patch_size, 
        strides=1, 
        padding='SAME'
    )
    return aggregated


# ==========================================
# 2. 前ターンで作成した ReConPatch 構成要素
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

        self.optimizer.minimize(rc_loss, self.trainable_variables, tape=tape)
        self.update_ema()
        return {"rc_loss": rc_loss}

    def update_ema(self):
        train_vars = self.embedding.variables + self.projection.variables
        ema_vars = self.ema_embedding.variables + self.ema_projection.variables
        for ema_var, train_var in zip(ema_vars, train_vars):
            ema_var.assign(self.gamma * ema_var + (1.0 - self.gamma) * train_var)


def greedy_k_center(features, coreset_ratio=0.01):
    n = features.shape[0]
    num_centers = max(1, int(n * coreset_ratio))
    centers = [np.random.randint(n)]
    min_dists = np.sum((features - features[centers[0]])**2, axis=1)
    for _ in range(1, num_centers):
        new_center = np.argmax(min_dists)
        centers.append(new_center)
        new_dists = np.sum((features - features[new_center])**2, axis=1)
        min_dists = np.minimum(min_dists, new_dists)
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
        """
        spatial_features: (B, H, W, C) のテンソルを受け取ってパッチ単位で学習
        """
        B, H, W, C = spatial_features.shape
        # (B * H * W, C) にフラット化して対照学習を行う
        flat_features = tf.reshape(spatial_features, (-1, C))
        
        print(f"--- 訓練開始 (総パッチ数: {flat_features.shape[0]}) ---")
        dataset = tf.data.Dataset.from_tensor_slices(flat_features).shuffle(2000).batch(batch_size)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer)

        for epoch in range(epochs):
            epoch_loss, steps = 0.0, 0
            for batch in dataset:
                metrics = self.model.train_step(batch)
                epoch_loss += metrics["rc_loss"].numpy()
                steps += 1
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / steps:.4f}")

        # コアセットメモリバンクの構築
        mapped_flat = self.model(flat_features).numpy()
        print("メモリバンク（コアセット）構築中...")
        self.memory_bank = greedy_k_center(mapped_flat, coreset_ratio=self.coreset_ratio)
        print(f"メモリバンク構築完了 (登録数: {self.memory_bank.shape[0]})")

    def predict_anomaly_map(self, test_spatial_features, spatial_shape):
        """
        2Dの各ピクセル（パッチ）に対して異常スコアを計算し、アノマリーマップを出力。
        test_spatial_features: (1, H, W, C)
        """
        H, W = spatial_shape
        flat_test = tf.reshape(test_spatial_features, (-1, test_spatial_features.shape[-1]))
        mapped_test = self.model(flat_test).numpy()

        # 最も近いコアセット点までの距離を計算 (Eq. 9)
        dists = batch_euclidean_distance(mapped_test, self.memory_bank)
        patch_scores = np.min(dists, axis=1) # (H * W,)

        # (H, W) のグリッド形状にリライト
        anomaly_map = patch_scores.reshape((H, W))
        return anomaly_map


# ==========================================
# 4. メインパイプライン（パス入出力とヒートマップ出力）
# ==========================================

def load_and_preprocess_img(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    x = np.array(img, dtype=np.float32) / 255.0
    return x


def run_pipeline(input_train_dir, input_test_dir, output_dir, image_size=(224, 224)):
    """
    パスを指定してU-Net特徴抽出 -> ReConPatch学習 -> テスト推論 -> 結果画像保存
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 画像のロード
    train_paths = glob.glob(os.path.join(input_train_dir, "*.png")) + glob.glob(os.path.join(input_train_dir, "*.jpg"))
    test_paths = glob.glob(os.path.join(input_test_dir, "*.png")) + glob.glob(os.path.join(input_test_dir, "*.jpg"))

    if not train_paths or not test_paths:
        raise ValueError("指定されたディレクトリに画像ファイル（jpg/png）が見つかりません。")

    print(f"訓練用（正常）画像数: {len(train_paths)}")
    print(f"テスト用画像数: {len(test_paths)}")

    train_images = np.array([load_and_preprocess_img(p, image_size) for p in train_paths])

    # 2. U-Netエンコーダによる特徴抽出
    unet_encoder = build_unet_encoder(input_shape=(image_size[0], image_size[1], 3))
    
    # 訓練用データの特徴量をU-Netから得る
    print("訓練データのU-Net特徴量を抽出中...")
    raw_train_features = unet_encoder.predict(train_images, batch_size=4)
    # パッチ特徴量への集約 (ターゲット解像度: 56x56, チャンネル合計数: 64+128+256=448)
    spatial_train_features = aggregate_features(raw_train_features, target_size=(56, 56), patch_size=3)
    
    # 3. 検出器の初期化と訓練
    input_dim = spatial_train_features.shape[-1]
    detector = ReConPatchSpatialDetector(
        input_dim=input_dim,
        embedding_dim=256,
        projection_dim=64,
        coreset_ratio=0.01  # メモリ容量1%に圧縮保存
    )
    detector.fit(spatial_train_features, epochs=5, batch_size=64)

    # 4. テスト画像に対する推論とアノマリーマップの保存
    print("\n--- テスト画像の異常スコアマップの生成と保存 ---")
    for test_path in test_paths:
        orig_img = load_and_preprocess_img(test_path, image_size)
        # バッチ次元を追加
        input_batch = np.expand_dims(orig_img, axis=0)
        
        # U-Netから特徴抽出と集約
        raw_test_features = unet_encoder.predict(input_batch, verbose=0)
        spatial_test_features = aggregate_features(raw_test_features, target_size=(56, 56), patch_size=3)
        
        # アノマリーマップ（異常確率マップ 56x56）を予測
        anomaly_map_small = detector.predict_anomaly_map(spatial_test_features, (56, 56))
        
        # 可視化のために元の画像サイズ (224x224) にリプレイス
        anomaly_map_resized = Image.fromarray(anomaly_map_small).resize(image_size, Image.Resampling.BILINEAR)
        anomaly_map_resized = np.array(anomaly_map_resized)

        # Matplotlibを用いてオリジナル画像にヒートマップをオーバーレイ表示して保存 [8]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(orig_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # ヒートマップ表示
        im = axes[1].imshow(orig_img)
        # jetカラーマップで異常値をオーバーレイ
        axes[1].imshow(anomaly_map_resized, cmap='jet', alpha=0.5)
        axes[1].set_title(f"Anomaly Heatmap (Max: {np.max(anomaly_map_resized):.2f})")
        axes[1].axis('off')

        # ファイル名を作成して保存
        base_name = os.path.basename(test_path)
        output_file_path = os.path.join(output_dir, f"anomaly_{base_name}")
        plt.savefig(output_file_path, bbox_inches='tight')
        plt.close()
        print(f"結果を保存しました: {output_file_path}")

    print("\nすべての推論処理と結果の保存が完了しました。")


# ==========================================
# 5. メイン実行部
# ==========================================

if __name__ == "__main__":
    # 実際のディレクトリパスを入力・出力に指定して実行します。
    # ※ 事前にディレクトリが存在し、中に画像があるかご確認ください。
    INPUT_TRAIN_DIR = "/home/medicot/ReconPatch/bottle/train/good"   # 正常画像フォルダ
    INPUT_TEST_DIR = "/home/medicot/ReconPatch/bottle/test"            # テスト画像フォルダ
    OUTPUT_DIR = "/home/medicot/ReconPatch/bottle/output_results"      # ヒートマップ結果保存先

    # 実行する際は、ダミーディレクトリでテストするか、実際のデータパスに書き換えてください。
    try:
        run_pipeline(
            input_train_dir=INPUT_TRAIN_DIR,
            input_test_dir=INPUT_TEST_DIR,
            output_dir=OUTPUT_DIR
        )
    except Exception as e:
        print(f"\n[エラー] 実行中に問題が発生しました。画像パス等を確認してください。")
        print(f"エラー詳細: {e}")