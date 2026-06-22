実装したプログラムを第三者（または将来の自分）がスムーズに使用・管理できるように、**README.md（設定・操作説明書）**を作成しました。

プロジェクトの概要、特徴、フォルダ構成、セットアップ、および実行方法を分かりやすくまとめています。この内容をそのまま README.md
というファイル名でプロジェクトのルートディレクトリに保存してご活用ください。

README.md

# ReConPatch with U-Net Feature Extractor (Industrial Anomaly Detection)

本プロジェクトは、論文 **"ReConPatch: Contrastive Patch Representation Learning for Industrial Anomaly Detection"** [1] の手法に基づき、U-Netのエンコーダからマルチスケールなパッチ特徴量を抽出し、対照学習（Metric Learning）を適用して製品の異常検知および異常個所のセグメンテーション（可視化）を行うシステムの実装です。

---

## 📌 主な特徴

1. **U-Net型マルチスケール特徴抽出:** U-Netのダウンサンプリング層（低レベル・中レベル・高レベルの異なる解像度の特徴マップ）から豊富な表現を抽出し、チャンネル方向に結合してパッチ特徴量として集約します [2, 5]。
2. **ReConPatch対照学習の再現:** 正常データのみの状況下で、ペア類似度（Pairwise）と文脈類似度（Contextual）から構築した「擬似ラベル」を基に、線形変換によるターゲットドメインへの最適化（Metric Learning）を行います [3, 4]。
3. **高速コアセット抽出:** 論文で採用されている **Greedy K-Center法** に基づき、大量の正常パッチから代表点をサンプリングして効率的なメモリバンクを構築します [4]。
4. **異常セグメンテーション（ヒートマップ表示）:** テスト画像の各ピクセル（パッチ）に対して異常スコアを計算し、オリジナル画像にヒートマップ（ジェットカラー）として重ね合わせた画像を出力します [4, 8]。

---

## 📁 フォルダ構成

本プログラムを実行する前に、以下のように画像データを配置してください。

```text
.
├── main.py                  # メインの実行スクリプト
├── requirements.txt         # 依存ライブラリ一覧
├── README.md                # 本ドキュメント
└── data/
    ├── train/
    │   └── normal/          # 訓練用の「正常な状態の画像」のみを配置 (.jpg, .png)
    ├── test/                # テスト用の画像（正常・異常が混在してOK）
    └── output_results/      # 異常検出（ヒートマップ）画像の保存先（自動作成されます）

🛠️ セットアップ（導入手順）

1. 仮想環境の作成（推奨）

Python 3.9 〜 3.11 の環境を推奨します。

python -m venv venv
source venv/bin/activate  # macOS / Linux の場合
# venv\Scripts\activate   # Windows の場合

2. 依存ライブラリのインストール

リポジトリ直下の requirements.txt を用いて、必要なライブラリを一括でインストールします。

pip install -r requirements.txt

🚀 使い方

1. データの準備

data/train/normal/ フォルダに正常な画像を数枚〜数十枚配置します。 data/test/ フォルダに推論を行いたいテスト画像を配置します。

2. 実行

準備ができたら、以下のコマンドでメインスクリプトを実行します。

python main.py

3. 結果の確認

プログラムの実行が完了すると、data/output_results/ フォルダ内に、オリジナル画像と異常箇所をヒートマップで可視化した画像（例:
anomaly_test_image.png）が自動的に保存されます。

⚙️ ハイパーパラメータについて

main.py 内の ReConPatchSpatialDetector の初期化時に、以下の主要なハイパーパラメータを調整することができます。

  - alpha: ペア類似度と文脈類似度の重要度バランス。デフォルトは 0.5（1:1）。
  - coreset_ratio: 正常パッチ特徴量をメモリバンクに保存する割合。デフォルトは 0.01 (全体の1%) [5]。
  - k_neighbors: 文脈類似度を計算する際の近傍数。デフォルトは 5 [11]。
  - margin: 異なる特徴量を遠ざける際のマージン m。デフォルトは 1.0 [4, 11]。

📄 参考文献

[1] Jeeho Hyun, Sangyun Kim, Giyoung Jeon, Seung Hwan Kim, Kyunghoon Bae, Byung
Jun Kang. "ReConPatch: Contrastive Patch Representation Learning for Industrial
Anomaly Detection." arXiv:2305.16713, 2023.

