# ReConPatch with U-Net Feature Extractor (Industrial Anomaly Detection)

本プロジェクトは、論文 **"ReConPatch: Contrastive Patch Representation Learning for Industrial Anomaly Detection"** [1] の手法に基づき、U-Netのエンコーダからマルチスケールなパッチ特徴量を抽出し、対照学習（Metric Learning）を適用して製品の異常検知および異常個所のセグメンテーション（可視化）を行うシステムの実装である。

---

## 主な特徴

1. **U-Net型マルチスケール特徴抽出:** U-Netのダウンサンプリング層（低レベル・中レベル・高レベルの異なる解像度の特徴マップ）から豊富な表現を抽出し、チャンネル方向に結合してパッチ特徴量として集約する [2, 5]。
2. **Keras 3 & TensorFlow 2.16+ 完全互換:** オプティマイザの勾配計算と更新処理に標準的な `apply_gradients` 方式を採用し、新しいTensorFlow/Keras環境でも安定して動作する。
3. **MVTec AD対応の再帰的画像探索:** テスト用フォルダ（`test/`）配下のサブフォルダ（`broken_large` や `contamination` など）の階層構造を再帰的に探索し、一括で読み込む。
4. **出力画像の上書き防止設計:** テスト画像の相対パス情報からユニークなファイル名（例: `anomaly_broken_large_000.png`）を自動生成し、異なるカテゴリ間で同名ファイルが上書きされるのを防ぐ。
5. **高速コアセット抽出:** 論文で採用されている **Greedy K-Center法** に基づき、大量の正常パッチから代表点をサンプリングして効率的なメモリバンクを構築 [4]。
6. **異常セグメンテーション（ヒートマップ表示）:** テスト画像の各ピクセル（パッチ）に対して異常スコアを計算し、オリジナル画像にヒートマップ（ジェットカラー）として重ね合わせた比較画像を出力する [4, 8]。

---

## フォルダ構成（MVTec ADデータセットの例）

本プログラムを実行する前に、以下のように画像データが配置されていることをご確認（ここでは `bottle` データセットを例としている）。

```text
.
├── ReconPatch.py            # 実行スクリプト
├── requirements.txt         # 依存ライブラリ一覧
├── README.md                # 本ドキュメント（本ファイル）
└── bottle/                  # データセットフォルダ
    ├── train/
    │   └── good/            # 訓練用の正常画像 (.png, .jpg)
    ├── test/
    │   ├── broken_large/    # テスト画像（サブフォルダに分かれていても自動検出されます）
    │   ├── contamination/
    │   └── good/
    └── output_results/      # 異常検出（ヒートマップ）画像の保存先（自動作成されます）
```  
---

## セットアップ（導入手順）

### 1. 仮想環境の作成（推奨）
Python 3.9 〜 3.11 の環境を推奨します。プロジェクトのルートディレクトリで以下のコマンドを実行し、仮想環境を作成・有効化する。

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux の場合
# .venv\Scripts\activate   # Windows の場合
```
### 2. 依存ライブラリのインストール
requirements.txt を用いて、必要なライブラリを一括でインストール。

```bash
pip install -r requirement.txt
```

## 使い方

1. **ディレクトリパスの確認:** ReconPatch.py 末尾の __main__ ブロックにおける入力パス・出力パスが、ご自身の環境のパスと一致しているか確認する。
INPUT_TRAIN_DIR = "/home/medicot/ReconPatch/bottle/train/good"
INPUT_TEST_DIR = "/home/medicot/ReconPatch/bottle/test"
OUTPUT_DIR = "/home/medicot/ReconPatch/bottle/output_results"
2. **スクリプトの実行:**準備ができたら、仮想環境が有効な状態で以下のコマンドを実行し、スクリプトを走らせる。
python ReconPatch.py
3. **結果の確認：** プログラムの実行が完了すると、OUTPUT_DIR（例: bottle/output_results/）フォルダ内に、オリジナル画像と異常箇所をヒートマップで可視化した比較画像が自動的に保存される。保存ファイル名は、同名ファイルによる上書きを防ぐためにサブフォルダ名が統合されます（例: anomaly_broken_large_000.png）。

---
## ハイパーパラメータについて
ReconPatch.py 内の ReConPatchSpatialDetector の初期化時に、以下の主要なハイパーパラメータを調整することができる。

* `alpha`: ペア類似度と文脈類似度の重要度バランス。デフォルトは `0.5`（1:1）。
* `coreset_ratio`: 正常パッチ特徴量をメモリバンクに保存する割合。デフォルトは `0.01` (全体の1%) [5]。
* `k_neighbors`: 文脈類似度を計算する際の近傍数。デフォルトは `5` [11]。
* `margin`: 異なる特徴量を遠ざける際のマージン $m$。デフォルトは `1.0` [4, 11]。

---

## 参考文献

[1] Jeeho Hyun, Sangyun Kim, Giyoung Jeon, Seung Hwan Kim, Kyunghoon Bae, Byung Jun Kang. "ReConPatch: Contrastive Patch Representation Learning for Industrial Anomaly Detection." arXiv:2305.16713, 2023.