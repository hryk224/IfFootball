# IfFootball

[English version](README.md)

> _「もしクロップがチェルシーの監督だったら？」_

別の監督やスカッド変更がサッカーチームのシーズン全体にどう影響したかをシミュレートします。StatsBomb Open Data、ルールベースの因果シミュレーション、オプションの LLM 分析レポートを使用。**API キー不要で動作 — LLM はオプションです。**

![IfFootball デモ - プリセットとサマリー](docs/images/demo-overview.png)
![IfFootball デモ - 選手インパクトと詳細分析](docs/images/demo-detailed-analysis.png)

**[Live Demo](https://iffootball-bfmlfowrz7fxf8j4fkuuqv.streamlit.app/)** — セットアップ不要でブラウザから試せます。

## クイックスタート

```bash
git clone https://github.com/hryk224/IfFootball.git
cd IfFootball
uv sync --extra dev
uv run python scripts/build_season_cache.py   # 初回のみ（約5分）
uv run streamlit run app.py
```

チームを選択し、シナリオ種別（監督交代 / 選手追加 / 選手除外）を選び、ドロップダウンから候補を選択して **Run Scenario** をクリック。候補は事前構築した season cache からのみ選択可能で、実行時の API 呼び出しはありません。

`.env` ファイル、API キー、LLM のセットアップなしで基本機能が使えます。

## 出力例

シーズンシナリオ: チェルシーの監督がモウリーニョではなくクロップだったら（プレミアリーグ 2015-16、5 回シミュレーション）:

```
Branch A（モウリーニョ）:  64.6 ポイント（平均）
Branch B（クロップ）:      56.0 ポイント（平均）
差分 (B - A):              -8.6 ポイント
```

38 試合のフルシーズンを各監督の下で比較します。戦術プロファイルの違い（プレッシング強度、ポゼッション傾向、フォーメーション）がラインナップ選択と適応ダイナミクスを駆動します。Streamlit UI では選手影響レーダーチャートと構造化レポートを含む完全な分析が表示されます。

## 免責事項

IfFootball は **what-if シミュレーションツール**であり、予測エンジンではありません。結果はルールベースのパラメータに基づく理論的な結果であり、実際のイベントの予測ではありません。すべてのシミュレーションパラメータは暫定値であり、根拠とともに [simulation-rules.md](docs/simulation-rules.md#known-limitations) に記載されています。本ソフトウェアまたはその出力に基づいて行われたいかなる判断についても、作者は責任を負いません。

## 主な機能

- **監督交代** — 別の監督でフルシーズンをシミュレート。戦術プロファイルの違い、選手の信頼度再調整、適応曲線を反映
- **選手追加** — 他チームの選手を役割に応じた信頼度初期化でスカッドに追加
- **選手除外** — 特定の選手なしでシーズンをシミュレート
- **A/B 比較** — Poisson 試合モデル、週次状態更新（疲労・信頼度・戦術理解度）、ターニングポイント検出とカスケードイベント追跡
- **可視化** — Branch A/B を比較するチーム・選手レーダーチャート
- **LLM レポート** — データ / 分析 / 仮説のラベル付き構造化比較レポート
- **Streamlit UI** — 入力からシミュレーション、出力までを一気通貫で操作できるシングルページアプリ

### 動作フロー

```
[初回のみ] build_season_cache.py
    StatsBomb データ --> Season Cache DB（全チーム、フルシーズン）

[実行時] Streamlit UI
    チーム + シナリオ選択（season cache から候補表示）
        |
        v
    シミュレーションエンジン（N 回 x 2 ブランチ、各 38 試合）
        |--- Branch A: ベースライン（実際の監督 / スカッド）
        |--- Branch B: シナリオ適用（別監督 / +選手 / -選手）
        |
        v
    比較 & 可視化（レーダーチャート、カスケードイベント、レポート）
```

## データソース

IfFootball は [StatsBomb Open Data](https://github.com/statsbomb/open-data) のみを使用します。すべての指標と用語は StatsBomb の定義に準拠しています。スクレイピングは一切行いません。帰属表示と利用条件は [StatsBomb Open Data ライセンス](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf)に従います。

現在対応しているリーグ:

| コンペティション | シーズン | チーム                                          |
| ---------------- | -------- | ----------------------------------------------- |
| プレミアリーグ   | 2015-16  | 全 20 チーム（season cache にリーグ全体を収録） |

## 詳細セットアップ

上記のクイックスタートで最小構成は動作します。ここでは開発ツールと LLM の設定を追加します。

```bash
git clone https://github.com/hryk224/IfFootball.git
cd IfFootball
cp .env.example .env
uv sync --extra dev
npm install                    # Markdown フォーマット用（任意）
```

**必要環境:** Python >= 3.11, [uv](https://docs.astral.sh/uv/), Node.js（任意、`npm run format:md` 用のみ）

### LLM セットアップ（任意）

LLM プロバイダー SDK は通常依存に含まれています。LLM によるレポート生成を有効にするには、`.env` に API キーを設定します:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
GROQ_API_KEY=gsk_...
```

対応プロバイダー: OpenAI, Anthropic, Google Gemini, Groq。LLM 未設定時はデータのみモードで動作し、シミュレーションデータから構造化レポートを生成します。

**モデル指定:** 各プロバイダーに個別のモデル環境変数があります（`OPENAI_MODEL`, `ANTHROPIC_MODEL`, `GEMINI_MODEL`, `GROQ_MODEL`）。未設定時はプロバイダーのデフォルトモデルが使用されます。

**OpenAI 互換 API:** `OPENAI_BASE_URL` を設定すると OpenAI 互換エンドポイント（Azure OpenAI、ローカル推論サーバーなど）を使用できます。

**注意:** LLM 有効時は、シナリオデータ（チーム名、選手名、シミュレーション結果）が設定されたプロバイダーに送信されます。データの取り扱いは各プロバイダーのポリシーに従います。

## 使い方

### Streamlit UI

```bash
uv run streamlit run app.py
```

チームとシナリオを選択して「Run Scenario」をクリック。事前に season cache の構築（`scripts/build_season_cache.py`）が必要です。

## 設計方針

- **ルールベースの核** — シミュレーションロジックは設定ファイルで定義。LLM は判断しない。Seeded RNG で再現可能
- **StatsBomb 準拠** — すべての指標は StatsBomb の定義に従う。独自指標は作らない
- **LLM は説明層** — LLM の役割は知識クエリ、行動説明、レポート生成に限定。シミュレーションの判断は行わない
- **透明なパラメータ** — すべてのシミュレーションパラメータは定義・値・根拠とともに [simulation-rules.md](docs/simulation-rules.md) に記載
- **データ / 分析 / 仮説ラベル** — すべての出力でシミュレーションデータ、モデル由来の分析、推測的仮説を区別

## テスト

```bash
uv run python -m pytest        # 826+ テスト
uv run python -m ruff check .  # リンター
uv run python -m mypy .        # 型チェック
```

## ドキュメント

- [Simulation Rules](docs/simulation-rules.md) — 全パラメータの定義と根拠（英語）
- [Contributing](CONTRIBUTING.md) — 開発フローとガイドライン

## ライセンス

[MIT](LICENSE)

## 謝辞

- [StatsBomb](https://statsbomb.com/) — Open Data の提供
- ルールベースのスポーツシミュレーションと LLM 統合を探求する個人プロジェクトとして構築
