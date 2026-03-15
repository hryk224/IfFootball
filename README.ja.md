# IfFootball

[English version](README.md)

StatsBomb Open Data を使ったサッカー "what-if" シミュレーター。ルールベースの因果シミュレーションと LLM による分析レポートで、監督交代や選手移籍がシーズンにどう影響したかを探ります。

## 概要

IfFootball は並行シミュレーションを実行します。何も変化しないケース（Branch A）とトリガーを適用したケース（Branch B）を N 回の確率的シミュレーションで比較します。

**シナリオ例:** _「2015-16 シーズン第 29 節後にマンチェスター・ユナイテッドがファン・ハールを解任していたら？」_

### 主な機能

- **監督交代** — シーズン中の解任をシミュレート。戦術プロファイルのリセット、選手の信頼度再調整、適応曲線を反映
- **選手移籍** _(実験的)_ — 役割に応じた信頼度初期化で選手をスカッドに追加
- **A/B 比較** — Poisson 試合モデル、週次状態更新（疲労・信頼度・戦術理解度）、ターニングポイント検出とカスケードイベント追跡
- **可視化** — Branch A/B を比較するチーム・選手レーダーチャート
- **LLM レポート** — 事実 / 分析 / 仮説のラベル付き構造化比較レポート
- **Streamlit UI** — 入力からシミュレーション、出力までを一気通貫で操作できるシングルページアプリ

### 動作フロー

```
ユーザー入力（チーム、監督、トリガー節）
    |
    v
StatsBomb データ --> エージェント初期化（選手、監督、チームベースライン）
    |
    v
シミュレーションエンジン（N 回 x 2 ブランチ）
    |--- Branch A: 変化なし
    |--- Branch B: トリガー適用
    |
    v
比較 & 可視化（レーダーチャート、カスケードイベント、レポート）
```

## データソース

IfFootball は [StatsBomb Open Data](https://github.com/statsbomb/open-data) のみを使用します。すべての指標と用語は StatsBomb の定義に準拠しています。スクレイピングは一切行いません。

現在対応しているコンペティション（`config/targets.toml`）:

| コンペティション | シーズン | チーム                                                                                             |
| ---------------- | -------- | -------------------------------------------------------------------------------------------------- |
| プレミアリーグ   | 2015-16  | Manchester United, Manchester City, Arsenal, Liverpool, Chelsea, Tottenham Hotspur, Leicester City |
| ラ・リーガ       | 2015-16  | Real Madrid, Barcelona, Atletico Madrid                                                            |

## セットアップ

### 必要環境

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)（パッケージマネージャー）
- Node.js（Markdown フォーマット用のみ）

### インストール

```bash
git clone https://github.com/hryk224/IfFootball.git
cd IfFootball
cp .env.example .env
uv sync --extra dev
npm install
```

### LLM セットアップ（任意）

LLM によるレポート生成を有効にするには、プロバイダー SDK をインストールし API キーを設定します:

```bash
uv sync --extra dev --extra llm
```

`.env` に以下のいずれかを設定:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
GROQ_API_KEY=gsk_...
```

対応プロバイダー: OpenAI, Anthropic, Google Gemini, Groq。LLM 未設定時はデータのみモードで動作し、シミュレーションデータから構造化レポートを生成します。

## 使い方

### Streamlit UI

```bash
uv run streamlit run app.py
```

サイドバーでパラメータ（チーム、監督、トリガー節、シミュレーション設定）を入力し「Run Simulation」をクリック。

### バックテストスクリプト

```bash
uv run python scripts/backtest_van_gaal.py
```

ファン・ハール解任シナリオ（マンチェスター・ユナイテッド、第 29 節、20 回実行）を実行し、結果を `output/backtest_van_gaal/results.json` に出力します。

## 設計方針

- **ルールベースの核** — シミュレーションロジックは設定ファイルで定義。LLM は判断しない。Seeded RNG で再現可能
- **StatsBomb 準拠** — すべての指標は StatsBomb の定義に従う。独自指標は作らない
- **LLM は説明層** — LLM の役割は知識クエリ、行動説明、レポート生成に限定。シミュレーションの判断は行わない
- **透明なパラメータ** — すべてのシミュレーションパラメータは定義・値・根拠とともに [simulation-rules.md](docs/simulation-rules.md) に記載
- **事実 / 分析 / 仮説ラベル** — すべての出力でデータに基づく事実、モデル由来の分析、推測的仮説を区別

## テスト

```bash
uv run python -m pytest        # 547+ テスト
uv run python -m ruff check .  # リンター
uv run python -m mypy .        # 型チェック
```

## ドキュメント

- [Simulation Rules](docs/simulation-rules.md) — 全パラメータの定義と根拠（英語）
- [Changelog](docs/CHANGELOG.md) — マイルストーンごとの変更履歴
- [Contributing](CONTRIBUTING.md) — 開発フローとガイドライン

## ライセンス

[MIT](LICENSE)

## 謝辞

- [StatsBomb](https://statsbomb.com/) — Open Data の提供
- ルールベースのスポーツシミュレーションと LLM 統合を探求する個人プロジェクトとして構築
