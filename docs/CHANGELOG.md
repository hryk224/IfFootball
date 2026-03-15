# Changelog

## M5 — MVP Finalization

MVP の仕上げとして、実 LLM 接続・Transfer Trigger の UI 統合・ユーザー受け入れテストを完了。

### Added

- 4 プロバイダー対応 LLM クライアント（`llm/providers/`）
  - OpenAI（MVP 標準）、Anthropic、Google Gemini、Groq（ライブデモ用）
  - `LLM_PROVIDER` 明示指定 → API キー auto-detect のフォールバック順
  - SDK import 可否を含めた provider 解決
  - `python-dotenv` で `.env` ファイル読み込み
- Transfer Trigger を UI に experimental 対応として追加
  - サイドバーに Trigger Type ラジオ（Manager Change / Player Transfer）
  - Transfer 時: team radar 非表示、新加入選手の info card 表示
  - MVP スコープに含める判断を記録（M6 で公開導線整備予定）
- `.env.example` — 全プロバイダーの API キー設定テンプレート

### Changed

- `app.py` — LLM 有効時は `generate_report()` で LLM 生成 Markdown を表示、失敗時はデータのみフォールバック
- `pyproject.toml` — `[project.optional-dependencies]` に `llm` グループ追加（`openai` / `anthropic` / `google-genai` / `groq`）、`python-dotenv` を通常依存に追加

---

## M4 — Backtest

Van Gaal 仮想解任シナリオ（Manchester United 2015-16, week 29）のバックテストを実行し、評価結果と課題を記録。

### Added

- `docs/simulation-rules.md` — 全 TOML パラメータの定義・根拠・参照元を記述
- `scripts/backtest_van_gaal.py` — バックテスト実行スクリプト（initialize → run_comparison → player_impact → JSON 出力）

### Findings

- Points delta: +0.5（B-A, 20 runs）— 微増傾向だが std 3.5 に対して統計的に不確実
- Cascade event 過剰発火: form_drop +29.9/run、tactical_confusion +36.0/run（9 試合で）
- tactical_understanding が全員一律 -0.250（個人差が反映されていない）
- 課題詳細は `private/docs/implementation-notes.md` M4 セクションに記録

---

## M3 — Output Layer

M2 のシミュレーション結果を可視化・レポート化し、Streamlit UI で一気通貫の操作を可能にした。

### Added

#### Trigger Enhancement

- `ManagerChangeTrigger.incoming_profile` — 後任監督の戦術プロファイルをオプションで指定可能に
- `TransferInTrigger.player` — 移籍選手の PlayerAgent を payload として保持、engine で squad に追加
  - role ベースの trust 初期化（starter=0.7, rotation=0.5, squad=0.3）
  - deepcopy で branch isolation を保証

#### Visualization

- チーム差分レーダーチャート（`visualization/radar_chart.py` + `radar_data.py`）
  - 5 軸: xG/90（simulation output）、xGA/90（fixed baseline）、PPDA / Possession / Prog Passes（tactical estimates）
  - league average 基準の 0-1 正規化、PPDA / xGA 反転
- 選手差分レーダーチャート（`visualization/player_radar.py` + `player_impact.py`）
  - 4 軸: Form / Fatigue（反転）/ Tactical Understanding / Manager Trust
  - player_id ベースのマッチング、impact score（per-run 差分絶対値の平均）でランキング
- 戦術推定モジュール（`visualization/tactical_estimate.py`）
  - manager profile → PPDA / Possession / Progressive Passes の推定（league average 回帰 + formation 補正）

#### LLM Output Layer

- TP 行動説明（`llm/action_explanation.py` + `prompts/action_explanation_v1.md`）
  - fact / analysis / hypothesis ラベル付き、source_types で data provenance を明示
- 構造化レポート生成（`llm/report_generation.py` + `prompts/report_generation_v1.md`）
  - 5 セクション固定（Summary / Key Differences / Causal Chain / Player Impact / Limitations）
  - LLM 出力のセクション検証、欠落時は構造化フォールバック
- 自然言語入力構造化（`llm/input_structuring.py` + `prompts/input_structuring_v1.md`）
  - manager_change / player_transfer_in の 2 trigger type をパース
  - enum フィールドのデフォルト適用、applied_at 型検証

#### UI

- Streamlit アプリ（`app.py`）
  - サイドバー: competition/team/manager/trigger_week/incoming manager/N runs/seed
  - メインエリア: delta metrics・team radar・player impact radars・structured report

#### Storage

- `ComparisonMeta` / `ComparisonResultWithMeta` — rng_seed / n_runs / trigger_summary / created_at (UTC ISO 8601) を永続化
- cascade*events の run_id 命名規約をドキュメント化（`{branch}*{index}`）

### Changed

#### Config Externalization

- `RuleBasedHandler` の action distribution をハードコードから `turning_points.toml` の `[action_distribution]` セクションに外部化
  - 3 条件（bench_streak_low_trust / low_understanding / default）、3 アクション必須バリデーション

#### Cascade Taxonomy

- `VALID_EVENT_TYPES` に 3 type 追加: `adaptation_progress` / `tactical_confusion`（使用開始）/ `squad_unrest`
- engine: adapt → `adaptation_progress` 記録、low_understanding + adapt → `tactical_confusion` 併記、2+ resist/week → `squad_unrest`

#### Consistency Model

- `PlayerAgent.consistency` をポジショングループ別複合指標に拡張
  - GK/MF: pass_std、DF: def_std（Tackle+Interception）、FW: xg_std
  - 共通副指標: pass_std、複合重み 0.5/0.5（暫定）
  - グループ内 percentile、ゼロイベント試合を含む真の per-match variance

#### Match Result

- `MatchResult` に `expected_goals_for` / `expected_goals_against` フィールド追加

#### Dependencies

- `matplotlib==3.10.8` 追加
- `streamlit==1.45.0` 追加
- `pandas` 3.0.1 → 2.3.3 にダウングレード（streamlit 互換）

---

## M2 — Simulation Foundation

M1 コンポーネントを初期化パイプラインで接続し、週次シミュレーションエンジン・Branch A/B 比較・結果永続化までを実装。

### Added

#### Initialization Pipeline

- M1 コンポーネント（collectors → converters → LLM → storage）を単一の `initialize()` で接続（`pipeline.py`）
  - データを 3 系統（target-team pre-trigger / league-wide pre-trigger / full-season）に分離し未来リークを構造的に防止
  - `build_league_context()` でリーグ平均 PPDA・xG・progressive passes を StatsBomb データから算出
  - `cultural_inertia` を manager tenure から自動更新
  - LLM 補完は optional（`style_stubbornness` のみ更新、`preferred_formation` は StatsBomb 事実値を尊重）
  - commit `5e01dee`

#### Domain Models

- `ManagerChangeTrigger` / `TransferInTrigger` ドメインモデルを追加（`agents/trigger.py`）
  - `trigger_type` は `field(init=False)` で固定値。`applied_at` の意味（第 N 節終了後注入、第 N+1 節から効果）を明記
  - commit `c9b7350`

#### Simulation Rules Config

- TOML 設定ファイルと型付きローダーを実装（`config.py` + `config/simulation_rules/`）
  - `adaptation.toml` — 疲労・戦術習熟度・trust・fatigue_penalty_weight・home_advantage_factor
  - `turning_points.toml` — player TP 閾値（bench_streak / tactical_understanding / trust_low）・manager TP 閾値（job_security / style_stubbornness）
  - `SimulationRules.load(config_dir)` で frozen dataclass として読み込み、`__post_init__` で値域バリデーション
  - commit `890c5c0`

#### Simulation Engine

- Poisson モデルによる試合結果決定を実装（`simulation/match_result.py`）
  - `agent_state_factor` — starters の form/fatigue から乗算係数算出（0.5–1.5 clamp）
  - `home_advantage_factor` で home/away 補正
  - `numpy.random.Generator` を外部注入（seed 制御で再現可能）
  - commit `dc60617`, `cac92c4`
- フォーメーション解析 + スタメン選定を実装（`simulation/lineup_selection.py`）
  - `parse_formation("4-3-3")` → ポジション枠変換
  - `selection_score` — 戦術適合度・trust・form・fatigue penalty・short-term understanding penalty（全て config 駆動）
  - commit `19338d1`
- 週次状態更新（手順 4–7）を実装（`simulation/state_update.py`）
  - fatigue・tactical_understanding（適応曲線）・manager_trust・job_security の更新関数
  - commit `7974fe5`
- ターニングポイント判定 + RuleBasedHandler を実装（`simulation/turning_point.py`）
  - `ActionDistribution` — 確率分布の正規化・バリデーション
  - `TurningPointHandler` Protocol（Phase 1/2 切り替え境界）
  - player TP（bench_streak / low_understanding）・manager TP（job_security warning/critical）
  - commit `50f68d4`
- CascadeEvent モデル + トラッカーを実装（`simulation/cascade_tracker.py`）
  - depth 制限（inclusive）・importance 閾値によるフィルタリング
  - `record_chained()` で因果連鎖を自動追跡
  - commit `6e48ba2`
- 週次シミュレーションループを実装（`simulation/engine.py`）
  - `Simulation.run()` — 10 ステップ週次ループを全フィクスチャに対して実行
  - `SimulationResult` — match_results・cascade_events・final_squad/manager の deepcopy スナップショット
  - `ManagerChangeTrigger` でマネージャー交代（戦術属性をニュートラルにリセット）
  - commit `cd116c1`
- Branch A/B 比較を実装（`simulation/comparison.py`）
  - `run_comparison()` — N 回並行実行、`rng.spawn(2)` で独立再現可能な乱数系列
  - `AggregatedResult` / `DeltaMetrics` / `ComparisonResult` — 勝ち点統計・cascade event 頻度・B-A 差分
  - commit `b928175`

#### Storage

- SQLite ストレージを拡張（`storage/db.py`）
  - `LeagueContext` / `CascadeEvent` / `ComparisonResult` の保存・読み込みを追加
  - `ComparisonResult` は JSON blob で保存（`run_results` 除外、`run_results=()` で復元）
  - `CascadeEvent` は `(comparison_key, run_id, ordinal)` で一意管理
  - commit `3f8028b`

#### Data Quality

- `PlayerAgent.consistency` を xG 分散から算出（`converters/stats_to_attributes.py`）
  - 試合ごとの xG 標準偏差を inverted percentile 化（低分散 = 高 consistency）
  - 5 試合未満の選手は 50.0 placeholder を維持
  - commit `cac92c4`

### Changed (M2 回収)

#### Scale Unification

- `PlayerAgent` の動的状態属性（`current_form` / `tactical_understanding` / `manager_trust`）を 0-100 → 0.0-1.0 に統一
  - config 値（`trust_increase_on_start` / `trust_decrease_on_bench` / `trust_low`）も 0.0-1.0 に変更
  - simulation 内の `* 100.0` 暫定変換コードを全て除去
  - `lineup_selection.py` の `tactical_fit` を `/100.0` で正規化し、全スコア成分を 0-1 スケールに揃えた
  - commit `4057f90`

#### Config Separation

- `home_advantage_factor` を `AdaptationConfig` から分離し、新設の `MatchConfig`（`config/simulation_rules/match.toml`）に移動
  - `simulate_match()` に `match_config: MatchConfig` 引数を追加
  - `AdaptationConfig` の「0-100 scale」例外注記を除去
  - commit `7bbc9f5`

---

## M1 — Data Foundation

StatsBomb Open Data からエージェント初期化に必要な全ドメインオブジェクトを構築するデータ基盤層を実装。

### Added

#### Data Collection

- StatsBomb Open Data 取得層を実装（`collectors/statsbomb.py`）
  - `get_competitions()` / `get_matches()` / `get_events()` / `get_lineups()` — statsbombpy ラッパー
  - commit `2e172ab`

#### Domain Models

- `PlayerAgent` ドメインモデルを追加（`agents/player.py`）
  - `RoleFamily` / `BroadPosition` Enum、技術属性・適応属性・動的状態の全フィールド定義
  - commit `676489e`
- `TeamBaseline` ドメインモデルを追加（`agents/team.py`）
  - StatsBomb 指標（xG/90・xGA/90・PPDA・progressive passes・possession）とリーグ立ち位置フィールド
  - commit `0174f81`
- `ManagerAgent` ドメインモデルを追加（`agents/manager.py`）
  - StatsBomb 由来の戦術属性と LLM 由来の仮説属性（`style_stubbornness`）を分離して定義
  - commit `f3166ac`
- `Fixture` / `FixtureList` / `OpponentStrength` ドメインモデルを追加（`agents/fixture.py`）
  - `frozen=True` — Branch A/B 並行比較で安全に共有できるよう不変設計
  - commit `eb3134b`
- `LeagueContext` ドメインモデルを追加（`agents/league.py`）
  - StatsBomb 由来の事実フィールドと LLM 由来の仮説フィールドを明確に分離
  - `frozen=True`、仮説フィールドは `dataclasses.replace()` で更新
  - commit `8870709`

#### Converters

- スタッツ → `PlayerAgent` 属性変換を実装（`converters/stats_to_attributes.py`）
  - 同リーグ・同シーズン内 percentile 正規化（出場時間フィルタ付き）
  - commit `e56cde4`
- チーム指標集計 → `TeamBaseline` ビルダーを実装（`converters/team_stats.py`）
  - xG/xGA・PPDA・progressive passes・possession・`cultural_inertia`（在任期間から算出）
  - commit `14bfae9`
- 監督指標集計 → `ManagerAgent` ビルダーを実装（`converters/manager_stats.py`）
  - 在任区間推定（`matches.home_managers` / `away_managers` から抽出）
  - pressing intensity・possession preference・preferred formation を StatsBomb イベントから導出
  - commit `422657b`
- `FixtureList` / `OpponentStrength` ビルダーを実装（`converters/fixture_stats.py`）
  - Elo レーティング算出（初期値 1500・K=20・match_week 昇順処理）
  - commit `9446410`

#### Storage

- SQLite ストレージ層を実装（`storage/db.py`）
  - `PlayerAgent` / `TeamBaseline` / `ManagerAgent` / `OpponentStrength` / `FixtureList` を保存・読み込み
  - UPSERT（`ON CONFLICT DO UPDATE`）で全テーブルを冪等に更新
  - `fixture_lists` ヘッダーテーブルにより「保存済み空 snapshot」と「未保存」を区別
  - `save_fixture_list` は DELETE+INSERT をトランザクション化し途中失敗を防止
  - commit `bd03ef1`

#### LLM Knowledge Query

- `LLMClient` Protocol を定義（`llm/client.py`）
  - 1メソッド（`complete`）のみでプロバイダー差し替え可能な設計
- LLM 知識クエリを実装（`llm/knowledge_query.py`）
  - `query_manager_style()` — `style_stubbornness`（float）・`preferred_formation` を仮説ラベルとして取得
  - `query_league_characteristics()` — `pressing_level` / `physicality_level` / `tactical_complexity` を取得
  - 不正 LLM 出力はモジュール定数のデフォルト値にフォールバック
  - system / user ロール分離によるプロンプトインジェクション対策
- システムプロンプトを外部ファイルで管理（`prompts/knowledge_query_v1.md`）
  - commit `8870709`
