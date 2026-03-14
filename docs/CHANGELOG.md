# Changelog

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
