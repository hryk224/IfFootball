# Changelog

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
