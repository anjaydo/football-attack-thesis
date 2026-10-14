# Tham chiếu module

Mọi đường dẫn tương đối với `src/attack_analysis/`. Mỗi mục gồm: mục đích, API chính, hằng số quan trọng, phụ thuộc trong dự án, lưu ý. Lỗi/giới hạn chi tiết được gom ở [known-issues.md](known-issues.md).

---

## 1. Hợp đồng miền

### `models.py`

Hợp đồng V1 (`Event`, `AttackSequence`) và V2 possession-first.

- **Hằng số:** `PITCH_LENGTH = 105.0`, `PITCH_WIDTH = 68.0`, `ATTACK_V2_SCHEMA_VERSION = "2.0"`, `ATTACK_V2_RUBRIC_VERSION = "attack-v2-rubric-1"`, `ATTACK_TERMINAL_ALLOWLIST = {shot_attempt, headed_attempt, forced_goal_line_out}`, `FORCED_GOAL_LINE_OUT_CAUSES = {keeper_save_out, defender_block_out, defender_deflection_out, defender_clearance_out}`.
- **Enum:** `SourceTier`, `SequenceStatus` (`complete`, `left_censored`, `right_censored`, `uncertain`), `TerminalType` (3 terminal attack + `turnover`, `unforced_out`, `foul`, `offside`, `period_end`, `unknown`), `PossessionState` (`dead_ball`, `controlled`, `contested`, `ball_flight`, `unknown`).
- **V1:** `Event` (timestamp, event_type, team, x/y, end_x/y, outcome, xg + `xg_source` + `xg_inputs`, xT, frame, player, confidence, source, evidence_id). `xg_inputs` bị từ chối nếu chứa khóa rò rỉ nhãn (`goal`, `statsbomb_xg`, `target`, `outcome`). `AttackSequence` yêu cầu một đội, event có thứ tự thời gian.
- **V2** (frozen, có `from_dict`/`to_dict`, id băm nội dung được kiểm lại khi khởi tạo):
  - `FieldProvenanceV2(field_name, source_tier, source_record_id, evidence_ids, confidence)`.
  - `ObservedActionV2`: `action_id = "actv2_" + sha256(match, period, frame, ts, action_type, actor_team, actor_id, jersey, raw_left_to_right)[:24]`. Không gồm possession, xác suất, x/y, nên làm giàu action không đổi id. Tier `fused` cần provenance đầy đủ, 2 tier gốc và `association_id`.
  - `TerminalEventV2`: id `termv2_…`; `forced_goal_line_out` cần beneficiary = đội kiểm soát, last touch là đối thủ, cause hợp lệ.
  - `PossessionStateTransitionV2`, `PossessionSequenceV2` (id `posv2_…`; `complete` cần terminal khác `unknown`; legacy phải `uncertain`).
  - `AttackSequenceV2` (id `atkv2_…`; cần possession `complete` và terminal allowlist).
  - `PossessionValueV2` (xT/EPV/p_score, hiện để trống/abstain).

### `observations.py`

Schema dòng `observations.jsonl`.

- `BoundingBox`, `TrackedObject` (track_id, object_type, confidence, bbox, team, pitch_x/y, anchor_image, keypoints 2/3/5 phần tử, speed + speed_unit `px/s` hoặc `km/h`), `FrameObservation` (frame_index, timestamp_sec, objects, calibration_confidence, scene_cut, field_geometry).
- `OBJECT_TYPES = {player, goalkeeper, referee, ball, other}`; vision chỉ phát `player`, `goalkeeper`, `ball`.
- Một dòng không hợp lệ làm hỏng cả file khi đọc.

### `research.py`

Mô hình evidence và verdict.

- `AttackFeatures`, `BenchmarkRecord`, `EvidenceBundle` (`downstream_ready` cần các gate `events`, `calibration`, `tracking`, và `attack_qualified` nếu có), `EvidenceClaim`, `EvaluationVerdict` (V1: nhãn `effective`/`promising`/`ineffective`/`insufficient_evidence`, score 1–5), `AttackConclusionV2` (schema `attack-conclusion-v2/1.0`; bắt buộc `effectiveness_verdict is None` và `abstained_effectiveness is True`), `insufficient_evidence_verdict(...)`.
- `CHANCE_QUALITY_VALUES = (high, medium, low, unknown)`.

### `provenance.py`

Provenance artifact theo băm, phát hiện artifact cũ.

- `canonical_json` (sắp khóa, bỏ khóa bí mật, từ chối NaN), `content_sha256`, `file_sha256` (từ chối file bí mật như `.env`), `tree_sha256`.
- `build_provenance(...)` trả về các băm video, dataset, config, model, schema, postprocess, cây mã nguồn (`.py/.toml/.json/.yaml/.yml`) và `artifact_identity_sha256`.
- `compare_manifest_provenance`, `require_current_manifest`, `mark_legacy_manifest`, `invalidate_artifact_tree` (ghi `STALE.json`).
- `MANIFEST_VERSION = "attack-analysis-manifest-v2"`, `PROVENANCE_SCHEMA_VERSION = "artifact-provenance-v2"`, `StaleArtifactError`.

---

## 2. Event FOOTPASS

### `footpass.py`

Adapter FOOTPASS cho V2 và V1 legacy.

- **Hằng số:** `FOOTPASS_ACTIONS = {1: drive, 2: pass, 3: cross, 4: throw_in, 5: shot, 6: header, 7: tackle, 8: block}`, `FOOTPASS_ROLES` (1 = goalkeeper … 13), `PENALTY_BOX_DEPTH_M = 16.5`, `PENALTY_BOX_WIDTH_M = 40.32`, `SHOT_CORRIDOR_M = 1.5`.
- **Định dạng vào:** play-by-play `{"events": {"game_18_H1": [[frame, direction_flag, jersey, action_code, conf | bbox_visible, replay?], ...]}}`. HDF5 mỗi khóa trận: cột 0 frame, 1 player_id, 2 cờ hướng/đội, 3 số áo, 4 role, 5–6 x/y [0, 1], 9–12 bbox x/y/w/h (cột 9 là NaN khi không visible), 13 mã action.
- **Hàm chính:**
  - `split_footpass_match_key("game_18_H1") -> ("game_18", 1)`.
  - `normalize_footpass_position`, `possession_relative_position`, `actions_in_attack_view`, `attack_view_median_x`.
  - `load_footpass_observed_actions_v2(playbyplay, *, match_key, source, tactical_h5, fps, min_confidence, model_name, class_confidence_thresholds, source_tier)`.
  - `prepare_footpass_actions_v2(...) -> TerminalInferenceResult`, `decode_footpass_sequences_v2(...)`, `footpass_v2_summary(...)`.
  - `footpass_freeze_frame(...)`, `load_footpass_freeze_frame(...)`: snapshot 11v11 có GK, foe/ally/defending, `defenders_in_box`, `allies_ahead`, `foes_in_shot_corridor`, `gk_distance_m`, evidence id `footpass-hdf5:{key}:{frame}:freeze`.
  - Legacy: `load_footpass_events`, `segment_footpass_attacks`, `footpass_summary`.
- **Lưu ý:** không có HDF5 thì không có team id. Hệ tọa độ V2 là attack-view của actor.

### `footpass_terminal_inference.py`

Gắn bằng chứng terminal cho `header`/`block` bằng qualifier (không đổi `action_type`).

- `FootpassTerminalInferenceConfig(fps=25, event_cluster_sec=0.5, max_observation_gap_sec=8, attacking_third_x=70, penalty_box_x=88.5)`.
- `infer_footpass_terminals(actions, config) -> TerminalInferenceResult` (đếm `headed_promoted`, `blocks_promoted`, ...).
- `INFERENCE_VERSION = "footpass-terminal-inference-v1"`, `RARE_FOOTPASS_CLASSES = ("shot", "header", "block")`.

### `event_calibration.py`

Ngưỡng confidence theo lớp cho DST.

- `fit_event_threshold_profile(ground_truth, predictions, *, tuning_matches, tolerance_frames=12, baseline_threshold=0.15)`: quét 0.00–0.95 bước 0.05, tối đa F1 từng lớp trên trận tuning; trả về `schema_version = "dst-thresholds-v1"` với báo cáo tuning / held-out / toàn bộ.
- `apply_event_thresholds`, `save_event_threshold_profile`, `load_event_thresholds`.

### `prediction_sources.py`

Registry file play-by-play dự đoán.

- `REGISTRY_SCHEMA_VERSION = "prediction-sources/1.0"`, `DEFAULT_REGISTRY_PATH = models/prediction-sources.json`, `DEFAULT_FOOTPASS_ROOT = ../FOOTPASS`, `OFFICIAL_DELTA = 12`, `OFFICIAL_CONFIDENCE = 0.15`.
- `OfficialScore`, `PredictionSource` (TAAD không bao giờ là primary), `PredictionRegistry` (tối đa một primary).
- `load_registry`, `save_registry`, `verify_source` (kiểm SHA-256), `resolve_prediction_source`, `registry_report`, `parse_official_output`, `official_footpass_score` (gọi `FOOTPASS/evaluation.py`), `decide_role` (DST chỉ lên primary khi F1 **lớn hơn hẳn**), `register_prediction_source` (sao chép thành `playbyplay_<id>_val.json`, không ghi đè file khác), `source_choices`.

### `footpass_audit.py`

`audit_footpass_v2(playbyplay, tactical_h5, video_root, output)`: kiểm toán oracle. Cổng hình học gồm: cú sút `game_18_H1` frame 2640 có x = 82.094; median x cú sút oracle > 85; ít nhất 95% cú sút có x ≥ 70. Kiểm thêm bất biến attack. Ghi `footpass-oracle-audit-v2`, `release_status = research_only`.

---

## 3. Decoder Attack-v2

### `attack_v2.py`

- `FootpassAttackV2Config(max_observation_gap_sec=8.0, event_cluster_sec=0.5, fps=25, controlled_action_types={drive, pass, cross}, restart_action_types={throw_in})`.
- `decode_footpass_attack_v2(actions, config) -> AttackV2DecodeResult(possessions, attacks, excluded_replay_action_ids, duplicate_action_ids, unassigned_action_ids)`.
- `qualify_attack_v2(possession, *, calibrated_confidence=None, calibration_id=None)`.
- `legacy_attack_to_possession_v2(...)`.
- **Ưu tiên terminal:** headed (100) > shot (90) > forced out (80) > turnover/out/foul/offside/period_end (50); ưu tiên cùng đội đang kiểm soát.
- **Gom cụm:** Δt ≤ 0.5 s và Δframe ≤ 14; tách cụm nếu đã có terminal mà gặp action kiểm soát/restart/terminal khác.
- **Trạng thái:** gap > 8 s hoặc scene cut → censor phải rồi censor trái; replay không rõ → `uncertain`.
- **Attack:** possession `complete` hoặc `left_censored` (kết thúc quan sát được) có terminal allowlist; `right_censored`/`uncertain` không bao giờ thành attack.

### `attack_v2_shot_insights.py`

Audit cú sút cho UI.

- `GoldOutcomeIndex.load(project_root)` đọc `data/attack-v2-{development,train}.jsonl`.
- `resolve_shot_outcome` (gold → scoreboard → terminal → action), `format_shot_outcome`, `build_shot_insights` (`promoted` / `censored` / `unassigned`), `filter_sequence_entries`, `actions_with_resolved_outcome`.
- Nhãn bộ lọc: `Confirmed Attack-v2`, `Shot terminal · bị censor`, `All possessions` (UI thêm `Attack-v2 · bàn thắng`).

### `scoreboard_goals.py`

Goal truth cho FOOTPASS (dataset không có kết quả cú sút).

- `stable_score_changes` / `goal_times_from_changes`: tỉ số giữ ≥ 3 lần đọc liên tiếp; tăng đúng 1 bàn mới tính, lần đọc lỗi bị đảo lại thì bỏ.
- `restart_goal_times`: cú sút/đánh đầu → dừng ≥ 45 s → đội kia chơi bóng đầu tiên ở x ∈ [20, 60] (phần sân nhà, không phải phát bóng lên). Chỉ dùng khi feed không đọc được scoreboard (nguồn `event_restart`); trên 162 bàn có scoreboard: recall 0.73, precision 0.94.
- `scoring_action`: cú sút/đánh đầu live cuối cùng trong 150 s trước khi tỉ số đổi, ưu tiên cú có ≥ 15 s dừng phía sau (ăn mừng).
- `ScoreboardGoalIndex` đọc/ghi `data/footpass-scoreboard-goals.json`; `possession_outcomes` trả `possession_id → (goal | no_goal, source)`.
- CLI: `attack-analysis build-scoreboard-goals <playbyplay_train.json> <playbyplay_val.json> [--video-root D:/videos]` (đọc cache `runs/goal-truth/<match>.readings.json`, bỏ qua frozen test).

### `v2_adapters.py`

Chuyển luồng `Event` V1 (từ video) sang V2: `event_to_observed_action_v2`, `event_stream_to_observed_actions_v2` (chèn action `scene_cut`), `decode_event_stream_v2`. Mặc định `replay_status="unknown"` nên mọi possession thành `uncertain`.

---

## 4. Vision

### `vision_pipeline.py`

- `VisionConfig` (frozen): `pose_model="yolo26n-pose.pt"`, `detection_model="yolo26n.pt"`, `ball_detection_model=None`, `ball_class_id=32`, `player_tracker="botsort.yaml"`, `ball_tracker="bytetrack.yaml"`, `device="cpu"`, `image_size=960`, `player_confidence=0.2`, `ball_confidence=0.15`, `keypoint_confidence=0.25`, `speed_smoothing=0.35`, `team_classification=True`, `team_min_tracks=4`, `scene_cut_threshold=0.35`, `field_lines=False`, `pose_on_crop=True`, `pose_crop_imgsz=320`, `ball_tile_size=0` (tắt), `ball_tile_overlap=0.1`, `ball_max_side_px=80`.
- `annotate_video(input, output, config, projector, observations_path, roi) -> VisionRunSummary`.
- `iter_tiles`, `merge_ball_detections`, `TileBallAssociator` (id bóng từ tile bắt đầu ở 10000).
- Batch: `BALL_TILE_BATCH = 2`, `POSE_CROP_BATCH = 4` (cho GPU 6 GB). Anchor cầu thủ là trung bình mắt cá (keypoint 15/16), nếu không có thì đáy bbox. Điểm ngoài sân bị bỏ. Trọng tài bị tính là cầu thủ.

### `dynamic_calibration.py`

- `DynamicCalibrationConfig(source_width=1920, source_height=1080, min_correspondences=6, ransac_threshold_m=2.0, max_median_error_m=0.5, max_p90_error_m=1.5, min_frame_coverage=0.9, min_inlier_ratio=0.6, max_interpolation_gap_frames=25, max_projector_gap_frames=25)`.
- `build_footpass_dynamic_projector(tactical_h5, match_key, *, start_frame, end_frame, video_width, video_height, config) -> DynamicPitchProjector` (`.report`, `.for_frame`, `.confidence_for_frame`, `.support_for_frame` trả `direct`/`interpolated`/`extrapolated`/`unsupported`).
- `candidate_bbox_source_sizes`, `select_bbox_source_size`: tự chọn độ phân giải nguồn của bbox.
- Cảnh báo báo cáo: `low_dynamic_calibration_coverage`, `no_valid_dynamic_homography`, `bbox_source_auto_selected`, `bounded_temporal_extrapolation_used`, `ineligible_replay_or_missing_bbox_excluded`.

### `geometry.py`

`PitchProjector` (homography 4 điểm, `from_json`), giao thức `FramePitchProjector`, `projector_for_frame`, `projector_confidence`, `ordered_image_quad`, `projector_from_pitch_polygon`. `VIDEO_ONLY_CALIBRATION_CONFIDENCE = 0.25`.

### `field_geometry.py`

Phân tích sân OpenCV: mask cỏ HSV, polygon cỏ, vạch trắng, Canny + Hough, giao điểm. `FieldLineConfig`, `FieldGeometryDetector`, `FieldGeometry` (`method = "opencv_hsv_hough_v1"`), `analyze_field_image`, `draw_field_geometry`.

### `roi.py`, `scene.py`, `team.py`, `motion.py`

- `PolygonROI.from_json/contains`.
- `SceneCutDetector(threshold=0.35, sample_step=16)`.
- `OnlineTeamClassifier(min_tracks=4, ...)` k-means 2 cụm, bỏ phiếu theo track; `extract_jersey_feature` (median Lab vùng thân áo).
- `SpeedEstimator(smoothing=0.35)` EMA, không lọc outlier.

### `video.py`, `video_paths.py`, `browser_video.py`

- `inspect_video(path) -> VideoMetadata`.
- `VideoPathResolver.from_environment(project_root)`: `processing_path`, `preview_path`, `yolo_video_path`, các hàm đổi giây/frame giữa đồng hồ trận và preview, `video_clock`, `describe_preview`.
- `PreviewClockProjector(inner, match_key, resolver, preview_fps)`.
- `operations_report(project_root, expected_match_ids)`.
- Hằng số: `DEFAULT_MATCH_FPS = 25.0`, `VIDEO_SYNC_SCHEMA = "footpass-video-sync/1.0"`, `OVERLAY_CACHE_KEY = "fullhd_pose_crop_ball_fullframe_v6"`, `CALIBRATION_CACHE_KEY = "footpass_dynamic_roi_homography_v3_temporal"`.
- `ensure_browser_compatible_video` chuyển mp4v sang H.264 bằng FFmpeg (`shutil.which` hoặc `imageio_ffmpeg`), ghi `<stem>-browser.mp4`; không có FFmpeg thì trả file gốc.

---

## 5. Chất lượng và pipeline

### `quality.py`

- `read_observations_jsonl`, `analyze_observation_quality(frames, keypoint_confidence=0.25) -> ObservationQualityReport` (khoảng 40 metric + warnings), `track_birth_limit_per_100_frames`.
- Điểm = 100 × (0.15 frame có người + 0.15 số người/10 + 0.25 frame có bóng + 0.08 tracking người + 0.07 continuity người + 0.05 tracking bóng + 0.05 continuity bóng + 0.20 keypoint). `fail` nếu không có người hoặc bóng, hoặc điểm < 40; `warn` nếu có cảnh báo hoặc điểm < 75.

### `pipeline.py`

- `validate_video_pipeline_source_contract(*, source_tier, projector, dataset_paths)`.
- `run_video_pipeline(input, run_directory, vision_config, projector, roi, event_config, xg_model_path, *, source_tier, dataset_paths)`: ghi `annotated.mp4`, `observations.jsonl`, `quality.json`, `manifest.json` (`attack-run-v2.0`), tùy chọn `calibration.json`; nếu qua cổng thì thêm `events.json`, `sequences-v2.json`, `possession-candidates-v1.json`. Thứ tự cổng xem [architecture.md](architecture.md#3-luồng-b-video--observations).

### `upload_pipeline.py`

`run_cached_video_pipeline(input, cache_root, video_digest, config, *, projector, xg_model_path, cache_context, source_tier, dataset_paths, ...)`: cache theo provenance tại `cache_root/<sha256 video>/<key16>/manifest.json`. `VISION_CACHE_SCHEMA = "upload-vision-v4"`. `vision_cache_key` chỉ dùng làm khóa session.

### `window_analysis.py`

`persist_footpass_window_stream(manifest, sequence, *, start_sec, end_sec)`: ghi sidecar legacy `structured-event-stream.json`, `selected-possession-candidate-v1.json`, `possession-candidate-v1-manifest.json`. Không bao giờ tạo Attack-v2 hay xG (`xg_status = withheld_legacy_unvalidated`).

### `event_inference.py`

`infer_events_from_observations(frames, config)`: suy `possession_start`, `carry`, `pass`, `turnover`, `shot_candidate` (chưa xác minh, không phải terminal) từ track đã hiệu chỉnh. `EventInferenceConfig` gồm khoảng cách sở hữu 3 m, cửa sổ sút 1.5 s, hướng tấn công từng team.

---

## 6. Fusion và xG

### `fusion.py`

- `FusionAssociation(action_id, association_id, track_id, method, distance_m, iou, xg_raw, spatial_context)`, `FusedXGResult(association, xg_record, withheld_reason)`.
- `associate_action_with_frame(...)`: ghép theo khoảng cách sân ≤ 3 m hoặc IoU ≥ 0.3; `method` là `pitch_nearest`, `bbox_iou`, `unmatched`, `footpass_tactical`, `event_location`.
- `vote_team_mapping`, `infer_goalkeeper`, `infer_body_part_from_pose`, `project_pose_to_pitch`, `align_yolo_tracks_to_freeze`, `possession_buildup`, `play_pattern_evidence`, `shot_type_evidence`, `merge_fusion_context`.
- `score_attack_with_observations(attack, frame, *, model, team_mapping, ball_f1, min_ball_f1=0.80, footpass_freeze) -> FusedXGResult`.
- Play pattern (thứ tự): `throw_in` (event) → `corner` (x ≥ 99 và gần cờ góc) → `free_kick` (sút ngoài vòng cấm, ≤ 3 action, bắt đầu gần) → `counter` (bắt đầu x < 40, ≤ 10 s, ≤ 3 pass) → `regular_play`.

### `xg.py`

Hình học cú sút (`shot_distance`, `shot_angle`, `GOAL_WIDTH = 7.32`, `GOAL_CENTER_Y = 34`), `heuristic_xg`, `TrainedXGModel` / `load_trained_xg_model` (xG-v1, điền median cho giá trị thiếu).

### `xg_v2.py`

- `XG_V2_SCHEMA_VERSION = "xg-v2.0"`, `XG_V2_FEATURE_COLUMNS` (74 cột: vị trí/góc/khoảng cách chuẩn hóa + mask, body part one-hot, áp lực + mask, GK visible/vị trí + mask, defenders visible + mask, play pattern one-hot 9 lớp, shot type one-hot 5 lớp, đếm observed/foe/ally/defending chuẩn hóa + mask).
- `normalize_xg_v2_features(raw)`, `XGFeatureVectorV2`, `TrainedXGModelV2` (`model_id = "xg-v2:sha256:…"`, `.predict`, `.score_attack`), `ConfirmedXGRecordV2` (chỉ cho terminal shot/header, không cho tier legacy).
- Metric: `evaluate_xg_v2` (Brier, ECE 10 bin, slope/intercept), `paired_match_cluster_bootstrap_brier_delta`, `xg_v2_acceptance_gates` (Brier ≤ 0.09, ECE ≤ 0.03, slope 0.8–1.2, |intercept| ≤ 0.10, cận trên delta Brier < 0).
- `load_trained_xg_v2_model(path, *, expected_sha256, allow_research_only)`.

### `xg_data.py`, `xg_training_v2.py`

- Xuất CSV từ StatsBomb open data: `export_statsbomb_xg_dataset` (V1, chỉ cú sút có 360), `export_statsbomb_xg_v2_dataset` (V2, mọi cú sút, giữ missing). `PRESSURE_RADIUS_M = 2.0`.
- `train_xg_v2_models(csv, output, *, config)`: split nhóm theo trận (train/calibration/test), GroupKFold nội bộ chọn giữa `geometry_logistic` và `hist_gradient_boosting`, Platt trên logit, baseline hình học, ghi joblib + `.metrics.json`; trạng thái `research_only` hoặc `statsbomb_gate_passed_domain_unvalidated`.

### `attack_evidence.py`

- `build_attack_evidence(attack_v2, *, window_manifest, xg_model_path, team_mapping, allow_research_only=True, benchmarks, web_context, model, frames, tactical_h5) -> AttackEvidenceBundle` (schema `attack-evidence-v2/1.0`).
- `AttackEvidenceBundle.evidence_ids()`, `.to_dict()`, `.as_legacy_bundle()`.
- Tiện ích: `default_team_mapping`, `resolve_xg_v2_model_path`, `observations_path_from_manifest`, `video_frame_for_match_frame`, `frame_for_terminal`, `conclusion_ready`.
- Gate trong bundle: `events`, `attack_v2`, `shot_terminal`, `yolo_observations`, `xg_model`, `footpass_freeze`, `tracking`, `calibration`.

---

## 7. Kết luận, công cụ, benchmark

### `llm.py`

- `LLMClient` (ABC), `DeterministicEvidenceClient` (`evaluate`, `evaluate_v2`, chạy offline), `OpenAIResponsesClient(*, model, project_root, api_key=None, max_tool_rounds=6, max_contract_retries=1, response_language="English")`.
- `PROMPT_VERSION = "attack-verdict-v1"`, `PROMPT_VERSION_V2 = "attack-verdict-v2"`; schema `attack_evaluation_verdict` / `attack_conclusion_v2`, strict.
- `describe_openai_error`, `load_project_skill_instructions` (ghép 3 file `.cursor/skills/*/SKILL.md`), `verdict_json_schema`, `conclusion_v2_json_schema`.
- Chance quality từ xG: ≥ 0.25 `high`, ≥ 0.10 `medium`, còn lại `low`, `None` là `unknown`.

### `agent_tools.py`

`EvidenceToolRegistry(bundles, benchmarks, web_search)` với 7 công cụ whitelist: `get_attack_evidence`, `get_role_benchmark`, `resolve_player_identity`, `lookup_player_snapshot`, `search_frozen_stats`, `web_search_context`, `verify_claims`. Có `trace` ghi lại mọi lần gọi. `verify_verdict(verdict, bundle)`.

### `mcp_server.py`

Server JSON-RPC qua stdio tự viết (không dùng MCP SDK), `protocolVersion 2024-11-05`, các method `initialize`, `tools/list`, `tools/call`. Chạy bằng `attack-analysis serve-mcp --bundle --benchmarks`.

### `web_search.py`

`brave_search_context(args)`: tối đa 3 kết quả, timeout 12 s, id `web:{i}`; cần `BRAVE_SEARCH_API_KEY`.

### `benchmarks.py`

`FrozenBenchmarkStore.from_jsonl(...)` (`get_role_benchmark`, `lookup_player`, `search`, `resolve_player_identity`), `build_footpass_role_benchmarks(tactical_h5, output)` (per-90 theo role, bỏ cầu thủ dưới 10 phút), `write_historical_snapshot`.

### `overview.py`

`attack_v2_research_overview(attack, *, xg_record, benchmarks, query)`: tổng quan research-only, `effectiveness_verdict = None`.

### `env.py`, `environment.py`

- `load_project_environment` (đọc `.env`, không ghi đè biến sẵn có), `openai_api_key_configured`, `openai_model` (mặc định `gpt-5.4-nano`), `brave_search_api_key`, `xg_v2_model_path`.
- `default_torch_device` (`"0"` nếu có CUDA), `default_inference_imgsz` (1280 trên GPU, 960 trên CPU), `environment_report` (gói, công cụ, khuyến nghị).

### `pitch_visualization.py`

`pitch_event_views`, `plot_pitch_events`, `plot_attack_pitch` (mplsoccer), `interactive_pitch_events` (Altair), `plot_freeze_frame` (freeze-frame + pose + inset). Nhãn tiếng Việt.

---

## 8. Đánh giá, gold, release

### `sequence_evaluation.py`

- `SequenceSpan.from_value` (nhận dict hoặc object), `temporal_iou`, `match_sequence_spans` (Hungarian theo nhóm match/period/team), `sequence_metrics_at_threshold`, `sequence_match_bootstrap_intervals`, `paired_f1_improvement_bootstrap`, `sequence_evaluation_report` (`attack-sequence-evaluation/v2`), `select_balanced_operating_point`, `evaluate_acceptance_gates`, `attack_v2_acceptance_specs`.
- `DEFAULT_TIOU_THRESHOLDS = (0.25, 0.5, 0.75)`, `ATTACK_V2_METRIC_REGISTRY` (xem cổng ở [architecture.md](architecture.md#5-luồng-d-gold--đánh-giá--release)).

### `research_evaluation.py`

`action_spotting_report` (Hungarian, ưu tiên số cặp rồi sai số frame), `action_spotting_dataset_report` (kèm audit lớp hiếm), `boundary_f1`, `top_view_error_report`, `match_bootstrap_interval`.

### `detection_evaluation.py`

So box với SoccerNet GSR `Labels-GameState.json`: `evaluate_soccernet_observations`, `evaluate_soccernet_jsonl`, `benchmark_ball_confidence` / `benchmark_ball_jsonl` (gate F1 ≥ 0.80, P ≥ 0.75, R ≥ 0.75; ngưỡng quét `DEFAULT_BALL_THRESHOLDS`).

### `tracking_evaluation.py`

Metric tracking không phụ thuộc thư viện ngoài: precision/recall/AP50, IDF1, DetA/AssA/LocA kiểu HOTA, độ chính xác team/số áo, recall track bóng. `evaluate_tracking(gt, pred)`. Không phải GS-HOTA chính thức.

### `gold_annotations.py`

- Split: `HUMAN_TRAIN_MATCHES = (6, 8, 16, 21, 33, 48)`, `DEVELOPMENT_MATCHES = (18, 24, 47)`, `FROZEN_TEST_MATCHES = (0, 36, 50)`; `GOLD_SCHEMA_VERSION = "attack-gold-v2.0"`.
- Nhãn: `possession`, `attack`, `uncertain`, `dead_ball`; `SHOT_OUTCOMES = (goal, on_target, blocked, off_target, woodwork, unknown)`.
- `GoldSequenceAnnotationV2` (id `gold-sequence:…`), `GoldPeriodIntervalV2`, `read_gold_annotations_jsonl` (từ chối trận test nếu không có cờ), `append_gold_annotation_jsonl`, `append_sealed_test_annotation_jsonl`, `continuous_full_match_coverage_report` (frame bao gồm hai đầu, không hở, không chồng), `build_gold_manifest`, `gold_progress_report`, `freeze_sealed_test_annotations`, `verify_sealed_test_annotations`, `intra_rater_report` (kappa ≥ 0.8, terminal ≥ 0.85, median biên ≤ 0.5 s, p90 ≤ 1.0 s, match rate ≥ 0.9), `deterministic_reannotation_blocks` (20%, khối 300 s, rửa 14 ngày).

### `release_control.py`

- `ReleaseComponentHashes`, `FrozenAttackV2Release` (`attack-v2-frozen-release/v2`), `freeze_attack_v2_release` (chỉ trên development), `write_frozen_release_manifest` (tạo một lần), `load_frozen_release_manifest`.
- Ledger test cuối: `claim_final_test_once`, `complete_final_test_claim`, `final_test_ledger_status` (`available`, `claimed`, `completed`).
- `FINAL_TEST_EVALUATION_PROTOCOL`: tIoU (0.25, 0.5, 0.75), chính 0.5, bootstrap 2000 seed 42, bắt buộc baseline đóng băng.

---

## 9. Hạ tầng mô hình học (chưa nối vào luồng chính)

### `action_spotting_v2.py`

Hợp đồng cho action spotter tương lai: sampler cân bằng trận/lớp, cửa sổ 30 s chồng 50%, ghép logit trọng số tâm, `global_constrained_decode` (NMS 12 frame, loại trừ xung đột), khai thác hard negative, chọn và đóng băng operating point trên development (`action-operating-point-v2.0`), kế hoạch tìm kiếm tối đa 12 cấu hình, ràng buộc tài nguyên ≤ 5.5 GB VRAM.

### `temporal_model.py`

TCN giãn nở đa nhiệm (208 đặc trưng ở 5 Hz, 10 head) và decoder semi-Markov theo đoạn. `MODEL_VALIDATION_STATUS = "infrastructure_untrained"`; chọn checkpoint theo attack F1 trên dev người.

### `ml.py`, `training.py`, `yolo_dataset.py`, `soccernet.py`

- `ml.py`: xG-v1 (16 cột) và baseline hiệu quả V1, split nhóm 60/20/20, kiểm rò rỉ nhãn.
- `training.py`: `train_ball_model(...)` (Ultralytics, 1 lớp, từ chối CPU ở 1280 và ≥ 20 epoch nếu không có `allow_cpu`).
- `yolo_dataset.py`: `export_soccernet_ball_dataset(output, train_sequences, val_sequences, link_mode)` ra `images/`, `labels/`, `data.yaml`, `export-report.json`.
- `soccernet.py`: tải từng sequence GSR bằng HTTP range (curl), `analyze_gamestate_sequence` (tốc độ, chất lượng tọa độ, đánh giá attack heuristic).

---

## 10. Legacy V1

| Module | Nội dung |
| --- | --- |
| `segmentation.py` | `segment_attacks(events, SegmentationConfig(max_gap_sec=15))` |
| `scoring.py` | `evaluate_attack(sequence)`: điểm 0–100, nhãn `effective` ≥ 60, `promising` ≥ 45 |
| `metrics.py` | `compute_metrics`; `FINAL_THIRD_X = 70`, `PENALTY_BOX_X = 88.5` |
| `possession.py` | `assess_possession`: rubric `possession-attack-qualification-v1` |
| `attack_features.py` | `build_attack_features` |
| `annotations.py` | Mẫu phân tầng và kappa có trọng số cho nhãn hiệu quả V1 |

Các nhãn hiệu quả V1 **không** được dùng trong Attack-v2; v2 bắt buộc abstain.

---

## 11. Điểm vào

- `__init__.py` re-export các lớp/hàm chính của `attack_v2`, `models`, `research`, `scoring`, `segmentation`.
- `__main__.py` gọi `cli.main`, nên `py -3 -m attack_analysis <lệnh>` hoạt động.
- Entry point cài đặt: `attack-analysis = attack_analysis.cli:main`.
