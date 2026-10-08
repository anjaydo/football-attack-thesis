# Dữ liệu, artifact, schema và cấu hình

## 1. Bố cục thư mục

```text
football-attack-thesis/
  src/attack_analysis/      package chính (66 module)
  app/streamlit_app.py      giao diện
  tests/                    50 file unittest
  runs/                     script thí nghiệm + output (gitignored)
  data/                     dữ liệu cục bộ (gitignored)
  models/                   registry, ngưỡng, model xG (joblib gitignored)
  docs/                     tài liệu
  examples/                 JSON mẫu cho CLI
  .cursor/skills/           3 skill dùng làm system instruction cho LLM
  build/                    bản build setuptools cũ, không dùng
../FOOTPASS/                dữ liệu và mã FOOTPASS (play-by-play, HDF5, evaluation.py)
```

## 2. Dữ liệu FOOTPASS (ngoài dự án)

| Đường dẫn | Nội dung |
| --- | --- |
| `FOOTPASS/playbyplay_GT/playbyplay_val.json` | Play-by-play oracle cho 6 hiệp val (game 18/24/47 × H1/H2) |
| `FOOTPASS/playbyplay_PRED/playbyplay_<ID>_val.json` | Dự đoán DST/TAAD (được registry ghim SHA-256) |
| `FOOTPASS/data/val_tactical_data.h5` | Tactical HDF5: vị trí, role, bbox mọi cầu thủ theo frame |
| `FOOTPASS/evaluation.py` | Evaluator chính thức (δ = 12 frame, τ = 0.15) |
| `D:\videos\game_N.mp4` | Video xử lý; Full HD qua `FOOTPASS_FULLHD_VIDEOS_DIR` |

## 3. `models/`

| File | Nội dung |
| --- | --- |
| `prediction-sources.json` | Registry `prediction-sources/1.0`: `DST_v1` primary (micro-F1 0.719), `TAAD_v1` (0.410) và `TAAD_h100` (0.346) ablation |
| `dst-thresholds-v1.json` | Ngưỡng DST theo lớp (`dst-thresholds-v1`), tune trên `game_18`, giữ ngoài `game_24/47`: block 0.65, cross 0.65, drive 0.8, header 0.5, pass 0.6, shot 0.35, tackle 0.45, throw_in 0.6. Micro-F1 6 hiệp: 0.7186 → 0.7286 |
| `xg-v2.joblib` / `xg-v2.metrics.json` | xG-v2 HistGradientBoosting, trạng thái `statsbomb_gate_passed_domain_unvalidated`, Brier khoảng 0.08–0.09 |
| `xg-v1.joblib` / `xg-v1.metrics.json` | xG-v1 legacy |

Registry ví dụ một mục:

```json
{
  "id": "DST_v1",
  "kind": "DST",
  "path": "../../FOOTPASS/playbyplay_PRED/playbyplay_DST_v1_val.json",
  "sha256": "34ef3f16...",
  "role": "primary",
  "official": {"micro_f1": 0.719, "precision": 0.735, "recall": 0.703, "delta": 12, "confidence_threshold": 0.15},
  "threshold_profile": "dst-thresholds-v1.json"
}
```

## 4. `data/` (cục bộ)

| Đường dẫn | Nội dung |
| --- | --- |
| `footpass-video-sync.json` | Offset preview/processing theo trận (`footpass-video-sync/1.0`); hiện chỉ có `game_18` |
| `attack-v2-development.jsonl`, `attack-v2-train.jsonl` | Nhãn gold người (chưa có) |
| `sealed-test/` | Nhãn test + niêm phong (chưa có) |
| `attack-v2-gold-manifest.json` | Manifest 12 trận |
| `benchmarks.jsonl` | Benchmark per-90 theo role |
| `xg-shots-v2.csv`, `xg-shots-360.csv` | Dataset xG từ StatsBomb |
| `yolo-ball-v2/` | Dataset YOLO bóng từ SoccerNet GS |
| `soccernet-gs/` | Sequence SoccerNet GameState đã giải nén |
| `runs/soccernet-ball-1280*` | Run fine-tune bóng (best F1 0.474 trên lát 50 frame) |

File sync mẫu:

```json
{
  "schema_version": "footpass-video-sync/1.0",
  "default_match_fps": 25,
  "matches": {
    "game_18": {"preview_offset_sec": 0.0, "processing_fps": 25.0, "preview_file": "game_18.mp4"}
  }
}
```

Tra cứu thử khóa `match_key` (ví dụ `game_18_H1`) trước, rồi `match_id`.

## 5. `runs/` (output)

| Đường dẫn | Nội dung |
| --- | --- |
| `window-cache/<sha video>/<key16>/` | Run YOLO cho cửa sổ FOOTPASS: `manifest.json`, `observations.jsonl`, `quality.json`, `annotated.mp4`, `annotated-browser.mp4`, sidecar legacy |
| `upload-cache/` | Clip tải lên và run vision |
| `verification-v2/` | `footpass-oracle-audit.json`, `prediction-source-comparison.json`, `attack-start-diagnosis.json`, `ingest-*.json` |
| `sngs033-bench/` | Benchmark bóng 50 frame SNGS-033 |

## 6. Schema version

| Chuỗi | Ở đâu |
| --- | --- |
| `2.0`, `attack-v2-rubric-1` | `models.py` (Attack-v2) |
| `attack-v2-run/2.0` | Đầu ra `footpass-ingest`, `sequences-v2.json` |
| `attack-run-v2.0` | `manifest.json` của pipeline video |
| `attack-analysis-manifest-v2`, `artifact-provenance-v2` | Provenance artifact |
| `upload-vision-v4` | Cache vision |
| `footpass-video-sync/1.0` | File sync video |
| `prediction-sources/1.0` | Registry nguồn dự đoán |
| `dst-thresholds-v1` | Hồ sơ ngưỡng event |
| `footpass-oracle-audit-v2` | Kiểm toán oracle |
| `footpass-terminal-inference-v1` | Suy luận terminal |
| `xg-v2.0`, `xg-features-v2`, `xg-training-v2` | xG-v2 |
| `attack-evidence-v2/1.0` | Evidence bundle |
| `attack-verdict-v2`, `attack-conclusion-v2/1.0` | Prompt và kết luận LLM |
| `attack-v2-research-overview/1.0` | Tổng quan research-only |
| `attack-sequence-evaluation/v2` | Báo cáo đánh giá sequence |
| `attack-gold-v2.0` | Nhãn gold |
| `attack-v2-frozen-release/v2`, `attack-v2-final-test-claim/v1`, `attack-v2-final-test-result/v1` | Release và ledger |
| `action-operating-point-v2.0`, `action-spotting-search-v2.0` | Hạ tầng action spotting |
| `attack-temporal-features-v2.0`, `attack-temporal-model-v2.0` | Hạ tầng temporal model |
| `footpass-window-event-stream-legacy-v1`, `possession-candidate-window-v1`, `PossessionCandidateV1` | Sidecar legacy |

## 7. Định dạng `observations.jsonl`

Mỗi dòng một frame:

```json
{"frame_index": 2640, "timestamp_sec": 105.6, "scene_cut": false, "calibration_confidence": 0.82,
 "objects": [
   {"track_id": "7", "object_type": "player", "confidence": 0.91,
    "bbox": {"x1": 812, "y1": 410, "x2": 850, "y2": 505},
    "anchor_image": [831, 503], "team": "team_a", "pitch_x": 88.2, "pitch_y": 30.1,
    "speed": 14.3, "speed_unit": "km/h",
    "keypoints": [[830, 420, 0.9, 88.1, 30.0]]},
   {"track_id": "1", "object_type": "ball", "confidence": 0.4,
    "bbox": {"x1": 900, "y1": 480, "x2": 908, "y2": 488}}
 ],
 "field_geometry": {"method": "opencv_hsv_hough_v1"}}
```

`frame_index` là tuyệt đối theo video (gồm `start_frame`). Keypoint 5 phần tử có thêm tọa độ sân.

## 8. Biến môi trường

| Biến | Mặc định | Dùng ở |
| --- | --- | --- |
| `FOOTPASS_VIDEOS_DIR` | `D:\videos` | `video_paths`, Streamlit |
| `FOOTPASS_FULLHD_VIDEOS_DIR` | (không) | `video_paths` |
| `FOOTPASS_VIDEO_SYNC_JSON` | `data/footpass-video-sync.json` | `video_paths` |
| `OPENAI_API_KEY` | (không) | `llm`, Streamlit |
| `OPENAI_MODEL` | `gpt-5.4-nano` | `env` |
| `BRAVE_SEARCH_API_KEY` | (không) | `web_search` |
| `XG_V2_MODEL_PATH` | `models/xg-v2.joblib` | `env` |

`.env` được đọc với `override=False` và bị `provenance.file_sha256` từ chối băm. Không in hoặc commit giá trị.

## 9. Extras cài đặt (`pyproject.toml`)

- Python ≥ 3.10; phụ thuộc lõi chỉ `numpy`.
- `video`: opencv-python, imageio-ffmpeg.
- `vision`: opencv-python, ultralytics, imageio-ffmpeg.
- `soccernet`: SoccerNet.
- `footpass`: h5py.
- `ml`: joblib, pandas, scikit-learn.
- `llm`: openai, python-dotenv.
- `demo`: streamlit, pandas, h5py, mplsoccer, openai, python-dotenv, imageio-ffmpeg.
- `research`: gộp `demo` + `ml` (không gồm vision).
