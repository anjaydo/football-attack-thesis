# Tham chiếu CLI `attack-analysis`

Gọi bằng `attack-analysis <lệnh>` (sau `pip install -e .`) hoặc `py -3 -m attack_analysis <lệnh>` với `PYTHONPATH=src`.

- Mọi lệnh in kết quả JSON ra stdout (UTF-8, `indent=2`).
- Nếu đối số đầu không phải tên lệnh và không bắt đầu bằng `-`, CLI tự chèn `evaluate` (tương thích ngược).
- `--device` mặc định là `default_torch_device()` (`"0"` khi có CUDA), `--imgsz` mặc định là `default_inference_imgsz()` (1280 trên GPU, 960 trên CPU).

Hướng dẫn từng bước dạng tutorial: [../cli-guide.md](../cli-guide.md) (chỉ phủ các lệnh cơ bản).

## 1. Legacy và event

| Lệnh | Mục đích | Tham số chính (mặc định) |
| --- | --- | --- |
| `evaluate` | Chấm một attack V1 đã cắt sẵn | `input` |
| `analyze-events` | Cắt luồng event V1 rồi chấm từng attack | `input`, `--max-gap 15.0`, `--min-events 1` |
| `video-info` | Metadata video (OpenCV) | `input` |
| `quality-report` | Báo cáo chất lượng `observations.jsonl` | `input`, `--keypoint-conf 0.25` |
| `infer-events` | Suy possession/carry/pass/turnover/shot_candidate từ track đã hiệu chỉnh | `input`, `--output`, `--possession-distance 3.0`, `--carry-distance 3.0`, `--transfer-gap 2.0`, `--max-transfer-speed 45.0`, `--shot-window 1.5`, `--shot-min-x 70.0`, `--shot-min-progress 8.0`, `--shot-min-speed 8.0`, `--team-a-direction right`, `--team-b-direction left` |

## 2. Video và vision

| Lệnh | Mục đích | Tham số chính (mặc định) |
| --- | --- | --- |
| `annotate-video` | Vẽ skeleton, box, track id, tốc độ; xuất observations | `input`, `output`, `--pose-model yolo26n-pose.pt`, `--detection-model yolo26n.pt`, `--ball-model`, `--ball-class-id 32`, `--player-tracker botsort.yaml`, `--ball-tracker bytetrack.yaml`, `--device`, `--imgsz`, `--player-conf 0.2`, `--ball-conf 0.15`, `--keypoint-conf 0.25`, `--speed-smoothing 0.35`, `--roi`, `--no-team-classification`, `--team-min-tracks 4`, `--scene-cut-threshold 0.35`, `--no-scene-cut-reset`, `--calibration`, `--observations`, `--max-frames`, `--start-frame 0`, `--field-lines`, `--field-line-min-length 0.06`, `--field-line-max-segments 40`, `--filter-ball-by-roi`, `--no-pose-crop`, `--pose-crop-imgsz 320`, `--ball-tile-size 0` (tắt; với model COCO, tile làm F1 giảm còn 0.136), `--ball-tile-overlap 0.1`, `--ball-max-side 80.0` |
| `run-pipeline` (alias `analyze-clip`) | Annotate, cổng chất lượng, rồi tùy chọn suy event và decode Attack-v2 | `input`, `run_directory`, cùng nhóm cờ model/conf/ROI/calibration/field-lines/pose-crop/ball-tile như trên, thêm `--xg-model` |
| `field-geometry` | Mask sân, vạch, giao điểm cho một ảnh | `input`, `output_json`, `--overlay`, `--min-length 0.06`, `--max-segments 40` |
| `doctor` | Môi trường + vận hành video + tiến độ gold | (không có) |
| `inspect-video-sync` | Kiểm thư mục video, file sync, cache key, file thiếu | `--project-root` (mặc định cwd) |

`run-pipeline` ghi vào `run_directory`: `annotated.mp4`, `observations.jsonl`, `quality.json`, `manifest.json`, tùy chọn `calibration.json`; nếu qua cổng thì thêm `events.json`, `sequences-v2.json`, `possession-candidates-v1.json`. CLI luôn chạy với source tier `video_only`.

## 3. SoccerNet và model bóng

| Lệnh | Mục đích | Tham số chính (mặc định) |
| --- | --- | --- |
| `soccernet-download` | Tải một sequence GameState | `output_directory`, `--split valid` (train/valid/test/challenge), `--sequence first` |
| `soccernet-analyze` | Phân tích nhãn GT và chuyển động | `sequence_directory`, `--fps`, `--max-speed 45.0`, `--pitch-margin 5.0` |
| `evaluate-observations` | So box observations với GT | `observations`, `labels`, `--start-frame 1`, `--iou 0.5` |
| `benchmark-ball` | Quét ngưỡng bóng và áp gate | `observations`, `labels`, `--start-frame 1`, `--iou 0.5`, `--threshold` (lặp được), `--min-f1 0.80`, `--min-precision 0.75`, `--min-recall 0.75`, `--output` |
| `soccernet-export-yolo` | Xuất dataset YOLO 1 lớp bóng | `output_directory`, `--train-sequence` (lặp, bắt buộc), `--val-sequence` (lặp, bắt buộc), `--link-mode hardlink` |
| `train-ball-model` | Fine-tune detector bóng | `data_yaml`, `output_directory`, `--model yolo26s.pt`, `--epochs 100`, `--imgsz 1280`, `--batch 4`, `--device`, `--workers 0`, `--name soccernet-ball`, `--allow-cpu` |

## 4. FOOTPASS

| Lệnh | Mục đích | Tham số chính (mặc định) |
| --- | --- | --- |
| `footpass-ingest` | Một hiệp FOOTPASS → possession V2 + attack; kèm V1 legacy. Chuỗi kết thúc bằng event `shot` được chấm xG-v2 StatsBomb | `playbyplay` (tùy chọn), `match_key`, `--source oracle` (oracle/predicted), `--prediction-source <id>` (ép predicted), `--prediction-registry`, `--tactical-h5`, `--fps 25.0`, `--min-confidence 0.15`, `--model-name`, `--event-threshold-profile`, `--max-gap 8.0`, `--max-duration 45.0` (legacy), `--min-events 1` (legacy), `--xg-model` (mặc định `models/xg-v2.joblib`), `--output` |
| `audit-footpass-v2` | Kiểm toán oracle toàn bộ (hình học + bất biến) | `playbyplay`, `tactical_h5`, `video_root`, `output` |
| `evaluate-footpass-actions` | Action spotting nghiêm ngặt (trận, đội, số áo, lớp) | `ground_truth`, `predictions`, `--fps 25.0`, `--delta 12`, `--min-confidence 0.15`, `--model-name predicted`, `--event-threshold-profile` |
| `calibrate-footpass-events` | Fit ngưỡng theo lớp trên trận tuning | `ground_truth`, `predictions`, `output`, `--tuning-match` (lặp, bắt buộc), `--fps 25.0`, `--delta 12`, `--baseline-threshold 0.15` |
| `prediction-sources` | Liệt kê registry | `--registry` |
| `swap-prediction-source` | Chấm bằng evaluator chính thức rồi đăng ký; chỉ lên primary nếu micro-F1 lớn hơn hẳn | `predictions`, `--kind TAAD\|DST` (bắt buộc), `--id` (bắt buộc), `--registry`, `--footpass-root`, `--ground-truth`, `--checkpoint`, `--threshold-profile`, `--note`, `--dry-run` |
| `build-footpass-benchmarks` | Benchmark per-90 theo role từ HDF5 | `tactical_h5`, `output`, `--fps 25.0`, `--season 2023-24`, `--snapshot-date` |

`footpass-ingest` cần đúng một trong hai: `playbyplay` hoặc `--prediction-source`. Đầu ra có `schema_version = "attack-v2-run/2.0"`, `source_tier`, `prediction_source` (id, kind, role, sha256, official_micro_f1), `summary`, `observed_actions`, possession/attack, `xg: {status: withheld}`, `legacy_compatibility`.

Ví dụ:

```powershell
$env:PYTHONPATH="src"
py -3 -m attack_analysis footpass-ingest game_18_H1 --prediction-source DST_v1 `
  --tactical-h5 ..\FOOTPASS\data\val_tactical_data.h5 `
  --event-threshold-profile models\dst-thresholds-v1.json --output runs\ingest.json
py -3 -m attack_analysis swap-prediction-source path\to\playbyplay_DST_h100_val.json --kind DST --id DST_h100 --dry-run
```

## 5. Attack-v2 release

| Lệnh | Mục đích | Tham số chính (mặc định) |
| --- | --- | --- |
| `evaluate-attack-v2` | So attack dự đoán với gold liên tục (tIoU, một-một) | `ground_truth`, `predictions`, `--output`, `--bootstrap-samples 2000`, `--seed 42`, `--allow-frozen-test`, `--release-manifest`, `--final-test-ledger`, `--frozen-baseline` |
| `freeze-attack-v2-release` | Chọn operating point trên dev và đóng băng các thành phần | `release_id`, `candidates`, `config`, `model`, `schema`, `rubric`, `development_dataset`, `frozen_baseline`, `output` |
| `verify-attack-v2-release` | Kiểm self-hash của manifest | `manifest` |
| `attack-v2-final-test-status` | Đọc ledger test cuối (không thay đổi) | `manifest`, `ledger` |

Chạy test cuối cần đủ `--allow-frozen-test --release-manifest --final-test-ledger --frozen-baseline`, bootstrap đúng 2000 với seed 42, và file GT tên `attack-v2-frozen-test.jsonl` đã niêm phong. Claim được ghi trước khi đọc nhãn; không chạy lại được.

## 6. Gold annotation

| Lệnh | Mục đích | Tham số chính (mặc định) |
| --- | --- | --- |
| `build-annotations` | Manifest nhãn V1 phân tầng (legacy) | `input`, `output`, `--sample-size 300`, `--seed 42` |
| `build-gold-manifest` | Manifest 12 trận (băm video) | `video_root`, `output`, `--sealed-directory` |
| `validate-gold-annotations` | Kiểm JSONL gold | `input`, `--allow-frozen-test`, `--period-intervals`, `--annotation-round 1` |
| `gold-intra-rater` | So hai vòng gán nhãn | `first_round`, `second_round`, `--allow-frozen-test` |
| `gold-status` | Tiến độ 12 trận, không mở nhãn test | `--project-root`, `--sealed-directory` |
| `seal-gold-test` | Niêm phong file test (không đảo ngược) | `sealed_directory`, `--period-intervals` (bắt buộc) |
| `verify-gold-test-seal` | Kiểm niêm phong | `sealed_directory` |

## 7. xG, evidence, phục vụ

| Lệnh | Mục đích | Tham số chính (mặc định) |
| --- | --- | --- |
| `build-xg-dataset` | CSV xG-v1 từ StatsBomb 360 | `open_data_root`, `output` |
| `build-xg-v2-dataset` | CSV xG-v2 có missingness | `open_data_root`, `output` |
| `train-xg` | Train xG-v1 (split theo trận) | `input`, `output`, `--match-column match_id`, `--target goal`, `--seed 42` |
| `train-xg-v2` | Train xG-v2 (CV lồng nhóm, calibration tách, test đóng băng) | `input`, `output`, `--match-column match_id`, `--target goal`, `--seed 42`, `--inner-folds 3`, `--bootstrap-samples 2000` |
| `train-effectiveness` | Baseline hiệu quả V1 (legacy) | `input`, `output`, `--seed 42` |
| `evaluate-evidence` | Bundle + verdict offline tất định | `input`, `--events-ready`, `--tracking-ready`, `--calibration-ready` |
| `build-historical-snapshot` | JSONL thống kê lịch sử đóng băng | `input`, `output`, `--snapshot-date` (bắt buộc), `--season 2023-24`, `--source-url` (bắt buộc) |
| `attack-v2-overview` | Tổng quan research-only, không verdict | `input`, `--xg-record`, `--benchmarks`, `--query`, `--attack-id` |
| `serve-demo` | Chạy Streamlit | `--port 8501` |
| `serve-mcp` | Server MCP qua stdio | `--bundle`, `--benchmarks` |
