# Báo cáo tiến độ — 2026-10-06

Phạm vi: dự án `football-attack-thesis` (Attack-v2, release `research_only`): decoder possession-first, event FOOTPASS + tactical HDF5, overlay YOLO, xG-v2, Streamlit.

Tóm tắt nhanh:

- Đã xong: benchmark bóng SNGS-033, fine-tune model bóng, quy trình gold development, registry nguồn dự đoán (DST/TAAD) với gate thay model, chạy TAAD H100 local + ablation Attack-v2, dọn các vấn đề tồn đọng (tile bóng, thư mục staging, tài liệu cũ), chẩn đoán attack-start.
- Đang chờ: file dự đoán DST H100 (không can thiệp Droplet).
- Còn mở: gate bóng (F1 0.474 / 0.80), đồng bộ video cho 8 trận, nhãn gold do người gán (P11).
- Test: 332 OK, 1 skip (`$env:PYTHONPATH="src"; py -3 -m unittest discover -s tests -p "test_*.py"`).

---

## 1. Ràng buộc đã tuân thủ

- Không tác động Droplet (không SSH/scp/destroy).
- Không mở frozen test 0/36/50.
- Không tự tạo nhãn người.
- Không khóa ngưỡng bóng trên lát 50 frame.
- Không kết luận hiệu quả dạng categorical; release giữ `research_only`.
- Không lộ secret trong `.env`.

---

## 2. Vấn đề đã xử lý

### 2.1 Benchmark bóng trên SNGS-033 (`bench-sngs`)

**Vấn đề:** chưa có số đo ball F1 trên dữ liệu có nhãn chuẩn; gate là IoU 0.5, F1 ≥ 0.80, P và R ≥ 0.75.

**Cách xử lý:** script `runs/sngs033_bench.py` chạy trên 50 frame SNGS-033 với các cấu hình khác nhau; kết quả lưu ở `runs/sngs033-bench/*.json`.

| Cấu hình | Ball F1 | Ghi chú |
| --- | ---: | --- |
| COCO full-frame (baseline) | 0.346 | |
| COCO + tile 480 | 0.136 | Nhiều false positive |
| Pose-crop | — | Độ phủ keypoint 0.834 (tăng mạnh) |

### 2.2 Fine-tune model bóng (`finetune-ball`)

**Vấn đề:** model COCO gốc không đạt gate.

**Cách xử lý:**

- Tạo dataset `data/yolo-ball-v2` từ SoccerNet GS (train).
- Train yolo26s ở 1280: `data/runs/soccernet-ball-1280` (31 epoch, best epoch 14, mAP50 0.373). Train dừng im lặng ở epoch 31.
- `resume=True` nạp nhầm dataset fallback 4 ảnh nên đã dừng ngay. Thay vào đó chạy tiếp từ `last.pt` thành run mới `soccernet-ball-1280-cont` (`runs/resume_ball.py`).
- Benchmark lại: F1 tốt nhất 0.474 (P 0.69–0.74, R 0.34–0.36).

**Kết quả:** vẫn fail gate, nên model bóng **không** được nối vào possession/shot/xG. Số liệu nằm trong `docs/ball-accuracy-plan.md`.

### 2.3 Quy trình gold development (`gold-dev`)

**Vấn đề:** chưa có quy trình gán nhãn người cho các trận development/train.

**Cách xử lý:** quy trình và runbook (`docs/gold-annotation-runbook.md`, lệnh `gold-status`) đã sẵn sàng. Development là 18/24/47; train là 6, 8, 16, 21, 33, 48; frozen test 0/36/50 vẫn niêm phong.

### 2.4 Registry nguồn dự đoán (TAAD/DST)

**Vấn đề:** chọn file play-by-play dự đoán là thủ công, không truy vết được, dễ ghi đè, không có quy tắc khi thay model DST mới.

**Cách xử lý:**

- Module mới `src/attack_analysis/prediction_sources.py` và registry `models/prediction-sources.json`.
- Quy tắc:
  - Chỉ đúng một primary.
  - TAAD luôn là ablation.
  - DST chỉ được promote khi micro-F1 chính thức **lớn hơn hẳn** primary hiện tại.
  - File được ghim bằng SHA-256 và không bao giờ bị ghi đè.
- Chấm điểm bằng `FOOTPASS/evaluation.py` (τ = 0.15, δ = 12).
- CLI:
  - `footpass-ingest --prediction-source <id>`
  - `prediction-sources`
  - `swap-prediction-source ... [--dry-run]`
- Streamlit chọn nguồn từ registry, hiển thị id, sha256 và F1. Toggle class-aware dùng `threshold_profile` của nguồn đó.
- Test mới `tests/test_prediction_sources.py` (8 test). Tài liệu: `docs/prediction-sources.md`.

Trạng thái registry:

| Id | Vai trò | Micro-F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: |
| DST_v1 | primary | 0.719 | 0.735 | 0.703 |
| TAAD_v1 | ablation | 0.410 | 0.304 | 0.630 |
| TAAD_h100 | ablation | 0.346 | 0.244 | 0.599 |

### 2.5 Chạy TAAD H100 local và ablation Attack-v2

**Vấn đề:** cần đánh giá checkpoint TAAD tốt nhất từ H100 mà không đụng Droplet.

**Cách xử lý:**

- Script `FOOTPASS/scripts/local_taad_val.ps1` chạy TAAD trên 6 hiệp val, chuyển sang JSON và chấm bằng `evaluation.py`.
- Kết quả: TAAD_h100 đạt 0.346, **kém hơn** TAAD cũ (0.410), nên được đăng ký làm ablation.
- `runs/compare_prediction_sources.py` decode Attack-v2 cho từng nguồn và so với oracle (F1 ranh giới ±1 s):

| Nguồn | Possession | Attack | Possession start / end F1 | Attack start / end F1 |
| --- | ---: | ---: | ---: | ---: |
| Oracle | 755 | 46 | — | — |
| DST_v1 | 696 | 42 | 0.647 / 0.652 | 0.386 / 0.591 |
| TAAD_v1 | 1862 | 194 | 0.296 / 0.302 | 0.050 / 0.167 |
| TAAD_h100 | 2357 | 295 | 0.264 / 0.273 | 0.041 / 0.111 |

**Kết luận:** DST vẫn là nguồn tốt nhất cho Attack-v2. TAAD sinh quá nhiều possession và attack giả. Số liệu nằm trong `docs/action-model-card.md`.

### 2.6 Tile bóng 480 px làm overlay mặc định kém đi (P13, đã đóng)

**Vấn đề:** `annotate-video`, `run-pipeline` và Streamlit mặc định cắt frame thành ô 480 px để tìm bóng. Với model COCO, việc này hạ F1 từ 0.346 xuống 0.136.

**Cách xử lý:**

- `--ball-tile-size` mặc định là `0` (tắt) trong `vision_pipeline.py` và `cli.py`. Chỉ nên bật khi dùng model bóng đã fine-tune.
- Thông báo của `doctor` (`environment.py`) được sửa cho khớp.
- Khóa cache overlay đổi thành `fullhd_pose_crop_ball_fullframe_v6` (`video_paths.py`), nên overlay tile cũ sẽ được dựng lại.
- Test mới trong `tests/test_vision_refine.py`.

### 2.7 Thư mục staging thừa

**Vấn đề:** `FOOTPASS/playbyplay_PRED/_staging_h100` còn sót lại sau khi chạy TAAD.

**Cách xử lý:** đã xóa thư mục; `local_taad_val.ps1` giờ tự dọn staging sau khi chuyển kết quả.

### 2.8 Tài liệu lỗi thời

**Vấn đề:** `docs/implementation-status.md` vẫn ghi ball F1 0.348 và chưa nhắc tới registry.

**Cách xử lý:**

- Cập nhật F1 tốt nhất 0.474 và trạng thái registry.
- Đồng bộ `docs/ops-checklist.md`, `docs/pipeline-issues.md` (thêm P13–P15) và `docs/ball-accuracy-plan.md`.

### 2.9 Attack-start F1 thấp với DST (P15, đã chẩn đoán)

**Vấn đề:** attack-start F1 chỉ 0.386 dù dùng DST.

**Cách xử lý:** `runs/diagnose_attack_start.py` ghép attack oracle với attack DST có cú sút kết thúc trong ±1 s. Kết quả lưu ở `runs/verification-v2/attack-start-diagnosis.json`.

- Chỉ 26/46 attack oracle có cặp; DST **bỏ sót hẳn** 20 cú sút kết thúc.
- Trong 26 cặp, 16 có điểm bắt đầu lệch không quá 1 s; độ lệch trung vị −0.12 s.

**Kết luận:** lỗi nằm ở việc DST bỏ sót cú sút, không phải ở luật xác định điểm bắt đầu của decoder. Không chỉnh decoder, vì chỉnh theo các trận val sẽ làm rò dữ liệu đánh giá.

---

## 3. Vấn đề tồn đọng

| # | Vấn đề | Trạng thái | Vì sao chưa đóng / bước tiếp theo |
| --- | --- | --- | --- |
| 1 | DST H100 | Chờ file JSON | Droplet đang train (dataloader-bound). Khi có file: `swap-prediction-source --kind DST --id DST_h100 --dry-run`, rồi đăng ký với `--checkpoint`. Chỉ promote nếu micro-F1 > 0.719; khi đó refit `dst-thresholds-v2.json` bằng `calibrate-footpass-events --tuning-match game_18_H1 --tuning-match game_18_H2`, ingest lại và cập nhật model card. |
| 2 | Gate bóng | Fail (F1 0.474 / 0.80, R ~0.35) | Cần phân tích false negative theo kích thước bóng và che khuất, thêm sequence/hard negative (xem cuối `docs/ball-accuracy-plan.md`). Không khóa ngưỡng trên lát 50 frame. |
| 3 | Đồng bộ video (P14) | Mở | `data/footpass-video-sync.json` chỉ có `game_18`. 8 trận còn lại đang tạm dùng offset 0 s ở 25 fps, nên overlay có thể lệch event. Cần đo offset thật, không đoán. |
| 4 | Nhãn gold (P11) | Mở | `gold-status` chưa có nhãn cho dev 18/24/47 và 6 trận train. Cần người gán tay; sau đó mới kiểm định xG với chuyên gia. |
| 5 | xG-v2 | `research_only` | Chưa có kiểm định human-domain; phụ thuộc #4. |
| 6 | Ontology FOOTPASS | Giới hạn dữ liệu | Không có corner/FK/penalty trong event; dùng heuristic hình học có ghi provenance, không thay nhãn StatsBomb. |
| 7 | Calibration homography | Rủi ro vận hành | Vẫn fail nếu sync/HDF5 trống hoặc không qua quality gate; video-only polygon (P12) chỉ là baseline nghiên cứu. |

---

## 4. File chính đã thêm hoặc sửa

- Code:
  - `src/attack_analysis/prediction_sources.py` (mới)
  - `src/attack_analysis/cli.py`
  - `src/attack_analysis/vision_pipeline.py`
  - `src/attack_analysis/environment.py`
  - `src/attack_analysis/video_paths.py`
  - `app/streamlit_app.py`
- Dữ liệu và cấu hình: `models/prediction-sources.json`
- Test:
  - `tests/test_prediction_sources.py` (mới)
  - `tests/test_vision_refine.py`
- Script:
  - `runs/sngs033_bench.py`
  - `runs/finetune_ball.py`
  - `runs/train_ball_full.py`
  - `runs/resume_ball.py`
  - `runs/compare_prediction_sources.py`
  - `runs/diagnose_attack_start.py`
  - `FOOTPASS/scripts/local_taad_val.ps1`
- Tài liệu:
  - `docs/prediction-sources.md` (mới)
  - `docs/implementation-status.md`
  - `docs/pipeline-issues.md`
  - `docs/ops-checklist.md`
  - `docs/ball-accuracy-plan.md`
  - `docs/action-model-card.md`
  - `docs/research-pipeline.md`
  - `FOOTPASS/docs/h100-pipeline-progress.md`
