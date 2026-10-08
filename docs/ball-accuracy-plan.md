# Kế hoạch nâng ball detector lên quality gate

## Tiêu chuẩn nghiệm thu

Model bóng chỉ được nối vào possession, shot và xG khi benchmark trên các sequence
SoccerNet không xuất hiện trong train trả:

| Metric | Ngưỡng |
|---|---:|
| IoU | `0.50` |
| Ball F1 | `>= 0.80` |
| Ball precision | `>= 0.75` |
| Ball recall | `>= 0.75` |
| Center error | theo dõi, mục tiêu `<= 5 px` |

Không chọn threshold trên test set. Threshold được chọn trên validation; test set
chỉ chạy một lần sau khi đã khóa model và cấu hình.

## Kết quả hiện tại

Benchmark 50 frame `SNGS-033`, frame gốc 575-624, IoU 0.5:

| Model | Best conf | Precision | Recall | F1 | Kết luận |
|---|---:|---:|---:|---:|---|
| YOLO26n COCO sports-ball | 0.15 | 0.381 | 0.320 | 0.348 | Reject |
| YOLO11n community ball-only | 0.30 | 0.500 | 0.260 | 0.342 | Reject |
| YOLOv8s community football | 0.02 | 0.077 | 0.060 | 0.067 | Reject |
| YOLOv9 community ball-only | 0.08 | 0.078 | 0.100 | 0.088 | Reject |

Cùng 50 frame, RTX 4050, `imgsz 1280`, `ball-conf 0.01`, không khóa threshold
(sweep `benchmark-ball`, chọn F1 cao nhất):

| Path | Best conf | Precision | Recall | F1 | Keypoint coverage | Kết luận |
|---|---:|---:|---:|---:|---:|---|
| Full-frame, không pose-crop | 0.10 | 0.333 | 0.360 | 0.346 | 0.334 | Reject |
| Tile 480 + pose-crop | 0.50 | 0.132 | 0.140 | 0.136 | 0.834 | Reject |

Pose-crop đưa keypoint coverage từ 0.334 lên 0.834. Tile bóng tăng false
positive (2874 box ở conf 0.01) nên F1 thấp hơn full-frame. Gate vẫn fail.
Vì vậy `annotate-video`, `run-pipeline` và Streamlit mặc định `--ball-tile-size 0`
(pose-crop vẫn bật); tile chỉ bật thủ công cùng model bóng fine-tune.
JSON: `runs/sngs033-bench/baseline.ball-benchmark.json` và
`tiled-pose.ball-benchmark.json`.

Fine-tune `yolo26s.pt`, `imgsz 1280`, batch 4, 3.000 frame train / 750 frame val.
Run đầu dừng ở epoch 31 (best epoch 14, val mAP50 0.373). Run nối tiếp từ
weight đó dừng sớm ở epoch 22 (patience 17; best epoch 5, val mAP50 0.342).
Sweep 50 frame, không khóa threshold:

| Checkpoint | Path | Best conf | Precision | Recall | F1 | Kết luận |
|---|---|---:|---:|---:|---:|---|
| Nối tiếp, best val | Tile | 0.20 | 0.692 | 0.360 | 0.474 | Reject |
| Nối tiếp, best val | Full-frame | 0.20 | 0.739 | 0.340 | 0.466 | Reject |
| Run 1, best val mAP | Tile | 0.20 | 0.682 | 0.300 | 0.417 | Reject |
| Run 1, best val mAP | Full-frame | 0.20 | 0.737 | 0.280 | 0.406 | Reject |

F1 cao nhất 0.474, vẫn dưới 0.80. Precision full-frame 0.739 sát ngưỡng 0.75
nhưng recall ~0.34. Checkpoint này không được nối vào possession, shot hay xG.

Các checkpoint community bị domain shift. Model-card metric không được dùng làm
metric luận văn. Report chi tiết nằm trong thư mục `derived` với hậu tố
`ball-benchmark.json`.

## Dataset đã chuẩn bị

Dataset `data/yolo-ball-v2` dùng class duy nhất `ball`:

| Split | Sequence | Images | Ball boxes | Median bbox |
|---|---|---:|---:|---:|
| Train | SNGS-170, SNGS-062, SNGS-097, SNGS-101 | 3.000 | 2.826 | 15x14 px |
| Validation | SNGS-033 | 750 | 716 | 10x10 px |

Split theo sequence, không split ngẫu nhiên theo frame, để các frame gần nhau
không rò rỉ từ train sang validation.

Export lại dataset:

```powershell
$CLI = ".\.venv\Scripts\attack-analysis.exe"
$ROOT = ".\data\soccernet-gs\gamestate-2024"

& $CLI soccernet-export-yolo .\data\yolo-ball-v2 `
  --train-sequence "$ROOT\train\SNGS-170" `
  --train-sequence "$ROOT\train\SNGS-062" `
  --train-sequence "$ROOT\train\SNGS-097" `
  --train-sequence "$ROOT\train\SNGS-101" `
  --val-sequence "$ROOT\valid\SNGS-033" `
  --link-mode hardlink
```

## Train đúng cấu hình

Bóng chỉ rộng 10-15 px trong ảnh 1920x1080. Không dùng smoke run `640` làm model
cuối. Cấu hình khởi đầu trên NVIDIA GPU:

```powershell
& $CLI train-ball-model `
  .\data\yolo-ball-v2\data.yaml `
  .\data\runs `
  --model .\data\models\yolo-football-ball-detection.pt `
  --epochs 100 `
  --imgsz 1280 `
  --batch 4 `
  --device 0 `
  --workers 4 `
  --name soccernet-ball-1280
```

Máy có RTX 4050 Laptop 6 GB và PyTorch `2.1.0+cu121` (`cuda=true` trên Python 3.11).
Fine-tune `imgsz=1280` dùng `--device 0`. Batch 4 trên 6 GB có thể sát VRAM; hết bộ nhớ thì hạ `--batch`.
Run smoke CPU trước đây đã dừng, không có checkpoint nửa chừng nào được coi là kết quả.

`train-ball-model` **từ chối** `imgsz=1280` trên CPU khi `epochs >= 20`, trừ khi
`--allow-cpu`. Fine-tune đạt gate chỉ chạy khi `doctor.tools.cuda_available=true`
với `--device 0` như lệnh GPU ở trên.

Path FOOTPASS + freeze HDF5 **không** chờ ball F1 0.80 (P9): xG location-only vẫn
score. Gate bóng chỉ chặn suy possession/shot **video-only**.

## Inference và benchmark

Chạy model riêng cho bóng, còn model người và pose giữ độc lập:

```powershell
$BEST = ".\data\runs\soccernet-ball-1280\weights\best.pt"
$S = ".\data\soccernet-gs\gamestate-2024\valid\SNGS-033"

& $CLI annotate-video "$S\derived\sngs-033-preview.mp4" `
  "$S\derived\sngs-033-finetuned-50f.mp4" `
  --observations "$S\derived\sngs-033-finetuned-50f.jsonl" `
  --ball-model $BEST `
  --ball-class-id 0 `
  --ball-conf 0.01 `
  --imgsz 1280 `
  --device 0 `
  --max-frames 50

& $CLI benchmark-ball "$S\derived\sngs-033-finetuned-50f.jsonl" `
  "$S\Labels-GameState.json" `
  --start-frame 575 `
  --output "$S\derived\sngs-033-finetuned-50f.ball-benchmark.json"
```

Inference ban đầu dùng confidence 0.01 để giữ candidate. `benchmark-ball` lọc lại
cùng JSONL ở nhiều threshold, chọn điểm F1 tốt nhất và trả `accepted`. Không tăng
độ chính xác bằng cách tự chọn vài frame đẹp.

## Vòng lặp tiếp theo nếu vẫn fail

1. Phân nhóm FN theo kích thước bóng, occlusion và camera shot.
2. Thêm hard negatives từ giày, tất, chấm sân và biển quảng cáo.
3. Export positive crops/overlapping tiles để bóng có kích thước hiệu dụng lớn hơn.
4. Chạy tiled inference và global NMS trên đúng cấu hình tile đã train.
5. Thêm temporal association sau detector, không dùng temporal fit để bịa bbox khi
   detector không có candidate.
6. Mở rộng validation/test bằng nhiều sequence và báo cáo bootstrap confidence interval.
