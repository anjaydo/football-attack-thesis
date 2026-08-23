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

Máy hiện tại không có CUDA. Smoke test CPU cho thấy một epoch `640` cần khoảng
25-40 phút; `1280` nhiều epoch sẽ tốn nhiều giờ. Run smoke đã được dừng, không có
checkpoint nửa chừng nào được coi là kết quả.

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
