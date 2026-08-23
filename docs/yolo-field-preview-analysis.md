# Thử nghiệm YOLO và hình học sân trên SoccerNet

## Mục tiêu

Phép thử này kiểm tra trực tiếp trên sequence SoccerNet-GSR `SNGS-033` liệu
pipeline có thể tạo player/ball bounding box, track ID, skeleton, vận tốc ảnh,
pitch ROI, line segment và intersection hay không.

Đây là smoke test 50 frame, tương đương 2 giây, để xác minh tích hợp; chưa phải benchmark học thuật.
Muốn báo cáo accuracy cần tập nhãn ground truth, protocol train/validation/test và
các metric như mAP, HOTA/IDF1, PCK và calibration reprojection error.

## Runtime

```text
Python       3.14.3
OpenCV       5.0.0
PyTorch      2.13.0+cpu
Ultralytics  8.4.126
CUDA         false
Input        1920x1080, 25 FPS
Inference    imgsz=1280, CPU
```

## Command tái lập

Chạy từ project root:

```powershell
$env:PYTHONPATH = "C:\Users\AnDVM\Documents\Codex\2026-08-22\to\work\vision_runtime;src"
$SNGS = "data\soccernet-gs\gamestate-2024\valid\SNGS-033\derived"

python -m attack_analysis annotate-video `
  "$SNGS\sngs-033-preview.mp4" `
  "$SNGS\sngs-033-yolo-improved-v2-50f.mp4" `
  --observations "$SNGS\sngs-033-yolo-improved-v2-50f.jsonl" `
  --field-lines `
  --device cpu `
  --imgsz 1280 `
  --player-conf 0.10 `
  --ball-conf 0.15 `
  --max-frames 50 `
  --no-team-classification

python -m attack_analysis quality-report `
  "$SNGS\sngs-033-yolo-improved-v2-50f.jsonl"

python -m attack_analysis evaluate-observations `
  "$SNGS\sngs-033-yolo-improved-v2-50f.jsonl" `
  "data\soccernet-gs\gamestate-2024\valid\SNGS-033\Labels-GameState.json" `
  --start-frame 575 `
  --iou 0.5
```

## Kiến trúc đã thử

```text
frame
  -> YOLO Detect person -> BoT-SORT -> player bbox + track ID
  -> YOLO Pose person   -> IoU match -> skeleton/keypoints nếu đủ rõ
  -> YOLO Detect ball   -> ByteTrack -> ball candidate + track ID
  -> HSV grass contour  -> automatic pitch ROI
  -> white mask + Hough -> line segments + intersections
  -> anchors over time  -> px/s
  -> JSONL + annotated MP4
```

Việc tách player detect khỏi pose rất quan trọng. Trên một frame đối chứng,
`yolo26n.pt` bắt 14 person trong khi `yolo26n-pose.pt` chỉ bắt 5. Nếu dùng bbox
của pose làm player list, phần lớn cầu thủ xa sẽ biến mất.

## Kết quả

| Metric | Kết quả |
|---|---:|
| Frame xử lý | 50 |
| Player detections | 632 |
| Trung bình player/frame | 12.64 |
| Unique player tracks | 22 |
| Player tracking coverage | 100% |
| Keypoint coverage | 36.21% |
| Ball candidates | 42 |
| Unique ball tracks | 1 |
| Ball tracking coverage | 7.14% |
| Field geometry frame coverage | 100% |
| Line segments/frame | 8.76 |
| Intersections/frame | 0.86 |
| Quality score | 68.5, `warn` |

Ground-truth evaluation tại IoU 0.5:

| Model | Person F1 | Ball precision | Ball recall | Ball F1 |
|---|---:|---:|---:|---:|
| YOLO26n cũ, `ball-conf=0.03`, ball bị auto ROI lọc | 0.726 | 0.087 | 0.260 | 0.130 |
| YOLO26n v2, `ball-conf=0.15`, không auto-filter bóng | 0.721 | 0.381 | 0.320 | 0.348 |
| YOLO26s mới | 0.741 | 0.395 | 0.300 | 0.341 |

Quality gate chính thức của project là ball F1 `>= 0.80`, precision/recall đều
`>= 0.75` tại IoU 0.5. Vì vậy toàn bộ kết quả trong bảng đều là baseline bị loại,
không được dùng để suy luận possession, shot hay xG.

`yolo26s` chỉ tăng nhẹ trên đủ 50 frame nhưng gần gấp đôi runtime CPU, nên
`yolo26n` vẫn là mặc định. Dùng `--detection-model yolo26s.pt` khi ưu tiên accuracy.

Trong ablation 10-frame trước đó, đổi từ pose bbox sang detect bbox làm số player
detections tăng từ 17 lên 78. Bản 50-frame tiếp tục cho thấy skeleton có trên các
cầu thủ đủ lớn; cầu thủ xa vẫn giữ bbox/ID nhưng không giả tạo keypoint.

## Điều đã dùng được

- Player bbox và track ID làm dữ liệu quan sát ban đầu.
- Skeleton dưới dạng trường tùy chọn, không bắt buộc mọi player phải có.
- Grass contour làm pitch ROI tự động để bỏ phần lớn khán đài.
- Halfway line, center-circle chords và intersection làm baseline geometry.
- JSONL chứa dữ liệu máy đọc được thay vì chỉ có video overlay.

## Điều chưa được dùng cho kết luận bóng đá

- Ball F1 đã tăng từ 0.13 lên 0.33 nhưng recall mới đạt 30%. Detector vẫn bám vào
  một số giày/chi tiết áo; không dùng trực tiếp để suy luận possession, pass, shot
  hoặc xG cho tới khi temporal track được xác nhận.
- Vận tốc hiện là `px/s` và chứa cả camera pan/zoom. Không diễn giải thành vận tốc
  cầu thủ; chỉ xuất `km/h` sau per-frame calibration/camera compensation.
- Erosion của pitch mask đã loại false line ở advertising board trên frame kiểm
  tra; Hough line vẫn chưa biết semantic class nên chưa đủ dựng homography chuẩn.
- 50 frame vẫn quá ngắn để kết luận độ bền track qua occlusion hoặc camera cut.

## Bước nghiên cứu tiếp theo

1. Fine-tune detector bóng trên crop độ phân giải cao, dùng temporal candidate
   association và negative samples từ tất/đường sân/biển quảng cáo.
2. Huấn luyện semantic segmentation hoặc keypoint model cho các mốc sân, sau đó
   fit mô hình sân chuẩn và ước lượng homography theo từng frame.
3. Thêm camera-motion compensation trước khi tính vận tốc và dùng pitch-space
   smoothing để giảm ID jitter.
4. Chạy benchmark trên toàn bộ sequence có ground truth và báo cáo metric theo
   khoảng kích thước object, mức che khuất và camera shot.
