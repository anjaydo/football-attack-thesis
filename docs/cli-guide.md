# Hướng dẫn CLI từ đầu đến cuối

Tài liệu này tổng hợp toàn bộ CLI hiện có và thứ tự chạy khuyến nghị trên
Windows PowerShell. Các lệnh được chạy từ thư mục gốc của project.

## 1. Mở project

```powershell
Set-Location "C:\Users\AnDVM\Documents\Codex\2026-08-22\to\outputs\football-attack-thesis"
```

Kiểm tra Python:

```powershell
python --version
```

Project yêu cầu Python 3.10 trở lên. Máy hiện tại đang dùng Python 3.14.

## 2. Tạo môi trường ảo

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Tài liệu gọi trực tiếp Python/CLI trong `.venv`, vì vậy không cần chạy lệnh
`Activate.ps1` và không bị ảnh hưởng bởi PowerShell execution policy.

## 3. Cài core package

Core package đủ để chạy schema, metric, scoring và attack segmentation:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Đặt biến giúp các lệnh sau ngắn hơn:

```powershell
$CLI = ".\.venv\Scripts\attack-analysis.exe"
```

Kiểm tra CLI:

```powershell
& $CLI --help
```

CLI hiện có mười lăm command:

| Command | Chức năng |
|---|---|
| `evaluate` | Chấm điểm một đợt tấn công đã được phân đoạn |
| `analyze-events` | Tách event stream thành các đợt tấn công rồi chấm từng đợt |
| `video-info` | Đọc metadata của video bằng OpenCV |
| `annotate-video` | Chạy YOLO pose/detect/tracking và tạo video overlay |
| `quality-report` | Validate JSONL và đo coverage của detection/tracking/pose |
| `infer-events` | Suy luận possession/carry/pass/turnover từ calibrated tracks |
| `run-pipeline` | Chạy video→quality→events→attacks trong một command |
| `doctor` | Kiểm tra Python, package vision, CUDA, FFmpeg, sync video và tiến độ gold |
| `gold-status` | Tiến độ nhãn 12 trận, không đọc frozen test (P11) |
| `inspect-video-sync` | Dual-video roots, sync JSON, overlay cache key, file thiếu |
| `soccernet-download` | Tải riêng một sequence GameState từ archive SoccerNet bằng HTTP Range |
| `soccernet-analyze` | Phân tích ground-truth track, vai trò, đội, tọa độ sân và vận tốc |
| `field-geometry` | Trích pitch mask, line segments và intersections từ một frame |
| `evaluate-observations` | Đo precision/recall/F1 bbox theo ground truth SoccerNet |
| `benchmark-ball` | Quét confidence và bắt buộc ball detector qua accuracy gate |
| `soccernet-export-yolo` | Export nhãn bóng SoccerNet thành dataset YOLO một lớp |
| `train-ball-model` | Fine-tune YOLO detector bóng trên dataset đã export |

## 4. Chạy unit tests

Đây là bước smoke test nên chạy trước mọi thử nghiệm video:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kết quả hiện tại phải là `Ran 53 tests` và `OK`.

Kiểm tra môi trường trước khi xử lý video:

```powershell
& $CLI doctor
```

## 5. Thử `evaluate`

Command này nhận một JSON chứa đúng một `AttackSequence`, tính metric và trả về
điểm hiệu quả, nhãn cùng explanation.

Pha tấn công hiệu quả:

```powershell
& $CLI evaluate examples\attack_effective.json
```

Kết quả chính dự kiến:

```text
score: 88.6
is_effective: true
label: effective
```

Pha tấn công không hiệu quả:

```powershell
& $CLI evaluate examples\attack_ineffective.json
```

Lệnh tương thích cũ sau đây cũng có cùng tác dụng:

```powershell
& $CLI examples\attack_effective.json
```

## 6. Thử `analyze-events`

Command này nhận event stream chứa nhiều đội, tách các attack theo thay đổi đội,
terminal event hoặc khoảng cách thời gian, rồi chấm từng attack.

```powershell
& $CLI analyze-events examples\match_events.json
```

Ví dụ hiện tại tạo hai attack:

```text
Team A: 81.6/100, effective
Team B: 2.6/100, ineffective
```

Thay timeout và số event tối thiểu:

```powershell
& $CLI analyze-events examples\match_events.json `
  --max-gap 10 `
  --min-events 2
```

## 7. Cài dependency video và YOLO

`video-info` chỉ cần extra `video`. `annotate-video` cần extra `vision`, bao gồm
NumPy, OpenCV, Ultralytics và các dependency inference liên quan.

Để chạy toàn bộ pipeline, chỉ cần cài `vision`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[vision]"
```

Lần chạy YOLO đầu tiên cần internet để tải model weights. Việc cài Torch và
Ultralytics có thể tốn thời gian và dung lượng.

Kiểm tra import:

```powershell
.\.venv\Scripts\python.exe -c "import cv2, torch, ultralytics; print('OpenCV', cv2.__version__); print('Torch', torch.__version__); print('CUDA', torch.cuda.is_available()); print('Ultralytics', ultralytics.__version__)"
```

## 8. Chọn video thử nghiệm

### Tải sequence SoccerNet chính thức để thử nghiệm

SoccerNet Game State Reconstruction (GSR) gồm các sequence broadcast khoảng 30
giây. Command dưới đây đọc mục lục `valid.zip` trên máy chủ chính thức nhưng chỉ
tải byte range của sequence đầu tiên, thay vì tải toàn bộ archive 11.17 GB:

```powershell
& $CLI soccernet-download .\data\soccernet-gs --split valid --sequence first
```

Với bản phát hành hiện tại, sequence đầu theo thứ tự archive là `SNGS-033`. Nó
gồm `Labels-GameState.json` và 750 frame JPG. Downloader kiểm tra kích thước và
CRC của từng member trước khi ghi. Chạy lại command sẽ dùng cache đã có.

Phân tích ground truth:

```powershell
$SNGS = ".\data\soccernet-gs\gamestate-2024\valid\SNGS-033"
& $CLI soccernet-analyze $SNGS
```

Report được lưu tại `$SNGS\analysis-report.json`. Các trường chính gồm số frame,
annotation/track, role, team, jersey number, pitch-coordinate coverage và vận tốc
theo role. Vận tốc này được suy ra từ ground-truth pitch coordinates, không phải
kết quả YOLO của project.

Mặc định analyzer lấy FPS từ metadata, chỉ giữ speed segment có hai đầu nằm trong
sân cộng margin 5 m, và đánh dấu speed trên 45 m/s là outlier. Report vẫn giữ
`raw_max_mps`, số mẫu ngoài sân và số mẫu bị loại để audit. Có thể thay threshold:

```powershell
& $CLI soccernet-analyze $SNGS --max-speed 45 --pitch-margin 5
```

GSR sequence không phải một trận 90 phút. Video full-match (`1_224p.mkv`,
`1_720p.mkv`, ...) yêu cầu SoccerNet NDA và password riêng từ nhóm tác giả. Không
ghi password vào Git; sau khi được cấp quyền, đặt nó trong biến môi trường hoặc
secret manager.

`soccernet-analyze` suy luận đội tấn công từ cầu thủ gần bóng ở đầu clip, dùng
hướng đội `left/right` của GSR để đo progression, final-third/box entry và kết hợp
`action_class` để tạo kết luận `provisional_*`. Kết luận luôn ghi caveat và trả
`insufficient_context` nếu không đủ ball/team trajectory; đây chưa phải xG model
được huấn luyện.

Nên dùng clip broadcast dài 10-30 giây, độ phân giải ít nhất 720p, không chứa
replay/camera cut ở lần chạy đầu tiên.

Khai báo đường dẫn tuyệt đối:

```powershell
$VIDEO = "C:\path\to\football-clip.mp4"
$RUN = ".\runs\first-video"
New-Item -ItemType Directory -Force -Path $RUN
```

## 9. Chạy `video-info`

### Kiểm tra line sân trên một frame

```powershell
& $CLI field-geometry "$SNGS\img1\000600.jpg" `
  "$SNGS\derived\field-geometry-000600.json" `
  --overlay "$SNGS\derived\field-geometry-000600-overlay.jpg"
```

Baseline dùng HSV để tách connected grass region lớn nhất, lấy white pixels nằm
trong vùng sân, sau đó dùng probabilistic Hough transform để tạo line segments
và giao điểm. Đây là classical CV baseline, không phải semantic line classifier.

Command này chưa chạy YOLO. Nó kiểm tra OpenCV có mở được file và in width,
height, FPS, frame count cùng duration.

```powershell
& $CLI video-info $VIDEO
```

Nếu bước này lỗi thì cần xử lý codec/path trước khi chạy model.

## 10. Chạy thử 300 frame đầu

Máy hiện tại không phát hiện NVIDIA GPU, vì vậy nên thử ngắn trên CPU trước:

```powershell
& $CLI annotate-video $VIDEO "$RUN\annotated-preview.mp4" `
  --observations "$RUN\observations-preview.jsonl" `
  --field-lines `
  --roi examples\roi.example.json `
  --device cpu `
  --max-frames 300
```

Trước khi dùng `--roi`, sửa polygon trong `examples\roi.example.json` theo biên
mặt sân nhìn thấy trong clip. Nếu chưa có ROI đúng, bỏ option này; dùng ROI mẫu
sai kích thước sẽ lọc nhầm cầu thủ.

Mặc định command sử dụng:

| Thành phần | Giá trị mặc định |
|---|---|
| Player detection | `yolo26n.pt`, COCO class `person` |
| Player pose | `yolo26n-pose.pt`, ghép với player bbox bằng IoU |
| Ball detection | `yolo26n.pt`, COCO class `sports ball` |
| Player tracker | `botsort.yaml` |
| Ball tracker | `bytetrack.yaml` |
| Inference size | `960` |
| Player confidence | `0.20` |
| Ball confidence | `0.15` |
| Keypoint confidence | `0.25` |

Output gồm:

- `annotated-preview.mp4`: skeleton, player/ball box, track ID và speed;
- `observations-preview.jsonl`: một JSON record cho mỗi frame;
- mỗi record có thêm `field_geometry` khi bật `--field-lines`;
- summary JSON in ra terminal.

Kiểm tra hai record JSONL đầu tiên:

```powershell
Get-Content "$RUN\observations-preview.jsonl" -TotalCount 2
```

Chạy quality gate ngay sau preview:

```powershell
& $CLI quality-report "$RUN\observations-preview.jsonl"
```

Report gồm `quality_score`, trạng thái `pass/warn/fail`, ball frame coverage,
player/tracking coverage, keypoint coverage, field-geometry coverage và cảnh báo
tốc độ bất thường. Không nên chuyển sang event/xG nếu report đang `fail`.

Nếu clip lấy từ SoccerNet-GSR, đo accuracy bbox bằng ground truth:

```powershell
& $CLI evaluate-observations "$RUN\observations-preview.jsonl" `
  "$SNGS\Labels-GameState.json" `
  --start-frame 575 `
  --iou 0.5
```

Command trả TP/FP/FN, precision, recall, F1, matched IoU và center error riêng cho
`person` và `ball`. `quality-report` chỉ đo sức khỏe pipeline; false positive cũng
có thể làm frame coverage tăng nên nó không thay thế ground-truth evaluation.

Quét threshold và áp quality gate trước khi dùng bóng cho possession/shot/xG:

```powershell
& $CLI benchmark-ball "$RUN\observations-preview.jsonl" `
  "$SNGS\Labels-GameState.json" `
  --start-frame 575 `
  --iou 0.5 `
  --min-f1 0.80 `
  --min-precision 0.75 `
  --min-recall 0.75 `
  --output "$RUN\ball-benchmark.json"
```

Chỉ tiếp tục downstream khi report có `accepted: true`. F1 `0.3` không phải mức
chấp nhận; nó chỉ là baseline thất bại để so sánh thí nghiệm.

Không có calibration, vận tốc được ghi bằng `px/s`. Đây là tốc độ trong ảnh,
không phải tốc độ vật lý của cầu thủ.

### Pipeline một lệnh

Thay cho việc gọi từng command thủ công:

```powershell
& $CLI run-pipeline $VIDEO "$RUN\pipeline-preview" `
  --roi examples\roi.example.json `
  --device cpu `
  --max-frames 300
```

Nếu có calibration:

```powershell
& $CLI run-pipeline $VIDEO "$RUN\pipeline-calibrated" `
  --roi examples\roi.example.json `
  --calibration "$RUN\calibration.json" `
  --team-a-direction right `
  --team-b-direction left `
  --device cpu `
  --max-frames 300
```

Thư mục run chứa `annotated.mp4`, `observations.jsonl`, `quality.json` và
`manifest.json`. Khi quality/calibration đủ điều kiện, nó có thêm `events.json`
và `attacks.json`. Manifest ghi rõ bước downstream bị skip và lý do nếu có.

## 11. Chạy toàn bộ video

Sau khi preview đúng, bỏ `--max-frames`:

```powershell
& $CLI annotate-video $VIDEO "$RUN\annotated-full.mp4" `
  --observations "$RUN\observations-full.jsonl" `
  --device cpu
```

Nếu máy có CUDA/NVIDIA GPU:

```powershell
& $CLI annotate-video $VIDEO "$RUN\annotated-full.mp4" `
  --observations "$RUN\observations-full.jsonl" `
  --device 0
```

## 12. Chạy với calibration

File `examples\calibration.example.json` chứa bốn cặp điểm ảnh và tọa độ sân.
Phải thay `image_points` bằng bốn landmark thực tế nhìn thấy trong video.

```powershell
Copy-Item examples\calibration.example.json "$RUN\calibration.json"
```

Sau khi sửa điểm calibration:

```powershell
& $CLI annotate-video $VIDEO "$RUN\annotated-calibrated.mp4" `
  --observations "$RUN\observations-calibrated.jsonl" `
  --calibration "$RUN\calibration.json" `
  --device cpu `
  --max-frames 300
```

Khi có calibration, output có `pitch_x`, `pitch_y` và speed `km/h`.

Homography bốn điểm hiện tại chỉ đáng tin với camera cố định. Video broadcast có
pan/tilt/zoom cần dynamic camera calibration theo từng frame; không nên dùng một
file calibration tĩnh cho cả clip rồi coi vận tốc là ground truth.

## 12.1. Suy luận event từ calibrated tracks

Chỉ chạy bước này khi JSONL có `pitch_x/pitch_y` và team assignment đủ tốt:

```powershell
& $CLI infer-events "$RUN\observations-calibrated.jsonl" `
  --output "$RUN\events.json" `
  --team-a-direction right `
  --team-b-direction left
```

Baseline xác định người gần bóng nhất trong bán kính mặc định 3 m rồi suy luận
`possession_start`, `carry`, `pass` và `turnover`. Sau đó tách attack và chấm điểm:

```powershell
& $CLI analyze-events "$RUN\events.json"
```

Điều chỉnh threshold:

```powershell
& $CLI infer-events "$RUN\observations-calibrated.jsonl" `
  --output "$RUN\events.json" `
  --possession-distance 3.0 `
  --carry-distance 3.0 `
  --transfer-gap 2.0
  --max-transfer-speed 45.0
```

`infer-events` có shot candidate bảo thủ và geometry-only xG với
`xg_source=heuristic_geometry_v1`; chưa nhận biết goal hoặc outcome chính xác.

## 13. Các tùy chọn YOLO thường dùng

Xem toàn bộ option:

```powershell
& $CLI annotate-video --help
```

Tham số đầy đủ:

| Tham số | Ý nghĩa |
|---|---|
| `input` | Video đầu vào |
| `output` | Video MP4 đã vẽ overlay |
| `--pose-model` | Model pose dùng cho người, mặc định `yolo26n-pose.pt` |
| `--detection-model` | Model detect dùng cho người và bóng, mặc định `yolo26n.pt` |
| `--ball-model` | Model detect bóng riêng; nếu bỏ trống sẽ dùng detection model |
| `--ball-class-id` | Class ID của bóng trong model riêng, mặc định COCO `32` |
| `--player-tracker` | Tracker cầu thủ, mặc định `botsort.yaml` |
| `--ball-tracker` | Tracker bóng, mặc định `bytetrack.yaml` |
| `--device` | `cpu`, GPU index như `0`, hoặc device Ultralytics hỗ trợ |
| `--imgsz` | Kích thước ảnh inference, mặc định `960` |
| `--player-conf` | Detection threshold cho người, mặc định `0.20` |
| `--ball-conf` | Detection threshold cho bóng, mặc định `0.15` |
| `--filter-ball-by-roi` | Ép bóng vào ROI; không phù hợp khi bóng bay cao |
| `--keypoint-conf` | Threshold để vẽ/dùng keypoint, mặc định `0.25` |
| `--speed-smoothing` | Hệ số EMA của vận tốc, mặc định `0.35` |
| `--roi` | Polygon mặt sân theo pixel để loại người/bóng ngoài sân |
| `--no-team-classification` | Tắt phân nhóm màu áo online |
| `--team-min-tracks` | Số track tối thiểu trước khi chia hai đội, mặc định `4` |
| `--scene-cut-threshold` | Ngưỡng reset speed/team khi camera cut, mặc định `0.35` |
| `--no-scene-cut-reset` | Tắt scene-cut detector |
| `--calibration` | JSON chứa bốn cặp điểm image→pitch |
| `--observations` | Đường dẫn JSONL lưu dữ liệu từng frame |
| `--max-frames` | Giới hạn số frame để chạy preview |

Các command còn lại:

```powershell
& $CLI evaluate --help
& $CLI analyze-events --help
& $CLI video-info --help
& $CLI quality-report --help
& $CLI infer-events --help
& $CLI run-pipeline --help
& $CLI doctor --help
& $CLI soccernet-download --help
& $CLI soccernet-analyze --help
& $CLI field-geometry --help
& $CLI evaluate-observations --help
& $CLI benchmark-ball --help
& $CLI soccernet-export-yolo --help
& $CLI train-ball-model --help
```

Tăng kích thước inference để bắt người/bóng nhỏ tốt hơn, đổi lại chạy chậm hơn:

```powershell
& $CLI annotate-video $VIDEO "$RUN\annotated-1280.mp4" `
  --observations "$RUN\observations-1280.jsonl" `
  --imgsz 1280 `
  --ball-conf 0.15 `
  --device cpu `
  --max-frames 300
```

Dùng pose model lớn hơn:

```powershell
& $CLI annotate-video $VIDEO "$RUN\annotated-pose-s.mp4" `
  --pose-model yolo26s-pose.pt `
  --observations "$RUN\observations-pose-s.jsonl" `
  --device cpu `
  --max-frames 300
```

Đổi tracker cầu thủ sang ByteTrack:

```powershell
& $CLI annotate-video $VIDEO "$RUN\annotated-bytetrack.mp4" `
  --player-tracker bytetrack.yaml `
  --max-frames 300
```

## 14. Cách đọc JSONL output

Mỗi dòng tương ứng một frame:

```json
{
  "frame_index": 0,
  "timestamp_sec": 0.0,
  "objects": [
    {
      "track_id": 1,
     "object_type": "player",
      "team": "team_a",
      "confidence": 0.91,
      "bbox": {"x1": 100.0, "y1": 200.0, "x2": 150.0, "y2": 320.0},
      "keypoints": [],
      "anchor_image": [125.0, 320.0],
      "speed": 0.0,
      "speed_unit": "px/s"
    }
  ]
}
```

Khi calibration được bật, mỗi object có thêm `pitch_x` và `pitch_y`.

## 15. Lỗi thường gặp

### `Vision dependencies are missing`

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[vision]"
```

### Cài Torch/Ultralytics lỗi trên Python 3.14

Thử Python 3.12, phiên bản thường có hệ sinh thái computer vision ổn định hơn:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -e ".[vision]"
$CLI = ".\.venv312\Scripts\attack-analysis.exe"
```

Lệnh này yêu cầu Python 3.12 đã được cài và hiển thị trong `py -0p`.

### Không tìm thấy `attack-analysis.exe`

Có thể gọi module trực tiếp bằng đúng Python trong venv:

```powershell
.\.venv\Scripts\python.exe -m attack_analysis --help
.\.venv\Scripts\python.exe -m attack_analysis evaluate examples\attack_effective.json
```

### Ball F1 thấp

Không xử lý bằng cách hạ confidence tùy ý. Chạy `benchmark-ball`; nếu recall vẫn
thấp ở threshold `0.01`, model đang lệch domain và bắt buộc fine-tune GPU
`imgsz=1280`. Path FOOTPASS+HDF5 không chờ gate này. Xem `docs/ball-accuracy-plan.md`.

### Calibration coverage 0% / overlay lệch đồng hồ

Chạy `inspect-video-sync` và đối chiếu `FOOTPASS_VIDEO_SYNC_JSON` (offset/fps).
Replay/bbox NaN không được ngoại suy. Checklist: `docs/ops-checklist.md`.

### Skeleton chập chờn

Dùng video 1080p, tăng `--imgsz`, giảm nhẹ `--player-conf`, hoặc thử
`yolo26s-pose.pt`. Cầu thủ quá nhỏ/che khuất vẫn là giới hạn của pose model generic.

### Track ID đổi liên tục

Loại bỏ replay/camera cut, thử BoT-SORT, hoặc sau này thêm ReID chuyên cho cầu thủ.

### Speed `km/h` bất thường

Kiểm tra thứ tự bốn điểm calibration và camera có zoom/pan hay không. Với camera
chuyển động, dùng `px/s` như diagnostic cho tới khi có per-frame calibration.

### Output mất tiếng

OpenCV writer hiện chỉ xuất video frame, không giữ audio gốc. Audio không ảnh
hưởng pipeline CV; có thể ghép lại bằng FFmpeg ở bước đóng gói sau.

## 16. Thứ tự chạy ngắn gọn

```text
1. Tạo .venv
2. pip install -e .
3. Chạy 53 tests
4. Chạy doctor
5. Thử evaluate/analyze-events
6. pip install -e ".[vision]"
7. Chạy lại doctor và video-info
8. run-pipeline --max-frames 300
9. Kiểm tra MP4 + JSONL + quality + manifest
10. Chạy benchmark-ball và yêu cầu accepted=true
11. Nếu fail: export dataset rồi fine-tune model bóng trên GPU
12. Chạy full video bằng --ball-model và threshold đã benchmark
13. Chỉ bật km/h/event inference khi calibration hợp lệ
```
