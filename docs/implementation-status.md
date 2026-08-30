# Trạng thái triển khai và tiêu chí đủ chức năng

Ba mức xác minh được dùng trong tài liệu này:

- **Unit-tested**: thuật toán độc lập đã có test;
- **Integration-tested**: luồng nhiều module đã chạy bằng backend YOLO/OpenCV giả;
- **Real-video verified**: đã chạy với model weights và clip bóng đá thật.

Release hiện là **`research_only`**. Unit/integration/real-data invariants không
thay thế human ground truth: 12 trận chưa gán nhãn xong, temporal/action/vision
models chưa qua development gates và frozen test chưa được mở. Vì vậy các con số
oracle-assisted hiện tại chỉ là audit hạ tầng, không phải accuracy claim.

## Ma trận chức năng

### SoccerNet GSR ingestion

- `soccernet-download` đọc central directory từ archive chính thức và chỉ tải
  byte range liên tục của một sequence, có path-safety, size và CRC validation.
- Sequence validation đầu tiên `SNGS-033` đã được tải: 750 frame/30 giây, dataset
  version 1.3.
- `soccernet-analyze` đọc nested GSR attributes, thống kê track/role/team/jersey,
  kiểm tra pitch-coordinate outlier, lọc speed phi vật lý và tạo provisional
  attack-effectiveness assessment.
- Full-match SoccerNet video chưa được tải vì nguồn chính thức yêu cầu NDA và
  password nghiên cứu riêng; project không nhúng hoặc né credential đó.

| Chức năng | Trạng thái | Mức xác minh | Ghi chú |
|---|---|---|---|
| JSON schema/validation | Hoàn thành | Unit + integration | Runner output đọc lại được bằng schema |
| Video metadata | Hoàn thành | Unit + real video | Đã đọc clip SoccerNet 1920x1080 |
| Player detection + skeleton | Hoàn thành baseline | Integration + real video | Detect người rồi ghép pose theo IoU |
| Ball detection | Chưa đạt quality gate | Integration + real video | F1 0.348, gate yêu cầu 0.80; đã có exporter/trainer/benchmark |
| Ball accuracy gate | Hoàn thành | Unit + real data | Quét threshold, yêu cầu F1 0.80 và P/R 0.75 |
| Player/ball tracking ID | Hoàn thành baseline | Integration + real video | Player tracking tốt ở preview; ball tracking chưa đạt |
| Pitch ROI | Hoàn thành baseline | Unit + real video | ROI tĩnh hoặc grass contour tự động |
| Field line/intersection | Hoàn thành baseline | Unit + real video | HSV grass/white mask + Hough; chưa semantic label |
| Team color clustering | Hoàn thành baseline | Unit | `team_a/team_b/unknown` |
| Scene-cut handling | Hoàn thành baseline | Unit + integration | Reset speed/team; chưa reset tracker nội bộ |
| Speed `px/s` | Hoàn thành | Unit + integration + real video | Có camera motion, chỉ là image-space diagnostic |
| Static homography | Hoàn thành | Unit | Cho camera cố định |
| Speed `km/h` | Hoàn thành code | Unit | Chỉ đúng khi calibration đúng |
| FOOTPASS-assisted dynamic calibration | Hoàn thành | Unit + real video | Per-frame ROI↔tactical homography, holdout median/P90 gate |
| Video-only dynamic calibration | Chưa có | Chưa test | Cần semantic pitch landmarks/checkpoint SoccerNet calibration |
| Observation quality report | Hoàn thành | Unit + CLI | Pass/warn/fail và warning codes |
| Possession inference | Deterministic v2 baseline | Unit | Contested touch không đổi owner; learned model chưa train |
| Carry/pass/turnover | Hoàn thành baseline | Unit + CLI | Có physical transfer gate |
| Shot candidate | Hoàn thành baseline | Unit | `shot_candidate` chưa xác minh; không terminal/xG |
| Geometry-only xG | Legacy ablation | Unit | Không được gắn vào Attack-v2 output |
| xG-v2 | Hạ tầng + trainer hoàn thành | Unit + StatsBomb run | Chưa có FOOTPASS human-domain validation |
| Attack-v2 decoder | Hoàn thành deterministic baseline | Unit + real FOOTPASS audit | Strict terminal; chưa human-GT accuracy |
| TCN/semi-Markov | Hạ tầng hoàn thành | Unit + Torch smoke | `infrastructure_untrained` |
| Effectiveness scoring | Legacy bị khóa | Unit | Không categorical verdict trước ≥300 confirmed labels + gates |
| Sequence evaluation | Hoàn thành | Unit | Hungarian, tIoU .25/.5/.75, 2k match bootstrap |
| Tracking evaluation | Hoàn thành | Unit | AP50-compatible, IDF1, HOTA, team/jersey, ball, GS-HOTA-style |
| One-command orchestration | Hoàn thành | Integration backend giả | `run-pipeline` + manifest |
| LLM grounded explanation | Legacy-only | Unit | Không được gọi cho censored/non-attack v2 |
| SoccerNet GSR sequence | Hoàn thành | Real-data verified | `SNGS-033`, 750 frame + GSR labels v1.3 |
| Real YOLO inference | Đã xác minh preview | Real video | 50 frame ở 1920x1080, inference size 1280, CPU |

## FOOTPASS calibrated-window verification lịch sử (stale)

Các artifact dưới đây dùng pipeline/window V1 và geometry/cache cũ; chúng đã bị
đánh dấu stale, không còn là kết quả release. Lịch sử đã chạy trên
`D:\videos\game_18.mp4`, giữ timestamp/frame toàn trận và chỉ
suy luận YOLO trong cửa sổ được chọn:

- attack `game_18_H1:attack-0001`, frame 39–193: calibration coverage 100%,
  median/P90 holdout error 0.222/0.732 m, player speed P95 31.2 km/h;
- attack có shot `game_18_H1:attack-0013`, frame 2492–2665: calibration coverage
  81.0%, median/P90 0.194/1.179 m và trained xG 0.010;
- video overlay được chuyển sang H.264 và phát trong Chromium (`readyState=4`);
- CV-only event inference abstain đúng vì ball tracking coverage lần lượt chỉ
  6.8% và 2.7%; event stream chính dùng FOOTPASS oracle/DST được báo nguồn riêng.

Kết quả lịch sử này là **oracle/assisted calibration** vì correspondence lấy từ
tactical HDF5. Không được báo cáo nó như chất lượng calibration video-only hoặc
Attack-v2. Audit mới nằm ở `runs/verification-v2` và mang content provenance.

## Điều kiện hoàn thành milestone video đầu tiên

Milestone được coi là đạt trên một clip thật khi:

1. `doctor` trả `vision_ready=true`;
2. `video-info` đọc được video;
3. `run-pipeline --max-frames 300` tạo đủ MP4, JSONL, quality và manifest;
4. không gắn khán giả/HLV hàng loạt thành player sau khi chỉnh ROI;
5. ball frame coverage, player tracking và keypoint coverage được báo cáo;
6. track ID không đổi bất thường trong cùng camera shot;
7. speed chỉ dùng `km/h` khi calibration đã kiểm tra;
8. một người xem xác nhận trực quan overlay trên ít nhất 300 frame.

## Kết quả real-video hiện tại

Kết quả `doctor` trên máy hiện tại:

- Python 3.14.3, OpenCV 5.0.0, Torch 2.13.0 CPU và Ultralytics 8.4.126;
- `vision_ready=true` trong runtime thử nghiệm;
- không phát hiện CUDA/NVIDIA GPU;
- FFmpeg chưa cài;
- SoccerNet GSR sequence `SNGS-033` đã được ghép thành clip preview và chạy YOLO.

Preview detect+pose có 632 player detections/50 frame, 22 player track, player
tracking coverage 100% và keypoint coverage 36.21%. Ground-truth person F1 là
0.721. Geometry có ở 100% frame. Sau khi tăng ball threshold và không dùng grass
ROI cho bóng bay cao, ball F1 tăng từ 0.13 lên 0.348 nhưng vẫn chưa đủ chất lượng
để suy luận possession/shot/xG. Chi tiết ở `docs/yolo-field-preview-analysis.md`.

Ba checkpoint chuyên bóng công khai cũng đã được kiểm tra trên cùng 50 frame.
F1 tốt nhất tương ứng là 0.342, 0.067 và 0.088, cho thấy domain shift rõ ràng.
Pipeline không coi các model-card metric là kết quả của đề tài; chỉ metric trên
held-out SoccerNet sequence mới được dùng. Dataset fine-tune hiện có 3.000 frame
train từ bốn sequence và 750 frame validation từ `SNGS-033`, không trùng sequence.
