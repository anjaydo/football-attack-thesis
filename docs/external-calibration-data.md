# Nguồn dữ liệu bổ sung cho calibration và tracking

Snapshot nghiên cứu: **2026-08-25**. Các nguồn dưới đây chỉ được đưa vào train sau
khi lưu dataset card, phiên bản, license và checksum. Không trộn ảnh của cùng một
trận vào nhiều split và không dùng challenge set để chọn hyperparameter.

## 1. SoccerNet Camera Calibration 2023 — ưu tiên cao nhất

- Nguồn chính thức: <https://github.com/SoccerNet/sn-calibration>
- Nội dung phù hợp: ảnh broadcast và annotation các extremity/point của từng
  thành phần đường sân; có protocol camera calibration và metric chính thức.
- Cách dùng: train detector keypoint/line cho đường `predicted`; giữ validation
  theo match để đo completeness và reprojection error. FOOTPASS vẫn là bộ đánh
  giá chính cho downstream attack analysis.
- Tải bằng package chính thức:

```python
from SoccerNet.Downloader import SoccerNetDownloader

downloader = SoccerNetDownloader(LocalDirectory=r"D:\datasets\SoccerNet")
downloader.downloadDataTask(
    task="calibration-2023",
    split=["train", "valid", "test"],
)
```

Không tải `challenge` trong giai đoạn phát triển. Việc truy cập video SoccerNet
có thể yêu cầu credential/NDA riêng của chủ dataset.

## 2. SoccerNet Game State Reconstruction v1.3 — ưu tiên cao

- Nguồn chính thức: <https://github.com/SoccerNet/sn-gamestate>
- Quy mô công bố: 200 clip đã gán nhãn.
- Nội dung phù hợp: bbox/track, role, team, jersey và vị trí 2D trên sân. Bản
  v1.3 sửa tính nhất quán theo thời gian của `bbox-pitch`.
- Cách dùng: benchmark tracking, calibration và error propagation với GS-HOTA;
  không coi đây là nhãn effectiveness.
- Trước khi dùng, kiểm tra `Labels-GameState.json -> info.version >= 1.3`.

## 3. Roboflow Football Field Key Points v11 — chỉ dùng auxiliary train

- Dataset card: <https://universe.roboflow.com/bronkscottema/football-field-key-points/dataset/11>
- Dataset card ghi **CC BY 4.0**, 2.149 ảnh ở version 11, task keypoint detection.
- Version này resize kiểu `Stretch` về 640×640 và tạo 7 output augmentation cho
  mỗi training example. Vì vậy không xem 2.149 ảnh là 2.149 cảnh độc lập.
- Chỉ dùng để pretrain/augmentation sau khi audit nguồn ảnh, keypoint schema,
  ảnh trùng và attribution. Không dùng split validation/test có sẵn để báo cáo
  kết quả luận văn; tự chia lại theo video/match gốc nếu metadata cho phép.

## 4. WorldPose — enrichment nghiên cứu, không nằm trên critical path

- Nguồn chính thức: <https://eth-ait.github.io/WorldPoseDataset/>
- Nội dung công bố: hơn 80 sequence, khoảng 2,5 triệu 3D pose, footage và camera
  parameters từ World Cup 2022.
- Phù hợp cho camera pose/3D player localization, nhưng phải gửi đơn xin dữ liệu;
  footage thuộc FIFA và có thỏa thuận bổ sung. Chỉ tích hợp sau khi quyền sử dụng
  cho luận văn được xác nhận bằng văn bản.

## 5. WC-2014 và TS-WorldCup — candidate để đo generalization

- Hướng dẫn và baseline: <https://github.com/mguti97/No-Bells-Just-Whistles>
- Repo cung cấp đường dẫn đến ảnh/homography WC-2014 và TS-WorldCup, cùng weights
  đã train/fine-tune cho SoccerNet/WC-2014/TS-WorldCup.
- Chỉ dùng làm external test hoặc pretrain sau khi kiểm tra license ở nguồn dữ
  liệu gốc. License của code không tự động cấp quyền cho footage/dataset.

## Thứ tự thực hiện được khuyến nghị

1. Download **SoccerNet Calibration 2023 train/valid/test** vào `D:\datasets`.
2. Chuẩn hóa keypoint/line labels về một pitch landmark schema duy nhất.
3. Train một model nhỏ ở 640 px; dùng gradient accumulation nếu VRAM 6 GB.
4. Đo trên validation theo match: keypoint precision/recall, completeness,
   median/P90 top-view error và tỷ lệ frame vượt quality gate.
5. Fine-tune thêm Roboflow chỉ như auxiliary train, rồi chạy ablation có/không
   có nguồn này để kiểm tra domain shift.
6. Tích hợp model tốt nhất vào đường video-only; FOOTPASS-assisted vẫn được giữ
   làm oracle upper bound và không được trình bày như kết quả video-only.

## Provenance tối thiểu phải lưu

Mỗi snapshot dữ liệu phải có: `dataset_name`, `version`, `downloaded_at`,
`source_url`, `license`, `split_policy`, `file_checksum`, `match_ids`,
`annotation_schema` và ghi chú các ảnh bị loại do trùng/không rõ nguồn.
