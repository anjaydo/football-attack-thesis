# Hợp đồng dữ liệu giữa các khối

Tài liệu này cố định dạng dữ liệu trung gian để detector/tracker có thể được thay
đổi mà không ảnh hưởng tới mô hình đánh giá hiệu quả.

## Video metadata

Lệnh `video-info` đọc `width`, `height`, `fps`, `frame_count` và `duration_sec`.
OpenCV là dependency tùy chọn, không cần thiết khi chỉ làm việc với event data.

## Frame observations

Mỗi frame chứa:

- `frame_index`, `timestamp_sec`;
- `calibration_confidence`;
- danh sách object với `track_id`, loại object và detection confidence;
- bounding box trong ảnh;
- `pitch_x`, `pitch_y` sau homography;
- team nếu object là cầu thủ/thủ môn.

Ví dụ đầy đủ nằm tại `examples/frame_observations.json`. Đây sẽ là output chuẩn
của khối SoccerNet/YOLO + tracker + camera calibration.

## Match event stream

Từ chuỗi frame observations, event extractor cần tạo ra các event như `pass`,
`carry`, `shot`, `turnover`, `ball_out`, `offside`. Bộ segmenter hiện tại tách
đợt tấn công khi:

- đội sở hữu bóng thay đổi;
- gặp terminal event;
- khoảng cách giữa hai event vượt `max_gap_sec`.

Sau segmentation, mỗi `AttackSequence` được đưa thẳng vào metric/scoring pipeline.

## Trạng thái triển khai

| Khối | Trạng thái |
|---|---|
| Video metadata | Đã có, cần OpenCV khi chạy video thật |
| Frame observation schema | Đã có và validate được |
| Detector/tracker | Đã có MVP YOLO26 Pose + Detect, BoT-SORT + ByteTrack |
| Static calibration | Đã có homography bốn điểm cho camera cố định |
| Dynamic broadcast calibration | Chưa tích hợp |
| Event extraction từ tracks | Có baseline possession/carry/pass/turnover |
| Attack segmentation | Đã có baseline |
| Metrics và effectiveness score | Đã có baseline |
| Observation quality gate | Đã có `quality-report` |
| Scene-cut handling | Reset speed/team và chặn event transition qua cut |
| LLM grounded explanation | Chưa tích hợp; hiện dùng template |
