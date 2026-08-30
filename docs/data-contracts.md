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

## Attack-v2 domain stream

`ObservedActionV2` giữ observation tức thời, xác suất action/actor/time độc lập,
stable team, period direction, bbox/replay và provenance từng field. Action không
được gọi là possession hay attack.

Decoder tạo `PossessionSequenceV2` chứa action của cả hai đội và state transition.
Chỉ possession `complete` có `shot_attempt`, `headed_attempt` hoặc
`forced_goal_line_out` mới được bọc thành `AttackSequenceV2`. Gap/cut/max duration
chỉ sinh `left_censored/right_censored/uncertain`. Generic header, tackle hoặc
block không đổi owner cho tới khi có established control.

`PossessionValueV2` tách threat/value khỏi attack ontology. Nếu không có ball/control
trajectory thì progression/xT/EPV là `unknown`, không lấy vị trí actor đầu/cuối
để thay thế. `shot_candidate` từ trajectory không có xG.

`AttackSequence` V1 chỉ còn là compatibility input. Converter luôn tạo
`legacy_unvalidated` + `uncertain`, không tự chuyển V1 shot thành Attack-v2.
Đường phân tích cửa sổ V1 chỉ ghi `PossessionCandidateV1` vào
`selected-possession-candidate-v1.json`: cửa sổ thiếu đầu/cuối mang trạng thái
`left_censored`/`right_censored`, terminal nằm ngoài cửa sổ không được mượn lại,
và mọi attack/xG/effectiveness verdict đều bị giữ lại. V1 không có các field
`beneficiary_team_id`, `last_touch_team_id` và `cause`, nên nhãn
`forced_goal_line_out` cũ chỉ là evidence thô chứ không phải terminal v2 hợp lệ.
Sidecar dùng manifest riêng `possession-candidate-v1-manifest.json`; nó không ghi
đè manifest vision upstream và luôn bị cache v2 từ chối.

## Trạng thái triển khai

| Khối | Trạng thái |
|---|---|
| Video metadata | Đã có, cần OpenCV khi chạy video thật |
| Frame observation schema | Đã có và validate được |
| Detector/tracker | Đã có MVP YOLO26 Pose + Detect, BoT-SORT + ByteTrack |
| Static calibration | Đã có homography bốn điểm cho camera cố định |
| FOOTPASS-assisted dynamic calibration | Đã có per-frame homography + holdout quality report |
| Video-only dynamic calibration | Chưa tích hợp semantic pitch landmarks |
| Event extraction từ tracks | Có baseline possession/carry/pass/turnover |
| Attack-v2 contract + deterministic decoder | Đã có và unit-tested; chưa human-validated |
| Learned TCN/semi-Markov | Hạ tầng/config/decoder đã có; checkpoint chưa train/validate |
| Sequence metric registry | Hungarian + tIoU + bootstrap + gates đã có |
| Effectiveness | Legacy score bị khóa; tactical-quality cần ≥300 confirmed labels |
| Observation quality gate | Đã có `quality-report` |
| Scene-cut handling | Reset speed/team và chặn event transition qua cut |
| LLM grounded explanation | Chưa tích hợp; hiện dùng template |

`xg_v2.py` chuẩn hóa location, body part, pressure, goalkeeper/defender visibility,
play pattern và observed-player fractions với missingness masks. xG chỉ được ghi
sau confirmed shot/headed terminal; outcome thiếu vẫn là `unknown`. Artifact V1
không được render như xG-v2.

Manifest v2 phải chứa video/dataset/content-tree/config/model/schema/postprocess
hash và source tier. Cache thiếu hoặc sai bất kỳ hash nào bị từ chối.

Gold manifest dùng `ATTACK_V2_RUBRIC_VERSION` từ public model contract và SHA-256
nội dung video, không dùng size/mtime. Trường `test_labels_sealed` mặc định là
`false`; chỉ trở thành `true` khi seal của đúng `game_0`, `game_36`, `game_50`
được kiểm tra lại thành công. Annotation timeline dùng frame interval inclusive
và phải phủ liên tục từng period bằng một trong `possession`, `attack`,
`uncertain`, `dead_ball`; gap/overlap làm validation thất bại và không được tự
động điền nhãn.
