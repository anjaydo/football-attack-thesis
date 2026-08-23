# Pipeline video, skeleton và tracking

## Vì sao vẫn dùng OpenCV?

OpenCV không cạnh tranh với YOLO. Hai thành phần làm hai nhiệm vụ khác nhau:

- OpenCV giải mã MP4 thành frame, tạo video output và vẽ overlay;
- YOLO chạy inference trên từng frame để phát hiện object hoặc keypoint;
- tracker nối detection giữa các frame để tạo track ID;
- homography đổi tọa độ ảnh sang tọa độ mặt sân;
- speed estimator tính chuyển vị theo thời gian cho từng track.

## Detect, pose và tracking song song

Pipeline mặc định chạy ba nhánh inference độc lập:

1. `yolo26n.pt` detect và track lớp `person` để lấy đủ player bbox và ID;
2. `yolo26n-pose.pt` lấy 17 COCO keypoints, sau đó ghép vào player bbox bằng IoU;
3. một instance `yolo26n.pt` độc lập detect/track lớp COCO `sports ball` (class 32).

Không dùng trực tiếp output bbox của pose làm toàn bộ player list vì cầu thủ xa
thường vẫn được detect model bắt thấy nhưng pose model không đủ pixel để suy ra
khớp. Cách ghép này giữ được bbox/ID và chỉ để trống `keypoints` khi pose chưa đủ
tin cậy. Model generic vẫn có thể bỏ sót hoặc nhận nhầm bóng nhỏ; fine-tune một
football-ball detector là thí nghiệm bắt buộc tiếp theo.

## Hình học và line trên sân

YOLO COCO không có class cho touchline, halfway line, penalty-area line hay center
circle. Vì vậy project tách thành hai nhánh:

1. YOLO pose/detect/track tạo player/ball box, skeleton, class, confidence và ID;
2. `opencv_hsv_hough_v1` tách connected grass mask, white-line mask, Hough line
   segments và các giao điểm `cross/junction`.

Bật nhánh thứ hai bằng `annotate-video --field-lines`. Mỗi JSONL frame có:

```json
{
  "field_geometry": {
    "method": "opencv_hsv_hough_v1",
    "pitch_mask_coverage": 0.78008,
    "line_mask_coverage": 0.01708,
    "segments": [
      {"x1": 1094, "y1": 242, "x2": 1094, "y2": 1079, "angle_deg": 90}
    ],
    "intersections": [
      {"x": 1094, "y": 505, "angle_deg": 88.1, "kind": "cross"}
    ]
  }
}
```

Muốn dùng YOLO cho line sân cần fine-tune segmentation/semantic model trên nhãn
riêng, ví dụ các class `pitch`, `touchline`, `goal_line`, `halfway_line`,
`penalty_area_line`, `center_circle`. `yolo26n-seg.pt` pretrained chỉ biết class
COCO; đổi model mà không có nhãn football sẽ không giải quyết calibration. Mask
line custom sau này sẽ thay white-line mask, còn bước fit line/intersection và
homography vẫn tái sử dụng được.

Player detect model chạy với BoT-SORT vì video broadcast có pan/zoom và tracker
này có camera-motion compensation. Ball model dùng ByteTrack để tận dụng cả
detection confidence thấp. Hai tracker có state độc lập; pose model chỉ chạy
predict và ghép keypoint vào bbox đã track.

## Pitch ROI và phân nhóm đội

Pose COCO nhận diện lớp `person`, không biết ai là cầu thủ. Pipeline hỗ trợ ROI
đa giác theo pixel; chỉ người có điểm chân nằm trong ROI mới được giữ lại. Bóng
không bị grass ROI tự động lọc vì một quả bóng bay cao có thể nằm ngoài projection
2D của mặt cỏ. Chỉ bật `--filter-ball-by-roi` cho clip camera cố định khi giả định
này chắc chắn đúng.

Với các player track còn lại, vùng thân trên được chuyển sang không gian Lab và
phân cụm online thành `team_a`, `team_b` hoặc `unknown`. Tâm cụm được ghép ổn định
qua frame và nhãn được voting theo track. Đây là baseline màu áo, chưa thay thế
team classifier/ReID được huấn luyện chuyên biệt.

## Tính vận tốc

Điểm theo dõi cầu thủ là trung bình hai mắt cá chân đủ confidence; nếu chân bị
che, pipeline dùng bottom-center của bounding box. Điểm bóng là tâm bounding box.

Vận tốc thô:

```text
speed = distance(position_t, position_t-1) / delta_time
```

Sau đó dùng exponential smoothing theo từng track ID.

- Không calibration: position là pixel, đơn vị `px/s`.
- Có homography: position là mét trên sân, output đổi thành `km/h`.

Không được diễn giải `px/s` thành tốc độ vật lý. Một homography tĩnh chỉ phù hợp
clip camera cố định. Video broadcast thay đổi pan/tilt/zoom cần ước lượng camera
calibration ở từng frame; đây là bước cần nối với SoccerNet-GSR.

## Output

Video output có:

- bounding box theo team color và skeleton cho `player`;
- bounding box cam cho `ball`;
- track ID;
- tốc độ `px/s` hoặc `km/h`.

JSONL tùy chọn lưu một record mỗi frame, gồm box, keypoints, anchor, speed và
`pitch_x/pitch_y` nếu đã calibration. Dữ liệu này là đầu vào cho event extraction,
xG/xT và LLM. Chạy `quality-report` trên JSONL trước khi dùng downstream. Report
cũng có `field_geometry_frame_coverage`, `average_field_line_segments` và
`average_field_intersections` khi bật `--field-lines`.

## Hạn chế MVP

- COCO pose có thể mất khớp khi cầu thủ quá nhỏ hoặc che khuất nhau;
- COCO sports-ball detector thường yếu với bóng vài pixel;
- track ID có thể đổi khi cut camera hoặc replay;
- scene-cut detector reset speed/team nhưng chưa reset tracker nội bộ của Ultralytics;
- OpenCV output hiện không giữ audio gốc;
- phân nhóm đội hiện chỉ dựa trên màu áo; thủ môn/trọng tài có thể thành `unknown`;
- event baseline nhận biết possession/carry/pass/turnover khi có team và calibrated
  coordinates; chưa nhận biết shot/goal.

## Tài liệu kỹ thuật

- Ultralytics YOLO26 Pose: https://docs.ultralytics.com/tasks/pose
- Ultralytics Multi-Object Tracking: https://docs.ultralytics.com/modes/track
- SoccerNet Game State Reconstruction: https://github.com/SoccerNet/sn-gamestate
