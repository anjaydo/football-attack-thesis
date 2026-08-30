# Tổng hợp vấn đề pipeline FOOTPASS + YOLO + xG

Tài liệu này ghi **các lỗ hổng đã gặp và còn tồn tại** trên triple pipeline (event FOOTPASS, overlay YOLO, xG-v2, kết luận Streamlit). Đây không phải changelog hay plan triển khai.

Triệu chứng hiện tại khớp mô tả: **event list đã có, tọa độ và skeleton đã gắn trên overlay, nhưng Spatial context trong kết luận vẫn nghèo** — gần như chỉ một dict `xg_raw` vài field, nhiều giá trị `null` / `0` / `unknown`.

---

## 1. Kiến trúc thực tế (vì sao “đã có data” mà context vẫn thiếu)

Ba nguồn **không cùng một không gian** và **không được gộp thành một freeze-frame giàu**:

| Lớp | Vai trò đúng | Thực tế hiện nay |
|-----|----------------|------------------|
| FOOTPASS play-by-play | Chuỗi hành động, timestamp, team, (thường) `x/y` possession-relative | Timeline UI đầy đủ; **không** được đưa vào Spatial context ngoài heuristic `play_pattern` |
| FOOTPASS tactical HDF5 | Oracle vị trí mọi cầu thủ + role (GK = role `1`) từng frame trận | Dùng để **fit homography**; `_footpass_shot_features` có GK/foe/pressure **nhưng fusion xG-v2 không đọc** |
| YOLO window | Bbox, track ID, skeleton COCO, team cluster, `pitch_x/y` nếu homography sống | Overlay review; fusion chỉ lấy **một frame** quanh terminal, rồi đếm/suy rất hẹp |

Luồng kết luận:

1. Decode Attack-v2 từ event.
2. Chạy cửa sổ calibrated → `observations.jsonl`.
3. `score_attack_with_observations` ghép **một** `ObservedActionV2` (terminal) với **một** `FrameObservation`.
4. UI “#### 2. Spatial context” in `association.method` + `st.json(xg_raw)`.

Mọi thứ giàu hơn (đủ 11+11 trên HDF5, skeleton 17 điểm, chuỗi pass/cross/throw-in, jersey/role) **không được materialize** thành context cho xG hay cho LLM.

---

## 2. Spatial context — lỗ hổng trung tâm

### 2.1 UI không phải spatial context

`render_conclusion_v2` chỉ hiện association (method, track, distance, foes) rồi dump JSON. Không có:

- sơ đồ freeze-frame (shooter, GK, đối phương gần, packing);
- khoảng cách shooter–GK, góc dứt điểm, số người trong đường sút;
- skeleton / bộ phận trúng bóng trên hình;
- đối chiếu event tại `t` (pass, cross, header) với track YOLO;
- quality của frame (calibration confidence, số keypoint nhìn thấy, ball có/không).

LLM nhận `EvidenceBundle` legacy (feature sequence + xG số), **không** nhận `xg_raw` đầy đủ hay observation JSONL.

### 2.2 Fusion chỉ dùng một frame, một action

- Terminal FOOTPASS `frame` được map sang frame video qua `video_clock` (Full HD vs 25 fps trận). Sai offset/fps → khớp nhầm khung, hoặc lấy frame gần nhất không phải lúc sút.
- Không fusion theo **cả possession** (ai chuyền cuối, cross trước shot, throw-in).
- `body_part` trên `ObservedActionV2` gần như luôn `null`, trừ header được promote `headed_attempt`.
- Suy `body_part` từ pose **chỉ khi cùng frame có object `ball`**. Ball F1 thấp / bóng không detect → `body_part` vẫn `null` dù skeleton shooter đã có.
- Không dùng cổ chân/gối/đầu so với **vị trí bóng FOOTPASS** hay last-touch event khi YOLO miss bóng.

### 2.3 Đếm người và GK không dùng oracle tactical

HDF5 đã có đủ cầu thủ + `FOOTPASS_ROLES`. Fusion lại:

- chỉ đếm YOLO đã `pitch_x/y` (mất người nếu calibration fail hoặc ngoài sân chiếu);
- team = cluster áo `team_a` / `team_b`, **không vote** với FOOTPASS team id;
- `default_team_mapping` gán đội tấn công → `team_a` mù; cluster ngược thì foe/ally đảo;
- GK: YOLO không có class GK; heuristic “gần khung thành theo median x đội” dễ gán nhầm CB/thủ môn đối phương;
- `goalkeeper_visible=false` khi không infer được GK dù overlay vẫn thấy người trong vòng cấm;
- `foe_players` / `ally_players` thường lệch (ví dụ 1 foe / 6 ally) vì nhiều track `team=None` bị loại;
- `defending_players` / `under_pressure` chỉ đếm đối phương trong **2 m** (`PRESSURE_RADIUS_M`, semantics StatsBomb). Sút từ xa hoặc 1 foe xa → `0` dù freeze-frame đầy người.

Kết quả điển hình đã thấy:

```text
body_part: null
x_goalkeeper / y_goalkeeper: null
goalkeeper_visible: false
foe_players: 1
ally_players: 6
defending_players: 0
under_pressure: 0
play_pattern: unknown   # đã chuyển heuristic; vẫn không có corner/FK/goal-kick vì FOOTPASS không có class
```

### 2.4 `play_pattern` / `shot_type` không có trong ontology FOOTPASS

Event chỉ: `drive, pass, cross, throw_in, shot, header, tackle, block`.

Không có corner, phạt trực tiếp, phát bóng, kick-off, penalty. Heuristic hiện tại (`throw_in` đầu chuỗi / counter nếu xuất phát nửa sân nhà) **không thay** StatsBomb `play_pattern`. xG-v2 được train one-hot StatsBomb → inference FOOTPASS phần lớn rơi `regular_play` hoặc `unknown` (trước khi sửa).

### 2.5 Hai hệ tọa độ

- Homography chiếu YOLO → pitch **tuyệt đối** HDF5 (`raw_x * 105`).
- Event / xG dùng **possession-relative** (đội kiểm soát luôn đánh về `x=105`).
- Fusion đã mirror khi `raw_left_to_right is False`; nếu flag thiếu hoặc direction team ≠ 0/1, shooter YOLO và `action.x` lệch nửa sân → association `unmatched`, đếm sai.

Pitch plot UI từng bị **đối phương mirror độc lập** (nhảy nửa sân). Plot possession-relative đã vá; fusion YOLO **không** dùng cùng helper `actions_in_attack_view` một cách thống nhất với HDF5 freeze-frame.

---

## 3. Calibration động (homography)

Triệu chứng đã gặp: **Calibration FAIL, Eligible coverage 0.0%, Median/P90 holdout = —**.

Nguyên nhân đã xác định / còn rủi ro:

1. **Scale bbox vs video overlay**  
   Correspondence: bottom-center bbox HDF5 → pitch. Config mặc định `source=1920×1080`. Overlay Full HD mà bbox native là video xử lý (ví dụ 640×352) thì điểm ảnh sai, mọi frame fail, không có holdout. Đã thêm auto-chọn scale; vẫn phụ thuộc percentile bbox và tập resolution ứng viên.

2. **Đồng hồ Full HD ≠ frame trận**  
   YOLO đọc Full HD; HDF5 theo ~25 fps. `PreviewClockProjector` đổi frame overlay → match. Sai `FOOTPASS_VIDEO_SYNC_JSON` (`preview_offset_sec`, `processing_fps`) → cửa sổ HDF5 trống → mọi frame *ineligible* → coverage 0%.

3. **Quality gate chặt**  
   Median ≤ 0.5 m, P90 ≤ 1.5 m, inlier ≥ 0.6, coverage ≥ 0.9, tối thiểu 6 correspondence. Một cửa sổ replay / bbox NaN / camera cắt → `no_valid_dynamic_homography`.

4. **Nội suy hẹp**  
   Extrapolation tối đa 8 frame trận (~0.3 s). Nhiều frame fail liên tiếp thì không chiếu `pitch_x/y`.

5. **Trước đây bỏ YOLO khi calibration fail**  
   Overlay/skeleton biến mất dù detect vẫn chạy được. Streamlit đã chạy YOLO cả khi FAIL (không chiếu sân). Spatial context khi đó **không có** tọa độ sân.

6. Calibration là **oracle-assisted** (tactical), không phải video-only. Không có landmark đường sân semantic.

---

## 4. YOLO: detect, pose, tracking, bóng

### 4.1 Pose không phải danh sách cầu thủ

Detect (BoT-SORT) và pose (`predict`, không track) tách nhau; ghép IoU. Pose thường thấy ít người hơn detect. Cầu thủ xa: có bbox/ID, **keypoints rỗng** → overlay “không skeleton”, không suy body part.

### 4.2 Lọc ROI / field lines

Mask cỏ + `player_roi.contains(anchor)` từng **cắt GK và người sát biên**. Đã nới: không lọc player theo ROI tự động; bóng vẫn có thể bị filter. Vòng cấm / GK vẫn dễ miss nếu detect conf cao hoặc cắt cảnh.

### 4.3 Độ phân giải suy luận

Full HD 1920×1080 với `imgsz=960` làm mất người nhỏ. Đã đưa cửa sổ calibrated lên 1280 khi video ≥ 1280. GPU 6 GB + cửa sổ 30 s vẫn là ràng buộc.

### 4.4 Team clustering

Jersey crop + cluster 2 nhóm, tối thiểu 4 track. Cắt cảnh reset cluster. Không jersey OCR. Không align FOOTPASS jersey/role.

### 4.5 Bóng

Ball ByteTrack, F1 historically ~0.35 vs gate 0.80. `score_attack_with_observations` **withhold xG** nếu `ball_f1 < 0.80` khi metric có trên manifest. Hệ quả: overlay có người nhưng xG abstain; hoặc body_part không suy được vì không có `object_type=ball`.

### 4.6 Overlay xuất review

Từng chạy YOLO trên video **chất lượng thấp** trong khi preview Full HD. Đã chuyển overlay sang `FOOTPASS_FULLHD_VIDEOS_DIR`. Cache cũ (low-res) dễ gây hiểu nhầm nếu không invalidate.

OpenCV `VideoWriter` H.264 trên Windows (OpenH264) từng fail; overlay trình duyệt cần FFmpeg/`imageio-ffmpeg` (mp4v → JPEG frames).

---

## 5. Event / Attack-v2 / ontology

- Attack = possession hoàn chỉnh + terminal ∈ `{shot_attempt, headed_attempt, forced_goal_line_out}`. **Goal không phải terminal** (`shot_outcome`).
- Possession không qualifying → không Attack-v2, nút kết luận / xG shot path hạn chế.
- Header: body_part=`head` chỉ khi infer terminal `headed_attempt`, không phải mọi `header` trên timeline.
- Không có aftermath bắt buộc trên mọi path; shot insights (audit / outcome / filter) phụ thuộc file gold thường **thiếu**.
- Replay FOOTPASS: bbox/tọa độ ineligible → calibration thủng đúng lúc highlight.

---

## 6. xG-v2

- Model `models/xg-v2.joblib` (gitignored), train StatsBomb, `research_only`.
- Feature one-hot: body_part, play_pattern, shot_type, GK loc, counts. **Missing đúng nghĩa** (mask) tốt hơn điền 0 giả; fusion từng điền `unknown` / `false` làm model tưởng “không có GK” thay vì “không quan sát”.
- Geometry-only / xG V1 heuristic không được gắn Attack-v2.
- `allow_research_only=True` trên UI — không phải human-domain validated.
- Nếu fusion `xg_raw` nghèo, xác suất xG **không phản ánh** freeze-frame thật, dù event list đẹp.

---

## 7. Kết luận LLM / UX

- Nút kết luận từng tách khỏi YOLO/xG (bundle V1, không fusion). Đã nối `build_attack_evidence`; vẫn **không** nhét freeze-frame/skeleton vào prompt.
- Schema `attack-verdict-v2`: chance quality + tactical read; **cấm** nhãn hiệu quả categorical.
- Brave Search: context chữ, không được dùng làm số. MCP mirror tool.
- Calibration FAIL trên UI dễ hiểu là “cả pipeline hỏng” dù event FOOTPASS vẫn đúng.
- Dual video: extract JSON từ `FOOTPASS_VIDEOS_DIR`; preview/YOLO Full HD. Mapping JSON dễ thiếu/sai.

---

## 8. Vấn đề đã gặp theo thời gian (đã vá một phần — không đóng file)

| # | Vấn đề | Trạng thái |
|---|---------|------------|
| P1 | Overlay YOLO không Full HD | Overlay path đã chuyển Full HD; cache/window cũ vẫn low-res nếu không chạy lại |
| P2 | Calibration 0% sau Full HD (scale bbox / clock) | Auto-scale + `PreviewClockProjector`; vẫn fail nếu sync/HDF5 trống hoặc gate không đạt |
| P3 | Pitch event nhảy nửa sân | Plot possession-relative + cảnh báo median x |
| P4 | Kết luận không dính xG/YOLO | Đã wire evidence; Spatial context UI vẫn JSON tối giản |
| P5 | OpenH264 / không phát mp4 trên browser | FFmpeg / frame JPEG preview |
| P6 | ROI cắt cầu thủ / pose yếu | Nới ROI, imgsz 1280, IoU pose thấp hơn; chưa đủ 22 người + skeleton |
| P7 | `body_part` / GK / play_pattern null | Heuristic pose+ball, GK theo vị trí, pattern từ sequence; **chưa** dùng HDF5 freeze-frame |
| P8 | Spatial context thiếu dù event + skeleton có | **Còn mở** — đây là gap sản phẩm chính |
| P9 | Ball F1 < 0.80 withhold xG | Còn; gate cố ý, làm đói feature pose-vs-ball |
| P10 | Team YOLO ≠ FOOTPASS team | Còn |
| P11 | xG research_only / chưa GT 12 trận | Còn (ràng buộc nghiên cứu) |
| P12 | Video-only calibration | Chưa có |

---

## 9. Chuẩn “Spatial context đủ” so với hiện trạng

Một freeze-frame phục vụ xG + luận chiến thuật **cần tối thiểu**:

1. Shooter: track, `x/y` (cùng hệ attack-view), body part, chân trụ / chân sút nếu có pose.
2. GK đối phương: vị trí, khoảng cách, có trong khung hình không (tách *missing* vs `visible=false`).
3. Mọi đối phương trong vòng cấm / trên đường sút (không chỉ bán kính 2 m).
4. Số đồng đội phía trước bóng / packing (không chỉ `ally_players` trừ 1).
5. Pattern từ **chuỗi event** (throw-in, cross→header, drive nhanh) + nói rõ FOOTPASS không có corner/FK.
6. Provenance: field nào lấy FOOTPASS tactical, field nào YOLO, field nào abstain.

Hiện chỉ đạt một phần (1) từ event `x/y`, còn lại phụ thuộc YOLO nghèo và **bỏ** oracle HDF5 lúc sút.

---

## 10. Hướng xử lý (chỉ để định hướng, chưa implement trong tài liệu này)

Ưu tiên đóng P8:

1. **Ghép freeze-frame FOOTPASS** (`_footpass_shot_features` / slice HDF5 tại `terminal.frame`) làm nguồn đếm GK/foe/pressure; YOLO bổ sung body_part và kiểm chứng nhìn thấy.
2. Materialize **spatial card** (không chỉ `st.json`): bảng + pitch mini + khoảng cách GK.
3. Body part: pose vs bóng YOLO **hoặc** vs last location bóng/event; header từ `action_type`.
4. Map team: majority vote track gần actor FOOTPASS tại frame sút.
5. Đưa `xg_raw` + freeze-frame vào evidence LLM (có `evidence_id`).
6. Đồng bộ `footpass-video-sync.json` và invalidate cache overlay khi đổi Full HD/scale.

Các mục P9–P12 là ràng buộc chất lượng / nghiên cứu, không nhầm với bug fusion.

---

*Cập nhật theo trạng thái repo sau các vá Full HD overlay, auto-scale calibration, nới tracking, và heuristic xG context. Gap còn lại: **không fusion đa nguồn thành spatial context**, dù từng nguồn đã tồn tại riêng.*
