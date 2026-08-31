# Tổng hợp vấn đề pipeline FOOTPASS + YOLO + xG

Tài liệu này ghi **các lỗ hổng đã gặp và còn tồn tại** trên triple pipeline (event FOOTPASS, overlay YOLO, xG-v2, kết luận Streamlit). Đây không phải changelog hay plan triển khai.

Triệu chứng trước đây: **event list đã có, tọa độ và skeleton đã gắn trên overlay, nhưng Spatial context trong kết luận vẫn nghèo**. Fusion hiện **ưu tiên freeze-frame tactical HDF5** (GK/foe/packing) và YOLO cho body_part/camera; UI spatial card không còn chỉ `st.json`.

---

## 1. Kiến trúc thực tế (vì sao “đã có data” mà context vẫn thiếu)

Ba nguồn **không cùng một không gian** và từng **không được gộp thành một freeze-frame giàu**:

| Lớp | Vai trò đúng | Thực tế hiện nay |
|-----|----------------|------------------|
| FOOTPASS play-by-play | Chuỗi hành động, timestamp, team, (thường) `x/y` possession-relative | Timeline + `possession_buildup` + heuristic `play_pattern`/`shot_type` |
| FOOTPASS tactical HDF5 | Oracle vị trí mọi cầu thủ + role (GK = role `1`) từng frame trận | Fit homography **và** freeze-frame xG (`footpass_freeze_frame`) |
| YOLO window | Bbox, track ID, skeleton COCO, team cluster, `pitch_x/y` nếu homography sống | Overlay review; fusion lấy **một frame** quanh terminal; team/role NN-align với freeze |

Luồng kết luận:

1. Decode Attack-v2 từ event.
2. (Tuỳ chọn) cửa sổ calibrated → `observations.jsonl`.
3. `score_attack_with_observations` ghép terminal + YOLO frame **và** freeze-frame HDF5 tại `terminal.frame`. Không có YOLO/HDF5 thì vẫn score **location-only** từ `x/y` sút (missingness masks).
4. UI spatial card: bảng provenance, khoảng cách GK, pitch mini (trail + pose trên sân nếu có `pitch_x`), `xg_raw`.

Jersey OCR / ontology corner-FK **gốc FOOTPASS** vẫn thiếu (chỉ có heuristic hình học).

---

## 2. Spatial context — lỗ hổng trung tâm

### 2.1 UI spatial card

`render_conclusion_v2` hiện bảng field/value/source, khoảng cách GK, pitch mini freeze-frame với **trail event + YOLO track + skeleton trên sân (nếu keypoint có `pitch_x`) + inset pose image-space**, và JSON `xg_raw` trong expander.

LLM nhận `EvidenceBundle` kèm `xg_raw` + `spatial_context` (freeze-frame, provenance, `evidence_id` HDF5). Không nhúng raw observation JSONL vào prompt.

### 2.2 Fusion terminal + freeze-frame

- Terminal FOOTPASS `frame` được map sang frame video qua `video_clock` (Full HD vs 25 fps trận). Sai offset/fps → khớp nhầm khung YOLO; freeze-frame HDF5 vẫn theo frame trận. UI hiện caption offset/fps.
- Possession: `possession_buildup` lấy pass/cross/throw-in gần nhất trước terminal (attack-view).
- `play_pattern` / `shot_type`: throw-in và counter từ chuỗi; **corner / free_kick / penalty** từ heuristic hình học (góc cờ, cụm ngoài vòng cấm, điểm 11 m). Provenance = `geometry_heuristic` hoặc `event_sequence` — **không** phải nhãn FOOTPASS.
- `body_part` trên action gần như luôn `null`; fusion suy từ pose vs bóng YOLO, không thì vs đáy bbox; header từ `action_type`.
- Keypoints: vision chiếu homography từng điểm nếu projector sống (5-tuple); fusion fallback local affine quanh anchor chân (image x → pitch y). Pose trên pitch mini **không** phải homography từng khớp nếu chỉ có affine.

### 2.3 Đếm người và GK

HDF5 freeze-frame là nguồn chính cho GK loc / foe / ally / pressure / packing khi có tactical file. YOLO còn:

- đếm `pitch_x/y` cho overlay và `goalkeeper_in_camera` (ưu tiên NN tới GK freeze nếu có);
- team cluster áo, **vote** track gần actor FOOTPASS;
- **align** track YOLO ↔ freeze side/role (nearest neighbour);
- heuristic GK YOLO chỉ khi không có oracle GK.
- `defending_players` / `under_pressure` 2 m vẫn là semantics StatsBomb trên **cả** freeze-frame; packing/`defenders_in_box`/`foes_in_shot_corridor` bổ sung cho UI.

Khi **không** có HDF5, triệu chứng YOLO nghèo (1 foe / 6 ally, GK null) vẫn xảy ra; xG lúc đó dùng missingness masks.

### 2.4 `play_pattern` / `shot_type` không có trong ontology FOOTPASS

Event chỉ: `drive, pass, cross, throw_in, shot, header, tackle, block`.

Không có corner, phạt trực tiếp, phát bóng, kick-off, penalty **trong file event**. Heuristic hình học **không thay** StatsBomb `play_pattern` đã gán tay. xG-v2 one-hot StatsBomb có thể nhận `corner` / `free_kick` / `penalty` từ heuristic — false positive (sút xa một mình bị gắn `free_kick`) là rủi ro đã ghi provenance.

### 2.5 Hai hệ tọa độ

- Homography chiếu YOLO → pitch **tuyệt đối** HDF5 (`raw_x * 105`).
- Event / xG dùng **possession-relative** (đội kiểm soát luôn đánh về `x=105`).
- Fusion mirror khi `raw_left_to_right is False`; **nếu flag thiếu**, chọn hướng mirror sao cho track YOLO gần `action.x/y` (attack-view qua `actions_in_attack_view`) nhất. Freeze-frame HDF5 và YOLO dùng cùng attack-view khi associate.

---

## 3. Calibration động (homography)

Triệu chứng đã gặp: **Calibration FAIL, Eligible coverage 0.0%, Median/P90 holdout = —**.

Nguyên nhân đã xác định / còn rủi ro:

1. **Scale bbox vs video overlay**  
   Correspondence: bottom-center bbox HDF5 → pitch. Config mặc định `source=1920×1080`. Overlay Full HD mà bbox native là video xử lý (ví dụ 640×352) thì điểm ảnh sai, mọi frame fail, không có holdout. Đã thêm auto-chọn scale; vẫn phụ thuộc percentile bbox và tập resolution ứng viên.

2. **Đồng hồ Full HD ≠ frame trận**  
   YOLO đọc Full HD; HDF5 theo ~25 fps. `PreviewClockProjector` đổi frame overlay → match. Sai `FOOTPASS_VIDEO_SYNC_JSON` (`preview_offset_sec`, `processing_fps`) → cửa sổ HDF5 trống → mọi frame *ineligible* → coverage 0%. UI hiện fps/offset.

3. **Quality gate chặt**  
   Median ≤ 0.5 m, P90 ≤ 1.5 m, inlier ≥ 0.6, coverage ≥ 0.9, tối thiểu 6 correspondence. Một cửa sổ replay / bbox NaN / camera cắt → `no_valid_dynamic_homography`.

4. **Nội suy**  
   Interpolation 25 frame giữa hai homography hợp lệ (kể cả lỗ bbox ngắn). **Extrapolation tối đa 25 frame trận (~1 s)** trên frame live; **không** ngoại suy vào frame replay/bbox NaN (0 correspondence) — những frame đó vẫn ineligible.

5. **Trước đây bỏ YOLO khi calibration fail**  
   Overlay/skeleton biến mất dù detect vẫn chạy được. Streamlit chạy YOLO cả khi FAIL. FAIL **không** có nghĩa cả pipeline hỏng: event FOOTPASS + freeze HDF5 vẫn dùng cho xG.

6. **Video-only fallback (P12, research baseline)**  
   Khi oracle projector vắng, `projector_from_pitch_polygon` map quad cỏ → 105×68 m. Không phải landmark semantic; sideline y có thể lật; confidence cố định 0.25. Overlay ghi `VIDEO-ONLY pitch (grass polygon)`.

---

## 4. YOLO: detect, pose, tracking, bóng

### 4.1 Pose không phải danh sách cầu thủ

Detect (BoT-SORT) và pose (`predict`, không track) tách nhau; ghép IoU (ngưỡng 0.05). Pose thường thấy ít người hơn detect. Cầu thủ xa: có bbox/ID, **keypoints rỗng** → overlay “không skeleton”, không suy body part.

### 4.2 Lọc ROI / field lines

Mask cỏ + `player_roi.contains(anchor)` từng **cắt GK và người sát biên**. Đã nới: không lọc player theo ROI tự động; bóng vẫn có thể bị filter. Vòng cấm / GK vẫn dễ miss nếu detect conf cao hoặc cắt cảnh.

### 4.3 Độ phân giải suy luận

Full HD 1920×1080 với `imgsz=960` làm mất người nhỏ. Đã đưa cửa sổ calibrated lên 1280 khi video ≥ 1280. GPU 6 GB + cửa sổ 30 s vẫn là ràng buộc.

### 4.4 Team clustering

Jersey crop + cluster 2 nhóm, tối thiểu 4 track. Cắt cảnh reset cluster. Không jersey OCR. Có NN-align FOOTPASS freeze side/role khi có tactical HDF5.

### 4.5 Bóng

Ball ByteTrack, F1 historically ~0.35 vs gate 0.80. `score_attack_with_observations` **không withhold xG** nếu đã có `x_shooter`/`y_shooter` (location-only + missingness) hoặc freeze-frame HDF5. Ball F1 thấp còn là **cảnh báo** (`ball_quality_warning`), không chặn model.

### 4.6 Overlay xuất review

Từng chạy YOLO trên video **chất lượng thấp** trong khi preview Full HD. Đã chuyển overlay sang `FOOTPASS_FULLHD_VIDEOS_DIR`. Cache invalidate theo `overlay=fullhd_pose_gk_v4_video_only` + `footpass-video-sync.json`.

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
- Feature one-hot: body_part, play_pattern, shot_type, GK loc, counts. **Missing đúng nghĩa** (mask) khi không có YOLO/HDF5 — location-only vẫn score.
- Geometry-only / xG V1 heuristic không được gắn Attack-v2.
- `allow_research_only=True` trên UI — không phải human-domain validated.
- Nếu fusion `xg_raw` nghèo, xác suất xG **không phản ánh** freeze-frame thật, dù event list đẹp.

---

## 7. Kết luận LLM / UX

- Nút kết luận từng tách khỏi YOLO/xG (bundle V1, không fusion). Đã nối `build_attack_evidence`; bundle có `xg_raw` + freeze-frame + `possession_buildup`. Inset pose không đưa vào prompt.
- Schema `attack-verdict-v2`: chance quality + tactical read; **cấm** nhãn hiệu quả categorical.
- Brave Search: context chữ, không được dùng làm số. MCP mirror tool.
- Calibration FAIL trên UI: cảnh báo tách “oracle homography” vs “event/xG/overlay vẫn chạy”.
- Dual video: extract JSON từ `FOOTPASS_VIDEOS_DIR`; preview/YOLO Full HD. Mapping JSON dễ thiếu/sai.

---

## 8. Vấn đề đã gặp theo thời gian

| # | Vấn đề | Trạng thái |
|---|---------|------------|
| P1 | Overlay YOLO không Full HD | Overlay path đã chuyển Full HD; cache/window cũ vẫn low-res nếu không chạy lại |
| P2 | Calibration 0% sau Full HD (scale bbox / clock) | Auto-scale + `PreviewClockProjector` + caption offset/fps; vẫn fail nếu sync/HDF5 trống hoặc gate không đạt |
| P3 | Pitch event nhảy nửa sân | Plot possession-relative + cảnh báo median x |
| P4 | Kết luận không dính xG/YOLO | Đã wire evidence + spatial card (freeze, buildup, pose inset/pitch) |
| P5 | OpenH264 / không phát mp4 trên browser | FFmpeg / frame JPEG preview |
| P6 | ROI cắt cầu thủ / pose yếu | Nới ROI, imgsz 1280, IoU pose 0.05; chưa đủ 22 người + skeleton |
| P7 | `body_part` / GK / play_pattern null | Pose+ball **hoặc bbox floor**; GK/foe từ **HDF5 freeze**; pattern từ sequence **và** heuristic hình học (corner/FK/penalty) + provenance |
| P8 | Spatial context thiếu dù event + skeleton có | **Đã đóng phần fusion + UI**: freeze HDF5, provenance, pitch mini, trail, YOLO track, pose trên sân + inset, buildup vào LLM |
| P9 | Ball F1 < 0.80 withhold xG | **Đóng**: không withhold khi có vị trí sút hoặc freeze; cảnh báo F1; location-only + missingness masks |
| P10 | Team YOLO ≠ FOOTPASS team | Vote track gần actor **và** NN-align freeze side/role |
| P11 | xG research_only / chưa GT 12 trận | **Còn** (ràng buộc nghiên cứu — không đóng bằng code) |
| P12 | Video-only calibration | **Baseline nghiên cứu**: grass-polygon homography khi oracle vắng; không phải oracle-quality |

---

## 9. Chuẩn “Spatial context đủ” so với hiện trạng

Một freeze-frame phục vụ xG + luận chiến thuật **cần tối thiểu**:

1. Shooter: track, `x/y` (cùng hệ attack-view), body part, chân trụ / chân sút nếu có pose.
2. GK đối phương: vị trí, khoảng cách, có trong khung hình không (tách *missing* vs `visible=false`).
3. Mọi đối phương trong vòng cấm / trên đường sút (không chỉ bán kính 2 m).
4. Số đồng đội phía trước bóng / packing (không chỉ `ally_players` trừ 1).
5. Pattern từ **chuỗi event** (throw-in, cross→header, drive nhanh) + nói rõ FOOTPASS không có corner/FK; heuristic hình học có provenance.
6. Provenance: field nào lấy FOOTPASS tactical, field nào YOLO, field nào abstain.

Hiện đạt (1)–(6) khi có tactical HDF5. Thiếu HDF5: location-only xG + YOLO nếu cửa sổ chạy; counts GK/foe vẫn missing (đúng mask). Video-only polygon không thay oracle HDF5.

---

## 10. Hướng xử lý

Đã implement (P8–P10, P9, P12 baseline):

1. Freeze-frame FOOTPASS là nguồn đếm GK/foe/pressure; YOLO bổ sung body_part và camera.
2. Spatial card: provenance + pitch mini + trail + YOLO track + pose trên sân (nếu có `pitch_x`) + inset.
3. Body part: pose vs bóng YOLO, không thì vs đáy bbox; header từ `action_type`.
4. Map team: majority vote + NN-align freeze.
5. `xg_raw` + freeze + `possession_buildup` vào `EvidenceBundle`.
6. Cache overlay invalidate khi đổi sync JSON / overlay version.
7. Location-only xG khi không YOLO/HDF5; ball F1 thấp không withhold nếu có `x/y`.
8. Video-only homography từ pitch polygon; extrapolation 25 frame.
9. Heuristic corner / FK / penalty + provenance `geometry_heuristic`.

**Không đóng bằng engineering:** P11 (cần human GT trên 12 trận). P12 không pretends oracle. Heuristic play_pattern không thay nhãn StatsBomb.

---

*Cập nhật: fusion attack-view thống nhất, location-only xG, video-only polygon homography, YOLO↔freeze role align, play_pattern heuristic, calib UX. P11 còn mở.*
