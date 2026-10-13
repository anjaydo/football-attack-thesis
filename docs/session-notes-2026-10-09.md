# Ghi lại phiên 2026-10-07 → 2026-10-09

Phạm vi: những việc đã sửa trong `football-attack-thesis` và demo Streamlit (`http://localhost:8504`) sau khi DST H100 trả kết quả, cộng những chỗ còn sai. Đây là biên bản, không phải báo cáo đã xong.

Ràng buộc đã giữ: không promote DST, không đụng droplet DigitalOcean, không đụng frozen test `game_0` / `36` / `50`, không bịa nhãn gold, không gán hiệu quả dạng có/không trước khi có gold.

---

## Việc đã làm

### 1. Chấm DST H100 và không đưa lên primary

Pipeline H100 kết thúc `PIPELINE_EXIT:0` lúc 2026-10-06T16:00:55Z. 15 epoch. Val JSON: 6 hiệp, 5 776 event.

Chấm lại bằng `FOOTPASS/evaluation.py`, τ = 0.15, δ = 12:

| Nguồn | Precision | Recall | Micro-F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `DST_v1` (số đã đăng ký) | 0.735 | 0.703 | 0.719 | 4267 | 1539 | 1803 |
| `DST_h100` | 0.706 | 0.671 | 0.688 | 4076 | 1700 | 1994 |

Lớp shot của H100: precision 0.564, recall 0.463 (31 TP, 24 FP, 36 FN, 67 GT).

Checkpoint `FOOTPASS/runs/DST_Baseline_h100/checkpoints/best_model.pt`, SHA-256 `f0acbab02fefc4a0f4c0a747b8f40197b388ba00426a507762e1526a001ddefa`. JSON H100 SHA-256 `1beba6ae95948f616fb2fea1bf4d631768ea5756aff29f6cbdbf6d386f8768ab`.

Đăng ký ablation `DST_h100` trong `models/prediction-sources.json`, `promoted: false`. Không refit `dst-thresholds-v1.json`.

Decode Attack-v2 so với oracle, biên ±1 giây:

| Nguồn | Possession | Attack | Possession start / end F1 | Attack start / end F1 |
| --- | ---: | ---: | ---: | ---: |
| `DST_v1` (lần chấm trước) | 696 | 42 | 0.647 / 0.652 | 0.386 / 0.591 |
| `DST_h100` | 700 | 38 | 0.609 / 0.608 | 0.333 / 0.452 |

H100 kém hơn `DST_v1` cả micro-F1 lẫn ranh giới attack.

Chi tiết số liệu: [progress-report-2026-10-07.md](progress-report-2026-10-07.md).

### 2. Chuỗi kết thúc bằng shot thì có xG StatsBomb

Thêm `src/attack_analysis/shot_xg.py`. Shot terminal trên attack hoàn chỉnh đi qua `build_attack_evidence`. Shot bị censor và shot giữa chuỗi đá bồi đi qua `_predict_shot_probability`. Thiếu model, hoặc model không mang trạng thái StatsBomb/validated, thì ném `ShotXGRequiredError`.

`models/xg-v2.joblib` (gitignored), SHA-256 `d6a78d180a3d3a05f36eaee6f760e9e3738c138503dfe45197fbb2128c0433ac`. HistGradientBoosting + Platt. StatsBomb open data: 87 738 cú sút, 3 461 trận, 9 688 bàn. Test Brier 0.0803, ECE10 0.0040. Trạng thái `statsbomb_gate_passed_domain_unvalidated`. UI và CLI load với `allow_research_only=True`.

`footpass-ingest --xg-model` chấm shot thay vì bỏ qua đường dẫn model. Ingest `game_18_H1` với DST H100: 10/10 shot có xG. Sáu hiệp val: 55/55 chuỗi kết thúc bằng shot có xG (35 confirmed, 20 censored), min / median / max = 0.015 / 0.061 / 0.254.

Header không nằm trong luật “mọi shot phải có xG”.

### 3. Demo Streamlit

`py -3 -m streamlit run app/streamlit_app.py --server.port 8504`. Process này chạy từ 2026-10-07 và vẫn là process đang listen cổng 8504. Code mới vào UI nhờ Streamlit chạy lại script, cộng `importlib.reload` cho vài module. Cache `@st.cache_data` không tự bỏ số cũ nếu chữ ký hàm không đổi.

### 4. Nhãn phút:giây dưới ô nhập giây

Ô cửa sổ video vẫn nhập bằng giây. Dưới mỗi ô có dòng `Trên video: m:ss.s (N phút S giây)`. Đã kiểm 89.84 giây → `1:29.8`.

### 5. Phản chiếu điểm lệch nửa sân khi vẽ

`pitch_visualization.aligned_event_xy`. Hai điểm cách nhau dưới 1.5 giây mà lệch hơn 40 m và hơn 40 m/s, nhưng lật `x → 105 − x` thì nằm trong 12 m, thì điểm ưu tiên thấp (block, tackle) bị lật khi vẽ. Y không lật. Bảng “Tọa độ sân” ghi “cùng mặt sân”.

Đây chỉ là lớp vẽ. Tọa độ gốc trong event không đổi.

### 6. Đá bồi trong 3 giây được nối thành một attack

`FootpassAttackV2Config.rebound_window_sec = 3.0`. Sau shot hoặc header, nếu cùng đội còn drive / pass / cross / shot / header trong 3 giây, trước khi đối phương kiểm soát, trước restart, trước khoảng im lặng dài, hoặc trước khi đổi hiệp, thì shot đầu không đóng possession.

Oracle `game_18_H1`: shot 105.60 rồi shot 107.64 được gom một attack `atkv2_6452e2a5ecba7ac3a771e005` (90.84–107.76 giây). Số attack của hiệp đó trong lần decode Python giảm từ 10 xuống 9. Cả hai shot vẫn được chấm xG.

Lúc mở tab FOOTPASS analysis (2026-10-09, khoảng 02:11), dropdown vẫn là `atkv2_0c4129e70e2f2adf7dd290a0` (90.8–106.0 giây, xG 0.017) và metric “Strict Attack-v2” vẫn là 10. Trang đang phục vụ cache cũ. Việc nối đá bồi chưa hiện trên demo.

### 7. Luồng upload: trích rồi mới phân tích

Tab Video upload có nút **Trích xuất sự kiện**, rồi chọn chuỗi attack, rồi nút phân tích. Không tự phân tích khi vừa trích. Kết luận đi cùng hàm `conclude_attack_v2` như tab FOOTPASS: xG-v2, chance quality, nhận định LLM hoặc bản deterministic. Không có tactical HDF5 cho video upload. Không gán hiệu quả có/không.

`sequences_from_video_events` đổi `shot_candidate` thành `shot`, `replay_status="live"`, `source_tier=video_only`. Decoder gốc không làm việc này.

### 8. Hết lỗi `unsupported shot outcome: 'unverified'`

`xg_v2._binary_outcome` coi `unverified` (và `unknown` / `unlabelled` / `na`) là thiếu nhãn, không ném `ValueError`. Có test.

### 9. Siết suy sự kiện từ video

Đo file mới nhất `runs/upload-cache/vision/f76405a18ebd9e561855b04f61a1a7a2413fe27263c68ca41da1289d38b12633/91c51e3004ff63ab/observations.jsonl`: 3 500 frame, 140 giây, calibration 0.25, bóng có tọa độ sân trên 1 040 frame, confidence bóng median 0.32. Trong 1 119 cặp bóng liên tiếp có tọa độ, 540 cặp nhanh hơn 40 m/s.

Trước khi siết: 94 sự kiện (68 `possession_start`, 4 `shot_candidate`), 2 attack, trong đó một attack dài 0 giây ở t = 3.6.

Sau khi siết (`event_inference.py`): bóng confidence dưới 0.35 bị bỏ; ở video dày (median khoảng cách frame ≤ 0.15 giây) chỉ giữ đoạn liên tục ≥ 4 mẫu và ≤ 40 m/s; đổi chủ bóng 1–2 frame không thành đường chuyền hay mất bóng; `shot_candidate` phải đi trên cùng quỹ đạo và kết thúc ở x ≥ 96 theo hướng tấn công.

Chạy lại cùng file: 4 sự kiện quanh giây 67–74 (possession, pass, carry, possession), 0 attack, 0 shot. Ba file upload ngắn hơn thì không còn possession ổn định và báo lỗi.

Test synthetic vẫn qua. Chưa xem video để đối chiếu 4 sự kiện còn lại với hình.

### 10. Tách hai phase trên tab Video upload

Hai khung:

- Phase 1 · Trích xuất sự kiện: nút trích, bảng sự kiện, bảng chuỗi tấn công.
- Phase 2 · Phân tích với YOLO, AI và xG: nút **Phân tích với YOLO, AI và xG**. Overlay YOLO, xG và nhận định chỉ hiện sau nút này.

Đã mở `http://localhost:8504`, tab Video upload, thấy đúng hai khung và Phase 2 không tự chạy.

---

## Việc làm còn chưa tốt

### Video upload vẫn không ra sự kiện đáng tin

Suy sự kiện không phải model hành động. Nó gán bóng cho cầu thủ gần nhất trong 3 m, rồi đoán carry / pass / mất bóng / sút từ vận tốc. Detector bóng chưa qua cổng chất lượng (F1 tốt nhất đã ghi là 0.474, cổng 0.80). Homography đa giác cỏ gắn `calibration_confidence = 0.25` và có thể lật sân. Hướng tấn công gắn cứng `team_a` sang phải, `team_b` sang trái.

Lần “sửa cho chính xác” không tìm ra sự kiện đúng. Nó bỏ quỹ đạo nhảy và, trên clip 140 giây vừa đo, không còn attack nào để phân tích. Nếu clip có cú sút thật mà bóng không được bám liên tục tới sát khung thành, cú sút biến mất. Bốn sự kiện còn lại cũng chưa được kiểm bằng mắt.

`shot_candidate` một khi lọt qua luật mới vẫn bị đổi thành `shot` rồi đem chấm xG. Số xG trông như tab FOOTPASS. Nguồn của nó là quỹ đạo video, outcome `unverified`.

### Tách phase chỉ tách chỗ hiện, không tách chỗ chạy

Phase 1 vẫn gọi `run_cached_video_pipeline`: YOLO pose + detect chạy ở nút **Trích xuất sự kiện**, vì không có observation thì không có bảng sự kiện. Phase 2 phát lại video overlay đã cache, rồi gọi `conclude_attack_v2` (fusion từ observation, xG-v2, LLM).

Người dùng yêu cầu trích sự kiện một phase, còn YOLO + AI + xG một phase. Giao diện đúng câu đó. Việc YOLO nặng vẫn nằm trong Phase 1.

### Đá bồi chưa có trên trang FOOTPASS đang mở

Decode Python của `game_18_H1` đã gom hai shot. Tab FOOTPASS analysis trên process Streamlit đang chạy vẫn hiện attack cũ `atkv2_0c4129e70e2f2adf7dd290a0` và 10 attack. `load_attack_v2_analysis` bị cache. Đổi docstring để phá cache đã làm trong code; trang người dùng nhìn lúc 02:11 chưa dùng kết quả mới.

### File DST_v1 trên đĩa không còn

`playbyplay_PRED/playbyplay_DST_val.json` bị ghi đè bởi output H100. Hash primary đã ghim `34ef3f16233bd2e64cc5336e6f9a6e2c623d724c0517e09b91a99665f03c5741` không còn file. Bản H100 được copy sang `playbyplay_DST_h100_val.json` trước khi đăng ký. Byte gốc của `DST_v1` không tìm thấy ở các thư mục MSE cạnh đó. UI bỏ qua primary không kiểm được hash và chọn nguồn verify được đầu tiên. Mất file chưa được khôi phục.

### xG video và xG FOOTPASS không cùng một loại bằng chứng

xG-v2 train trên StatsBomb, trạng thái `domain_unvalidated`. Fusion bỏ chặn sớm vì ball F1 thấp: shot vẫn được chấm khi thiếu vị trí, bằng missingness mask. Trên upload, shot là ước lượng quỹ đạo. Trên FOOTPASS, shot là nhãn action. Cùng một ô “xG-v2” cho cả hai.

### Lớp vẽ “cùng mặt sân” có thể giấu điểm đúng

Block ở x = 9.9 ngay sau shot ở x ≈ 92 được lật thành x ≈ 95 khi vẽ, vì khoảng thời gian ngắn và vận tốc vô lý. Nếu tọa độ block là điểm chạm thật ở nửa sân kia, bản đồ sẽ đặt nó nhầm sang phía tấn công. Annotation có chữ “đã phản chiếu”, nhưng người xem vẫn thấy một điểm không có trong dữ liệu gốc.

### Lỗi bảng FOOTPASS chưa sửa

Log Streamlit 2026-10-09 01:11 còn `ArrowInvalid`: cột `Value` của bảng spatial không convert được chuỗi (`other`, trước đó `left_foot`) sang double. Đã sửa sau đó: cột `Value` ép về chuỗi trước khi hiển thị.

### Demo là một process cũ cộng nhiều lớp cache

Server PID lắng nghe 8504 bắt đầu 2026-10-07. Session key upload đã đổi (`upload_extract_stable_...`) nên bảng sự kiện cũ không hiện lại, và người dùng phải bấm trích lại. Phân tích FOOTPASS thì vẫn dính cache. Reload module vá được import cũ, không vá được mọi số liệu đã cache.

### Test xanh không đo đúng video của người dùng

Các test mới kiểm luật trên frame giả (1 Hz, hoặc vài frame 25 fps sạch). Chúng không phát hiện trường hợp bóng nhảy 500 m/s suốt 140 giây rồi quỹ đạo “ổn” còn lại vẫn sai sân. Không có lần nào ngồi xem clip upload và đánh dấu từng sự kiện đúng hay sai.

---

## Upload Phase 1 dùng FOOTPASS TAAD (videoplayback.mp4)

Clip highlight 140 s, 25 fps. Người dùng đếm bằng mắt 5 đợt tấn công.

`footpass_video_events.py` thay suy luận quỹ đạo bóng:

1. Đoạn toàn cảnh: frame có ít nhất 6 cầu thủ YOLO và chiều cao bbox trung vị ≤ 15% khung hình, làm mượt 13 frame, giữ đoạn ≥ 3 s. Cận cảnh, ăn mừng, replay và end card bị loại.
2. Mỗi đoạn chạy TAAD (`runs/TAAD_Baseline_h100/checkpoints/best_model.pt`): 26 track YOLO nhiều frame nhất vào 26 ô, cửa sổ 50 frame, hai lượt lệch 25 frame lấy trung bình, NMS 25 frame, ngưỡng 0.15 như FOOTPASS. Kết quả cache ở `taad_detections.json` cạnh `observations.jsonl`.
3. Một đoạn là một đợt tấn công. Đội tấn công là đội thắng phiếu theo điểm TAAD trên drive/pass/cross/shot/header. Kết thúc chọn trong các shot/header (và block của chính đội tấn công) cách cú cuối ≤ 1 s, ưu tiên cầu thủ đội tấn công, track YOLO ít khi gọi là thủ môn, shot/header thật hơn block. Decode từng đoạn riêng bằng `decode_event_stream_v2`.

Kết quả trên clip, khớp 5 đoạn người dùng đếm:

| Đoạn | Đội | Kết thúc | Track |
| --- | --- | --- | --- |
| 0:00.0–0:14.5 | team_a | đánh đầu 12.20 s | 782 |
| 0:27.3–0:36.8 | team_a | sút 34.64 s | 1437 |
| 0:43.3–0:53.2 | team_b | sút 50.60 s | 2050 |
| 1:06.1–1:21.9 | team_b | sút 79.52 s (TAAD gọi là block) | 3173 |
| 1:37.7–1:44.2 | team_b | sút 100.56 s | 4319 |

Phase 2 chạy cả 5 chuỗi không lỗi. xG-v2 0.019–0.038, band low (chuỗi 4 medium).

Video YOLO ở Phase 2 chỉ phát đoạn của chuỗi đang chọn: từ 1 s trước sự kiện đầu tới 2 s sau cú dứt điểm. `extract_browser_clip` cắt bằng FFmpeg (H.264, không tiếng) vào `runs/upload-cache/clips` và dùng lại nếu đã cắt. Không có FFmpeg thì phát video đầy đủ với `start_time`/`end_time`. Chuỗi 1 ra 14.2 s, chuỗi 4 ra 1:05.1–1:21.5 (16.4 s).

### Còn yếu

- Tọa độ sân sai. GK distance 31–63 m cho mọi cú sút, kể cả pha sút cửa trống ở 79.5 s. xG vì vậy không phản ánh cơ hội thật. (Đã sửa bằng hiệu chỉnh điểm mốc, xem phần 2026-10-10.)
- Không có kết quả cú sút. `shots_on_target` và `goals` là số nguyên nên “không biết” thành 0. Nhận định AI viết “không trúng đích, không ghi bàn” dù clip có ăn mừng sau các đoạn 1, 3, 4.
- Đoạn 5 kết thúc sớm: TAAD sút 100.56 s, pha lộn xộn trước khung thành kéo tới khoảng 103 s.
- Đội theo phiếu đoạn, không theo áo từng cầu thủ. `team_a`/`team_b` là cụm màu YOLO, chưa gắn tên đội.
- Replay quay từ camera chính sẽ bị đếm thành đợt tấn công thứ hai.
- Cache vision băm cả mã nguồn package. Sửa code bất kỳ thì YOLO chạy lại 3500 frame (khoảng 8 phút).
- Sửa code khi Streamlit đang chạy làm session cũ giữ class `ObservedActionV2` cũ. `pitch_visualization` đã chuyển sang kiểm theo thuộc tính, `xg_v2` vẫn `isinstance` nên cần trích lại sau khi sửa.

---

## Hiệu chỉnh sân bằng điểm mốc và thử DST (2026-10-10)

### Tọa độ sân từ điểm mốc

`pitch_calibration.py` thay homography đa giác cỏ trên GPU. Model Roboflow `football-pitch-detection.pt` (YOLO-pose, 32 điểm mốc, giấy phép AGPL, gitignored ở `models/`) chạy hai cỡ ảnh 640 và 1280. Mỗi frame fit homography RANSAC trong đơn vị mét, sân 105×68. Fit bị loại nếu ít hơn 5 inlier, RMSE trên 1 m, sai số chiếu ngược trên 1% đường chéo ảnh, hoặc điểm ảnh rơi ra ngoài sân quá 20 m. Bốn điểm luôn fit khớp tuyệt đối, nên ngưỡng 5 inlier là để chặn nhãn điểm mốc sai ở góc quay chéo. Sau đó lấy trung vị lưới 3×3 điểm ảnh trong ±3 frame, nối khoảng trống ≤ 25 frame.

Trên `videoplayback.mp4` (3 500 frame): 1 840 frame fit trực tiếp, 80.3% frame có calibration sau khi làm mượt, trung vị 6 inlier, RMSE 0.29 m. Hiệu chỉnh mất khoảng 7.5 phút trên GPU 6 GB, điểm mốc thô cache ở `runs/upload-cache/pitch/`. Vẽ lại vạch sân lên frame 30, 34.6, 99 và 100.5 giây (góc nhìn khung thành trái): vòng cấm, vòng cung và khung thành nhỏ trùng vạch thật.

RANSAC của OpenCV ngẫu nhiên, nên digest calibration thay đổi mỗi lần và kéo YOLO chạy lại. Đã cố định seed.

Overlay YOLO vẽ mô hình sân màu hồng. Nhãn overlay ghi `VIDEO-ONLY pitch (landmarks)`.

### Hướng tấn công

Sự kiện TAAD giờ được lật theo hướng tấn công như FOOTPASS: đội tấn công luôn hướng về x = 105. Hướng lấy từ cú dứt điểm (x > 52.5 là khung thành phải), không có thì lấy trung vị x của đội tấn công. Đội phòng ngự nhận hướng ngược lại. `raw_left_to_right` được truyền vào `decode_event_stream_v2`.

Kết quả: cú sút nằm ở x ≈ 95–101, y ≈ 24–36. Tackle/block của đội phòng ngự trong đợt 1 nằm ở x ≈ 9 theo hướng của đội đó.

### Đoạn toàn cảnh

`BroadcastViewConfig.min_players` từ 6 xuống 4. Ở 77.8–78.3 giây camera chính vẫn toàn cảnh (cầu thủ cao 8–9% khung) nhưng chỉ thấy 4–6 cầu thủ gần khung thành. Với 6, đợt 4 bị cắt ở 78.0 giây và cú sút cửa trống 79.5 giây rơi sang đoạn không có shot. Với 3, 4, 5 hai lần chạy YOLO cho cùng 5 đoạn.

### DST_h100 không chạy được trên video phát sóng

Đã nối TAAD → DST (`footpass_dst.py`): gán track YOLO vào 26 ô vai trò theo vị trí trung bình so với mẫu vai trò FOOTPASS, dựng team vector (x, y, vx, vy, obs) và logit TAAD giống `run_DST_on_matches.py`.

Trên clip, DST trả 0 sự kiện. Decoder lặp một cặp token, không có EOS, frame dự đoán vượt quá độ dài đoạn.

Kiểm trên FOOTPASS val `game_18_H1` (không phải frozen test), cửa sổ 750 frame, logit tổng hợp từ nhãn val (chỉ để chẩn đoán):

| Đầu vào | Kết quả DST |
| --- | --- |
| 22 cầu thủ, 750 frame đủ | 15 sự kiện, khớp gần như toàn bộ 13 nhãn |
| Giữ 14 ô, còn lại -15 | 0 sự kiện, lặp token ở frame 0 |
| Giữ 14 ô, ô thiếu điền vị trí mẫu, obs = 0 | 0 sự kiện, EOS ngay ở frame 0 |
| 22 cầu thủ, chỉ 400 frame giữa có logit, hai đầu giữ vị trí | 0 sự kiện |

`DST_Dataset` luôn cắt cửa sổ 750 frame đầy đủ, mọi cầu thủ có tọa độ từ tracking (obs chỉ báo có bbox hay không). DST_h100 chưa từng thấy ô trống hay cửa sổ đệm, và sụp khi gặp. Video phát sóng thấy khoảng 10 cầu thủ, đoạn toàn cảnh dài 150–400 frame. Muốn dùng DST cho upload phải train lại với cầu thủ bị che ngẫu nhiên và cửa sổ ngắn. Chưa làm.

App vẫn chạy DST sau TAAD ở chế độ research, nhưng khi không ra sự kiện nào thì hiện cảnh báo nêu lý do trên, không hiện tab rỗng. Registry không đổi, DST_h100 không promote.

### Kết quả trên app (2026-10-10)

Phase 1 trên `videoplayback.mp4`: 5 đoạn, 54 sự kiện, 5 chuỗi tấn công, cột “Hướng tấn công” đúng với video (đợt 2 và 5 về khung thành trái).

Phase 2 đợt 1 (đánh đầu 12.20 s): GK distance 5.6 m (trước đó 31–63 m cho mọi cú sút), xG-v2 0.086, band medium.

Đợt 4 giờ kết thúc ở đánh đầu 74.36 s, x ≈ 64, không phải cú sút cửa trống 79.5 s. Trong 77.5–81.5 s TAAD có shot 0.37 (77.68 s) và block 0.32 (79.48 s), nhưng background vẫn là argmax nên peak picking kiểu FOOTPASS bỏ qua. Lần chạy YOLO trước, block 79.52 s may mắn thắng argmax. Chưa sửa: muốn bắt các cú dứt điểm yếu này phải đổi luật peak picking, khác FOOTPASS.

## Kết quả cú dứt điểm từ video (2026-10-12)

xG chỉ đo chất lượng cơ hội; trước đây upload luôn để `outcome="unverified"`, nên AI viết “không ghi bàn” cho cả bàn thắng thật. Module `shot_outcome.py` ước tính kết quả thực tế, gắn vào action dứt điểm với qualifier `outcome_source:video_scoreboard` / `video_ball`. Đây là ước tính từ video, không phải nhãn gold.

- **Scoreboard OCR** (RapidOCR ONNX, CPU): dò đầy đủ dải trên-trái một lần để tìm tên hai đội và hai ô tỷ số, các giây sau chỉ nhận dạng 4 ô đó (~0.07 s/frame). Chỉ nhận lần đọc khi tên hai đội khớp, nên replay/cận cảnh không có scoreboard bị bỏ qua thay vì đọc sai. “T/I/l” một ký tự đổi thành 1, “O” thành 0. Cả video 140 s mất khoảng 30–40 s, cache `scoreboard.json` cạnh observations.
- **Luật**: bàn thắng khi tổng tỷ số tăng 1 sau cú sút, có ít nhất 2 lần đọc cùng tỷ số mới, và trước cú sút của đợt kế tiếp. Không thành bàn khi tỷ số giữ nguyên ít nhất 10 s sau cú sút. Còn lại là chưa rõ.
- **Bóng YOLO trong khung thành**: bóng conf ≥ 0.25, sau vạch vôi (x′ ≥ 104.5 theo hướng tấn công) và giữa hai cột ±1 m, ít nhất 2 frame trong 3 s sau cú sút. Chỉ dùng khi scoreboard không kết luận được. Trên clip này không có frame nào đạt, vì bóng trong lưới bị che và detector bóng yếu.
- Scoreboard không gắn được với team_a/team_b, nên bàn thắng quy cho đội đang tấn công.
- `metrics` trước đây chỉ đếm `event_type == "shot"`, nên đợt kết thúc bằng đánh đầu có shots = goals = 0. Giờ đánh đầu là terminal `headed_attempt` của attack được tính là dứt điểm; header thường (phá bóng, chuyền đầu) vẫn không tính.
- Clip Phase 2 kéo dài tới 3 s sau cú sút (trước là 2 s).

Trên `videoplayback.mp4` (đối chiếu bằng mắt, không phải gold):

| Đợt | Dứt điểm | Ước tính | Căn cứ | Thực tế |
| --- | --- | --- | --- | --- |
| 1 | đánh đầu 12.2 s | bàn thắng | 0-0 → 1-0 lúc 19 s | bàn |
| 2 | sút 34.6 s | không thành bàn | giữ 1-0 tới 50 s | không bàn |
| 3 | sút 50.6 s | bàn thắng | 1-0 → 2-0 lúc 57 s | bàn |
| 4 | sút 79.0 s (trước là đánh đầu 74.4 s) | bàn thắng | 2-0 → 3-0 lúc 84 s | bàn |
| 5 | sút 100.3 s | chưa rõ | scoreboard biến mất sau 106 s, chưa kịp lên 4-0 | bàn |

### Đợt 4: dứt điểm yếu bị TAAD bỏ sót (2026-10-13)

Đợt 4 từng kết thúc ở cú đánh đầu giữa sân 74.4 s (x′ ≈ 60), nên clip dừng trước bàn thắng và xG tính cho cú đánh đầu xa. Thực tế: số 10 đội đỏ đi bóng qua thủ môn rồi sút ở 79.0–79.1 s, bóng vào lưới khoảng 79.7 s.

- `weak_shot_detections`: đỉnh "shot" của TAAD không thắng argmax (thua lớp background) thì peak-picking của FOOTPASS bỏ qua. Các đỉnh ≥ 0.15 này chỉ dùng làm dứt điểm dự phòng khi terminal cách khung thành hơn 30 m (x′ < 75); đỉnh dự phòng phải ở x′ ≥ 83, gần trục giữa, không thuộc track thủ môn. Lý do ghi `weak_finish_after_far_terminal`. 4 đợt còn lại không đổi.
- Đỉnh yếu của TAAD rơi vào thủ môn vàng vừa bị qua người, nên cầu thủ dứt điểm được gán lại cho người cuối cùng có bóng YOLO (conf ≥ 0.25) cách chân trong vòng một chiều cao người, trong 1 s trước đó (track 3718).
- `infer_goalkeeper` loại cầu thủ dứt điểm: YOLO lúc thì gắn nhãn "goalkeeper" cho số 10.
- `infer_body_part_from_pose`: chọn quả bóng gần chân cầu thủ dứt điểm thay vì quả đầu tiên trong frame (frame có bóng giả tĩnh). Chỉ là đầu khi bóng cao hơn vai; bóng ngang gối/hông tính là chân bên đó. Trước đó bóng nảy ngang hông bị coi là đánh đầu.
- Kết quả: sút từ (96.1, 28.7), 10.4 m, góc 0.61 rad, chân phải, xG 0.06 → 0.22. Clip Phase 2 giờ chạy 65.1–82.0 s, thấy bóng vào lưới.
- Giới hạn: thủ môn vàng thật nằm sau cầu thủ dứt điểm nhưng YOLO gắn nhãn "player"; hậu vệ đứng gần khung thành (3090) bị nhận là thủ môn ở 5.1 m. Đổi chỗ thủ môn trong xG-v2 gần như không đổi kết quả (0.225 cả hai trường hợp), vì model không có feature “thủ môn đã bị vượt qua / khung thành trống”; xG chủ yếu theo khoảng cách và góc. Cú sút từ sát đường biên ngang vòng cấm, góc hẹp, nên 0.22 là trần của model này.

Khóa cache YOLO gồm hash mã nguồn (`content_tree_sha256`), nên sau mỗi lần sửa code, lần trích đầu tiên chạy lại YOLO khoảng 10 phút.

---

## Không đụng tới

- Droplet DigitalOcean và job train trên đó.
- Frozen test `game_0`, `game_36`, `game_50` để tune.
- Nhãn gold. P11 vẫn mở.
- Promote `DST_h100`.
- Train lại detector bóng. F1 bóng vẫn dưới cổng 0.80.
