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

Log Streamlit 2026-10-09 01:11 còn `ArrowInvalid`: cột `Value` của bảng spatial không convert được chuỗi (`other`, trước đó `left_foot`) sang double. Streamlit tự sửa kiểu cột. Bảng vẫn có thể hiện, nhưng cảnh báo và kiểu cột hỗn hợp còn đó.

### Demo là một process cũ cộng nhiều lớp cache

Server PID lắng nghe 8504 bắt đầu 2026-10-07. Session key upload đã đổi (`upload_extract_stable_...`) nên bảng sự kiện cũ không hiện lại, và người dùng phải bấm trích lại. Phân tích FOOTPASS thì vẫn dính cache. Reload module vá được import cũ, không vá được mọi số liệu đã cache.

### Test xanh không đo đúng video của người dùng

Các test mới kiểm luật trên frame giả (1 Hz, hoặc vài frame 25 fps sạch). Chúng không phát hiện trường hợp bóng nhảy 500 m/s suốt 140 giây rồi quỹ đạo “ổn” còn lại vẫn sai sân. Không có lần nào ngồi xem clip upload và đánh dấu từng sự kiện đúng hay sai.

---

## Không đụng tới

- Droplet DigitalOcean và job train trên đó.
- Frozen test `game_0`, `game_36`, `game_50` để tune.
- Nhãn gold. P11 vẫn mở.
- Promote `DST_h100`.
- Train lại detector bóng. F1 bóng vẫn dưới cổng 0.80.
