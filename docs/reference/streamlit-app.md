# Ứng dụng Streamlit (`app/streamlit_app.py`)

## Chạy

Từ thư mục `football-attack-thesis`:

```powershell
py -3 -m pip install -e ".[research,vision]"
py -3 -m streamlit run app/streamlit_app.py --server.port 8504
```

Hoặc `attack-analysis serve-demo --port 8504`. Gói tối thiểu là extra `demo`; `python-dotenv` là bắt buộc. Weights `yolo26n-pose.pt` và `yolo26n.pt` đặt ở gốc dự án; dữ liệu FOOTPASS ở `../FOOTPASS`.

## Khởi động

- Nạp `.env` bằng `load_project_environment` (không ghi đè biến môi trường sẵn có).
- `importlib.reload` ba module `browser_video`, `dynamic_calibration`, `pitch_visualization` để sửa code không cần khởi động lại.
- Đường dẫn mặc định:

| Tên | Giá trị |
| --- | --- |
| FOOTPASS | `<workspace>/FOOTPASS` |
| Video (tab FOOTPASS) | `FOOTPASS_VIDEOS_DIR` (mặc định `D:\videos`), qua `VideoPathResolver` |
| xG-v1 | `models/xg-v1.joblib` |
| xG-v2 | `XG_V2_MODEL_PATH` hoặc `models/xg-v2.joblib` |
| Benchmark | `data/benchmarks.jsonl` |
| Ngưỡng event | `models/dst-thresholds-v1.json` |

Không có sidebar; mọi điều khiển nằm trong 3 tab.

## Tab 1: FOOTPASS analysis

1. **Evidence mode:** `oracle` hoặc `predicted`.
2. **Predicted:**
   - Selectbox **Prediction source** lấy từ `models/prediction-sources.json` (mặc định primary). Caption hiển thị id, 12 ký tự đầu SHA-256 và micro-F1 chính thức (τ = 0.15, δ = 12). File được kiểm SHA-256 trước khi dùng.
   - Registry rỗng hoặc lỗi thì quay về selectbox DST/TAAD cũ.
   - Slider ngưỡng confidence (mặc định 0.15).
   - Toggle class-aware dùng `threshold_profile` của nguồn; bật thì tên model thành `{id}_calibrated`. TAAD được ghi chú "ablation only".
3. **Đường dẫn:** play-by-play (oracle: `playbyplay_GT/playbyplay_val.json`; predicted: đường dẫn từ registry) và tactical HDF5 (`FOOTPASS/data/val_tactical_data.h5`).
4. **Observation gap:** 4–15 s, mặc định 8 (gap gây censor, không tạo attack).
5. **Phân tích hiệp:** chọn match half → `load_attack_v2_analysis` (cache). Metric: observed actions, possessions, strict Attack-v2, replay excluded.
6. **Audit shot → Attack-v2:** bảng promoted / censored / unassigned, kết quả cú sút từ gold nếu có.
7. **Bộ lọc sequence:** Confirmed Attack-v2, Shot terminal bị censor, All possessions.
8. **Chi tiết sequence:** status, team, số action, thời lượng, terminal, kết quả cú sút; bảng timeline (frame, time, action, team, x, x_attack, jersey, replay, P(action), evidence); cảnh báo nếu median x attack-view < 52.5 (có thể lật hướng); sân tương tác Altair.
9. **Video preview:** `st.video` mở tại giây đã đổi qua đồng hồ preview.
10. **Cửa sổ calibration + YOLO** (mục dưới).
11. **Kết luận:** nút "Đưa ra kết luận", chỉ bật khi đang chọn Attack-v2.
12. **V1 compatibility ablation:** JSON legacy trong expander.

### Cửa sổ calibration + YOLO (`render_calibrated_window`)

1. Video overlay: Full HD nếu cấu hình, nếu không thì video xử lý.
2. Nhập start/end theo giây video overlay (mặc định sequence ± 1 s), tối đa 30 s (giới hạn cho GPU 6 GB).
3. Đổi frame preview sang frame trận; `load_dynamic_projector` fit homography và bọc trong `PreviewClockProjector`.
4. `VisionConfig`: `image_size` 1280 nếu video rộng ≥ 1280 (ngược lại 960), `field_lines=True`, `player_confidence=0.15`, `keypoint_confidence=0.2`, device CUDA `"0"` nếu có.
5. `run_cached_video_pipeline(..., runs/window-cache, source_tier="predicted_actions_oracle_tactical", dataset_paths=(playbyplay, tactical))`, sau đó gắn báo cáo calibration và `video_clock` vào manifest, rồi `persist_footpass_window_stream`.
6. Hiển thị: Calibration PASS/FAIL, eligible coverage, median/P90 holdout, caption đồng hồ (role, fps, offset, cache key), phân rã direct/interpolated/extrapolated.
7. Overlay: phát video nếu codec H.264, nếu không thì lưới 8 frame mẫu.
8. Metric chất lượng (người/frame, độ phủ bóng, tracking, đối tượng đã hiệu chỉnh) và cổng (Assisted calibration, Window boundary, Terminal evidence, CV event gate).

### Luồng kết luận

1. Không phải Attack-v2 → `insufficient_evidence_verdict`.
2. Attack-v2:
   - Benchmark `central_forward` / `xg` từ `data/benchmarks.jsonl`.
   - Nếu có Brave: tìm "football {terminal} chance quality {match_id}" làm `web_context`.
   - `build_attack_evidence(...)` với xG-v2 (nếu có), `allow_research_only=True`, tactical HDF5.
   - `attack_v2_research_overview`, `as_legacy_bundle`, `EvidenceToolRegistry`.
   - Có `OPENAI_API_KEY` → `OpenAIResponsesClient(model=openai_model(), response_language="Vietnamese").evaluate_v2`; không có → `DeterministicEvidenceClient`. Lỗi bất kỳ → báo lỗi rồi chạy lại bằng client tất định.
3. `render_conclusion_v2`:
   - Tóm tắt: terminal, xG-v2, chance quality, số action.
   - **Chance quality:** xác suất và dải.
   - **Spatial context:** association method, track, khoảng cách GK, số đối thủ, body part; bảng field/value/source (provenance); `plot_freeze_frame` (cầu thủ, người sút, GK, build-up, pose); JSON `xg_raw`.
   - **Benchmark**, **Web context** (chỉ ngữ cảnh), **Tactical read** (structural features, claims kèm evidence id, limitations).

## Tab 2: Exploratory upload

- Luôn chọn video từ máy qua `st.file_uploader` (MP4/MKV/MOV, tối đa 4 GB theo `.streamlit/config.toml`); file lưu theo SHA-256 nội dung vào `runs/upload-cache`, nên chọn lại cùng file dùng lại cache. Không còn danh sách video gắn sẵn.
- Slider số frame YOLO 25–750 (mặc định 250); chọn device.
- `run_cached_video_pipeline(..., runs/upload-cache/vision)` với `VisionConfig(image_size=960, field_lines=True)`, tier `video_only`.
- Luôn kết thúc bằng verdict `insufficient_evidence`: video-only không xác nhận được terminal Attack-v2.

## Tab 3: Annotation

- Tải manifest gold (`attack-gold-v2.0`, tạo bằng `build-gold-manifest`).
- Trận sắp xếp theo split; trận test có cảnh báo.
- Form: period, team, start/end (giây và frame ở 25 fps), label, terminal, completeness, terminal time/frame, cause, body part, shot outcome, beneficiary, last touch, replay, confidence, reason codes, notes, annotator, round.
- Lưu: `GoldSequenceAnnotationV2(...).with_stable_id().validate()`. Trận test vào `data/sealed-test` qua `append_sealed_test_annotation_jsonl`; còn lại vào `data/attack-v2-{split}.jsonl`.

## Cache

| Loại | Hàm / thư mục | Khóa |
| --- | --- | --- |
| `st.cache_data` | `playbyplay_keys`, `load_attack_v2_analysis` | Toàn bộ tham số (gồm ngưỡng class-aware) |
| `st.cache_resource` | `load_gold_outcome_index`, `load_benchmark_store`, `load_dynamic_projector`, `load_video_path_resolver` | Tham số + mtime HDF5 + `calibration_policy=4` |
| Đĩa | `runs/window-cache` | Provenance + `CALIBRATION_CACHE_KEY`, `OVERLAY_CACHE_KEY`, match_key, mtime HDF5 và sync, offset preview |
| Đĩa | `runs/upload-cache/vision` | Provenance của clip |
| Session | `calibrated_window_result_*`, `window_manifest_<attack_id>`, `conclusion_<attack_id>`, `upload_vision_<key>` | |

Muốn chạy lại YOLO cho mọi cửa sổ cũ, đổi `OVERLAY_CACHE_KEY` trong `video_paths.py` hoặc xóa `runs/window-cache`.

## Biến môi trường

`FOOTPASS_VIDEOS_DIR`, `FOOTPASS_FULLHD_VIDEOS_DIR`, `FOOTPASS_VIDEO_SYNC_JSON`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `BRAVE_SEARCH_API_KEY`, `XG_V2_MODEL_PATH`. Mẫu ở `.env.example`; không commit `.env`.
