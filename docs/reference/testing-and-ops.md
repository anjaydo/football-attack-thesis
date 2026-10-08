# Cài đặt, kiểm thử và vận hành

## 1. Cài đặt (Windows, máy RTX 4050 6 GB)

```powershell
Set-Location football-attack-thesis
py -3 -m pip install -e ".[research,vision,footpass]"
Copy-Item .env.example .env   # rồi điền key, không commit
```

- Dùng `py -3` (Python 3.11, Torch 2.1 + CUDA 12.1, Ultralytics 8.4). `python` có thể không nằm trong PATH.
- Đường dẫn workspace có chữ có dấu ("Tài liệu"); nên dùng đường dẫn tương đối.
- FFmpeg: dùng bản hệ thống hoặc `imageio-ffmpeg` (cần để phát overlay trong trình duyệt).

## 2. Chạy test

```powershell
$env:PYTHONPATH="src"; py -3 -m unittest discover -s tests -p "test_*.py"
```

Kết quả gần nhất: 332 OK, 1 skip. Test không cần GPU, video thật hay key API; backend YOLO/OpenCV được giả lập trong test tích hợp.

| Nhóm | File test |
| --- | --- |
| Hợp đồng và provenance | `test_research_contracts`, `test_provenance`, `test_observations`, `test_env`, `test_environment` |
| FOOTPASS và Attack-v2 | `test_footpass_research`, `test_footpass_terminal_inference`, `test_attack_v2`, `test_attack_v2_shot_insights`, `test_possession_relative_coords`, `test_v2_adapters`, `test_event_calibration`, `test_prediction_sources` |
| Vision | `test_vision_integration`, `test_vision_refine`, `test_dynamic_calibration`, `test_field_geometry`, `test_geometry_motion`, `test_roi_team`, `test_scene`, `test_video_paths`, `test_browser_video` |
| Pipeline và cache | `test_pipeline`, `test_upload_pipeline`, `test_window_analysis`, `test_quality`, `test_event_inference` |
| Fusion, xG, LLM | `test_fusion_overview`, `test_attack_evidence`, `test_xg`, `test_xg_data`, `test_xg_v2`, `test_xg_training_v2`, `test_agent_tools`, `test_web_search`, `test_pitch_visualization` |
| Đánh giá và release | `test_sequence_evaluation`, `test_research_evaluation`, `test_detection_evaluation`, `test_tracking_evaluation`, `test_gold_annotations`, `test_release_control` |
| Hạ tầng học | `test_action_spotting_v2`, `test_temporal_model`, `test_training`, `test_yolo_dataset`, `test_soccernet` |
| Legacy | `test_scoring`, `test_segmentation`, `test_possession` |

## 3. Script trong `runs/`

Chạy bằng `py -3 runs/<tên>.py` từ `football-attack-thesis` (script tự thêm `src` vào `sys.path`).

| Script | Mục đích | Output |
| --- | --- | --- |
| `compare_prediction_sources.py` | Decode Attack-v2 trên 6 hiệp val cho oracle và mọi nguồn trong registry; đếm possession/attack, boundary F1 ± 1 s so với oracle | `runs/verification-v2/prediction-source-comparison.json` |
| `diagnose_attack_start.py` | Ghép attack oracle với attack của primary theo điểm kết thúc ± 1 s, đo lệch điểm bắt đầu | `runs/verification-v2/attack-start-diagnosis.json` |
| `sngs033_bench.py` | Clip 50 frame SNGS-033, chạy baseline và tiled-pose, benchmark bóng | `runs/sngs033-bench/` |
| `download_train_gs.py` | Tải SoccerNet GameState train | `../sn-gamestate/data/SoccerNetGS` |
| `finetune_ball.py` | Giải nén sequence, xuất `data/yolo-ball-v2`, chạy 1 epoch smoke | `data/yolo-ball-v2`, run smoke |
| `train_ball_full.py` | Fine-tune `yolo26s` 100 epoch ở 1280 rồi benchmark lại | `data/runs/soccernet-ball-1280`, `runs/sngs033-bench/finetuned-summary.json` |
| `resume_ball.py` | Chạy tiếp từ `last.pt` thành run mới (tránh `resume=True` nạp nhầm dataset) | `data/runs/soccernet-ball-1280-cont` |

Script phía FOOTPASS: `FOOTPASS/scripts/local_taad_val.ps1` chạy TAAD trên 6 hiệp val, chuyển sang JSON, chấm bằng `evaluation.py`, rồi dọn thư mục staging.

## 4. Vận hành hằng ngày

| Việc | Lệnh |
| --- | --- |
| Kiểm môi trường, video, gold | `py -3 -m attack_analysis doctor` |
| Kiểm đồng bộ video | `py -3 -m attack_analysis inspect-video-sync --project-root .` |
| Tiến độ gold | `py -3 -m attack_analysis gold-status` |
| Xem registry | `py -3 -m attack_analysis prediction-sources` |
| Thử thay DST | `py -3 -m attack_analysis swap-prediction-source <file> --kind DST --id DST_h100 --dry-run` |
| Chạy giao diện | `py -3 -m streamlit run app/streamlit_app.py --server.port 8504` |

Chi tiết xử lý sự cố (calibration 0%, overlay cũ, mp4 không phát): [../ops-checklist.md](../ops-checklist.md).

## 5. Quy trình thay DST khi có kết quả H100

1. `swap-prediction-source <file> --kind DST --id DST_h100 --dry-run`: xem F1 chính thức và vai trò dự kiến.
2. Bỏ `--dry-run`, thêm `--checkpoint <best_model.pt>` để đăng ký. File được sao chép thành `playbyplay_DST_h100_val.json` và ghim SHA-256.
3. Chỉ khi được promote (micro-F1 > 0.719): fit lại ngưỡng
   `calibrate-footpass-events <GT> <file> models/dst-thresholds-v2.json --tuning-match game_18_H1 --tuning-match game_18_H2`,
   rồi cập nhật `threshold_profile` của nguồn.
4. Chạy lại `runs/compare_prediction_sources.py`, cập nhật [../action-model-card.md](../action-model-card.md).

Chi tiết: [../prediction-sources.md](../prediction-sources.md).

## 6. Ràng buộc nghiên cứu

- Không mở frozen test (game 0/36/50) ngoài luồng test cuối có ledger.
- Không tạo nhãn người giả.
- Không khóa ngưỡng bóng trên lát benchmark 50 frame.
- Không đưa ra nhãn hiệu quả dạng categorical.
- Release giữ `research_only`.
