# Lỗi và giới hạn ở mức mã nguồn

Danh sách này được lập khi đọc toàn bộ mã nguồn ngày 2026-10-06. Đây là phát hiện từ đọc code, **chưa được sửa và phần lớn chưa được tái hiện bằng test**. Vấn đề cấp pipeline/vận hành (calibration 0%, ball F1, đồng bộ video, nhãn gold) xem [../pipeline-issues.md](../pipeline-issues.md) và [../progress-report-2026-10-06.md](../progress-report-2026-10-06.md).

## Mức ảnh hưởng cao

| # | Vị trí | Vấn đề | Hệ quả |
| --- | --- | --- | --- |
| K1 | `app/streamlit_app.py` + `pipeline.validate_video_pipeline_source_contract` | Khi calibration không có frame hợp lệ, UI truyền `projector=None` nhưng vẫn để `source_tier="predicted_actions_oracle_tactical"`; hợp đồng tier này yêu cầu projector | Nhánh dự phòng "vẫn chạy YOLO, dùng tọa độ video-only" ném `ValueError` thay vì chạy |
| K2 | `video_paths.PreviewClockProjector` + `pipeline.run_video_pipeline` | Wrapper không có thuộc tính `.report`, nên pipeline không ghi `calibration.json` và không áp cổng trạng thái calibration | Trạng thái calibration chỉ có trong dict trả về (Streamlit tự chèn), không có trong `manifest.json` trên đĩa |
| K3 | `fusion.shot_type_evidence` | Heuristic penalty dùng x ≥ 99 và \|y − 34\| ≤ 4; chấm phạt đền ở x = 94 | Penalty gần như không bao giờ được nhận; cú sút trong khu 5 m50 có thể bị gắn nhầm `penalty` |
| K4 | `quality.analyze_observation_quality` | Keypoint 5 phần tử (đã chiếu sân) luôn được tính là visible | `keypoint_coverage` bị thổi phồng trên run đã hiệu chỉnh |
| K5 | `attack_evidence.frame_for_terminal` | Chọn frame YOLO gần nhất mà không giới hạn khoảng cách | Có thể ghép terminal với frame cách nhiều giây; lý do `terminal_frame_missing` gần như không xảy ra |

## Mức trung bình

| # | Vị trí | Vấn đề |
| --- | --- | --- |
| K6 | `mcp_server.py` | `tools/list` trả schema kiểu OpenAI (`parameters`, `strict`) thay vì `inputSchema` của MCP; tên công cụ lạ hoặc dòng JSON lỗi làm sập vòng lặp stdio |
| K7 | `attack_evidence.as_legacy_bundle` + `web_search` | Id `web:N` của Brave trở thành evidence id được phép trích dẫn, và id bắt đầu lại từ `web:0` mỗi lần tìm |
| K8 | `llm.OpenAIResponsesClient.evaluate_v2` | Không có vòng thử lại theo hợp đồng, không báo lỗi khi vượt số vòng công cụ, không kiểm lại claims cuối, không đối chiếu `xg_probability` / `chance_quality` của LLM với bundle |
| K9 | `fusion.merge_fusion_context` | Provenance `body_part` ghi `yolo_pose` cả khi giá trị đến từ event (nếu người sút có keypoint) |
| K10 | `fusion._xg_raw_from_frame` | Khi không biết đối thủ, `defending_players` đếm cả đồng đội trong bán kính 2 m |
| K11 | `xg_data` và fusion | `under_pressure` khi train lấy từ cờ StatsBomb (thường thiếu); khi suy luận lấy từ khoảng cách 2 m, nên hai định nghĩa khác nhau |
| K12 | `event_calibration` | Đánh giá dùng `max(baseline 0.15, ngưỡng lớp)`, còn ingest dùng `max(min_confidence, ngưỡng lớp)`; ngưỡng dưới 0.15 cho kết quả khác nhau |
| K13 | `dynamic_calibration.DynamicPitchProjector.for_frame` | Báo cáo coi frame không có hàng HDF5 là ineligible, nhưng `for_frame` vẫn có thể trả projector ngoại suy cho frame đó |
| K14 | `pipeline` (CLI `run-pipeline`) | Luôn `video_only`; `xg_model_path` chỉ được băm, không dùng; decode gán cứng `match_id = tên file`, `period = 1` |

## Mức thấp và giới hạn thiết kế

- `vision_pipeline`: trọng tài bị tính là cầu thủ; điểm ngoài sân bị bỏ không có biên; nguồn calibration (oracle hay polygon) không được ghi vào JSONL ngoài `calibration_confidence = 0.25`; tốc độ đặt lại khi chuyển giữa có và không có hiệu chỉnh.
- `team.py`: nhãn `team_a/team_b` có thể đổi sau scene cut.
- `footpass_terminal_inference`: điều kiện 1/3 cuối sân thực chất chỉ còn `x ≥ 70`; biến `last_controlled_team` ở pass 2 không được dùng.
- `attack_v2.FootpassAttackV2Config.contested_action_types` được khai báo nhưng không đọc.
- `v2_adapters`: mặc định `replay_status="unknown"` khiến mọi possession từ video thành `uncertain`.
- `gold_annotations`: `validate-gold-annotations --allow-frozen-test` và `gold-intra-rater --allow-frozen-test` đọc nhãn test mà không kiểm niêm phong; chỉ `evaluate-attack-v2` thực thi niêm phong. Trường `attack_label_kappa` thực ra tính trên cả 4 nhãn.
- `upload_pipeline`: băm SHA-256 toàn bộ video ở mỗi lần gọi (chậm với video lớn).
- `xg_v2`: script train không bao giờ ghi `status="validated"`, nên runtime luôn cần `allow_research_only=True`.
- `pyproject.toml`: không extra nào khai báo phụ thuộc cho MCP (server tự viết nên hiện không cần).
- `docs/cli-guide.md` chỉ phủ 17/51 lệnh và có thông tin cũ (số test 53, `attacks.json`, `--imgsz 960`); dùng [cli.md](cli.md) làm tham chiếu đầy đủ.
- Mọi phép mirror chỉ lật trục x (phản xạ, không xoay), nhất quán trong toàn bộ code.
