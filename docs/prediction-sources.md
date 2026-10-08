# Nguồn predicted FOOTPASS và quy trình thay DST

Attack-v2 đọc event predicted qua registry `models/prediction-sources.json`.
Mỗi entry ghim một file play-by-play bằng SHA-256, kèm micro-F1 chính thức của
FOOTPASS `evaluation.py` (τ=0.15, δ=12) trên 6 hiệp val (game 18/24/47).

| Quy tắc | Ý nghĩa |
|---|---|
| Đúng một `primary` | Attack-v2 CLI/Streamlit dùng mặc định |
| DST lên primary khi micro-F1 **lớn hơn** primary hiện tại | Bằng điểm thì vẫn là ablation |
| TAAD thô luôn là `ablation` | DST mới là đường decode của luận văn |
| Không ghi đè JSON cũ | File mới được copy thành `playbyplay_<id>_val.json` |
| Hash lệch thì từ chối chạy | File bị sửa sau khi đăng ký sẽ không được dùng |

## Lệnh

```powershell
attack-analysis prediction-sources                       # danh sách + trạng thái hash
attack-analysis footpass-ingest game_18_H1 --prediction-source DST_v1 `
  --tactical-h5 ..\FOOTPASS\data\val_tactical_data.h5 --output runs\g18h1.json
attack-analysis swap-prediction-source <json> --kind DST --id <id> --dry-run
```

`footpass-ingest --prediction-source` tự đặt `--source predicted`, dùng `id`
làm `model_name`, và ghi khối `prediction_source` (id, role, sha256, F1) vào output.
Đường positional cũ (`footpass-ingest <json> <match_key>`) vẫn chạy.

Streamlit tab FOOTPASS → `predicted` hiện selectbox lấy từ registry, mặc định
primary, kèm sha256 và F1. Toggle class-aware chỉ bật khi entry có
`threshold_profile` riêng.

## Thay DST sau khi H100 chạy xong

Không SSH/scp vào droplet từ pipeline này. Khi đã có file DST val trên máy
(bạn tự tải về, ví dụ `FOOTPASS\playbyplay_PRED\incoming\playbyplay_DST_val.json`):

1. Kiểm tra gate, chưa ghi registry:

   ```powershell
   attack-analysis swap-prediction-source ..\FOOTPASS\playbyplay_PRED\incoming\playbyplay_DST_val.json `
     --kind DST --id DST_h100 --dry-run
   ```

2. Đăng ký thật (ghim checkpoint nếu đã tải `best_model.pt` của DST):

   ```powershell
   attack-analysis swap-prediction-source ..\FOOTPASS\playbyplay_PRED\incoming\playbyplay_DST_val.json `
     --kind DST --id DST_h100 `
     --checkpoint ..\FOOTPASS\runs\DST_Baseline_h100\checkpoints\best_model.pt `
     --note "DST trained on H100 from TAAD_h100 logits"
   ```

   - `promoted: true`: `DST_h100` thành primary, `DST_v1` xuống ablation.
   - `promoted: false`: giữ `DST_v1`; `DST_h100` được lưu làm ablation để so sánh.

3. Chỉ khi promoted, refit threshold class-aware cho file mới (tuning game 18,
   held-out 24/47), rồi sửa `threshold_profile` của entry `DST_h100` thành
   `dst-thresholds-v2.json`:

   ```powershell
   attack-analysis calibrate-footpass-events ..\FOOTPASS\playbyplay_GT\playbyplay_val.json `
     ..\FOOTPASS\playbyplay_PRED\playbyplay_DST_h100_val.json models\dst-thresholds-v2.json `
     --tuning-match game_18_H1 --tuning-match game_18_H2
   ```

   Không dùng `dst-thresholds-v1.json` cho file DST mới; nó được fit trên DST cũ.

4. Chạy lại ingest 6 hiệp với `--prediction-source DST_h100`, so số attack/shot
   với `DST_v1`, cập nhật [action-model-card.md](action-model-card.md).

Frozen test (game 0/36/50) không tham gia chọn nguồn. Release vẫn `research_only`.

## Kết quả DST H100 (2026-10-07)

`DST_h100` đã được chấm local và đăng ký ablation. Micro-F1 0.688 không lớn hơn
`DST_v1` 0.719, nên primary không đổi. File canonical là
`FOOTPASS/playbyplay_PRED/playbyplay_DST_h100_val.json`.

JSON này từng được ghi đè lên `playbyplay_DST_val.json`. Hash đã ghim của
`DST_v1` (`34ef3f16…`) không còn khớp file ở đường dẫn cũ. Đừng dùng đường dẫn
đó làm `DST_v1`. Cần bản JSON gốc (đúng hash) nếu muốn chạy lại primary.
Threshold class-aware không được refit, vì nguồn này không được promote.
