# Train FOOTPASS 48 trận trên GPU thuê

Kế hoạch chính: TAAD + DST trên **toàn bộ 48 trận train**. Máy đã chọn: **DigitalOcean 1× H100 80 GB** ($4.41/h, scratch 5 TB). Video copy xuống scratch NVMe, không Colab/Drive FUSE. Chi tiết lệnh:

[FOOTPASS/docs/rented-gpu-full-train.md](../../FOOTPASS/docs/rented-gpu-full-train.md)

Chỉ thay `playbyplay_DST_val.json` trong thesis khi `evaluation.py` (τ=0.15, ±12) **vượt** micro-F1 0.719. Job này không sửa calib YOLO, xG domain, hay effectiveness.
