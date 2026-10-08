# Checklist vận hành FOOTPASS + YOLO overlay

Các lỗi calib 0% / overlay cũ / không phát mp4 thường đến từ môi trường, không từ decoder Attack-v2.

## Lệnh

```powershell
attack-analysis doctor
attack-analysis inspect-video-sync
```

`doctor` gộp `operations` (sync, Full HD, cache key) và `gold`. `inspect-video-sync` chỉ phần video.

## Sync JSON

- File: `FOOTPASS_VIDEO_SYNC_JSON` hoặc `data/footpass-video-sync.json`.
- Schema: `footpass-video-sync/1.0`.
- Mỗi trận cần `preview_offset_sec`, `processing_fps` (thường 25). Sai offset → HDF5 trống → coverage 0%.
- Overlay cache lấy `video_sync_mtime`; sửa JSON rồi **chạy lại cửa sổ** (không dùng cache cũ).

## Dual video

| Biến | Vai trò |
|------|---------|
| `FOOTPASS_VIDEOS_DIR` | File xử lý / extract JSON |
| `FOOTPASS_FULLHD_VIDEOS_DIR` | Preview + YOLO overlay |

Thiếu Full HD → overlay chạy processing (thường thấp hơn 1080p). Cache key hiện tại: `fullhd_pose_crop_ball_fullframe_v6` (bóng full-frame, không tile; overlay v5 cũ được chạy lại).

## FFmpeg / trình duyệt

OpenCV H.264 trên Windows hay fail (OpenH264). Cần `ffmpeg` trên PATH hoặc:

```powershell
python -m pip install "imageio-ffmpeg>=0.5"
```

`doctor.tools.ffmpeg` phải khác `null`.

## Calibration FAIL

Không có nghĩa pipeline hỏng:

- Event FOOTPASS và freeze HDF5 vẫn dùng cho xG.
- YOLO overlay vẫn chạy; tọa độ sân có thể là grass-polygon (P12, không oracle).
- Frame replay / bbox NaN **không** được ngoại suy homography.

## Ball F1

Gate 0.80 chỉ bắt buộc cho path **video-only** (possession/shot từ bóng). Oracle FOOTPASS + location-only xG không chờ gate này. Fine-tune `imgsz=1280` cần GPU: `train-ball-model ... --device 0`.
