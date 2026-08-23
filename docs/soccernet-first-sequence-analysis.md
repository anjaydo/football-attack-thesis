# Phân tích SoccerNet GameState sequence đầu tiên

## Dữ liệu

- Nguồn: SoccerNet Game State Reconstruction, downloader/archive chính thức.
- Split: `valid`.
- Sequence đầu theo thứ tự archive: `SNGS-033`.
- Dataset version: `1.3`.
- Kích thước archive đầy đủ: `11,169,497,921` byte.
- Byte range thực tải: `157,231,994` byte.
- Dữ liệu sau giải nén: 750 frame JPG 1920x1080, 25 FPS, 30 giây.
- Ngữ cảnh nhãn: hiệp 2, `29:33-30:03`, action class `Foul`.

Đây là một sequence GSR, không phải video đầy đủ 90 phút. Full-match video của
SoccerNet yêu cầu NDA/password nghiên cứu do nhóm tác giả cấp.

## Cách chạy lại

```powershell
Set-Location "C:\Users\AnDVM\Documents\Codex\2026-08-22\to\outputs\football-attack-thesis"
$env:PYTHONPATH = "src"

python -m attack_analysis soccernet-download data\soccernet-gs `
  --split valid `
  --sequence first

python -m attack_analysis soccernet-analyze `
  data\soccernet-gs\gamestate-2024\valid\SNGS-033
```

## Thống kê tracking

| Chỉ số | Kết quả |
|---|---:|
| Object annotations | 8,677 |
| Unique tracks | 27 |
| Player observations | 6,233 |
| Goalkeeper observations | 520 |
| Referee observations | 1,208 |
| Ball observations | 716 |
| Pitch-coordinate coverage | 99.9% |
| Jersey numbers nhận biết | 9 |

Vận tốc cầu thủ sau quality filtering có mean `1.087 m/s`, p95 `2.968 m/s` và
max `5.212 m/s`. Đây là tốc độ tức thời trên toàn clip, gồm nhiều thời điểm đứng
chờ phát bóng nên mean thấp là hợp lý.

Ball calibration có 46/716 điểm nằm ngoài sân cộng margin 5 m. Raw ball speed
max lên tới `173.452 m/s`; analyzer không coi đây là chuyển động thật. Sau khi
lọc endpoint ngoài sân và 20 segment vượt ngưỡng vật lý 45 m/s, ball speed còn
mean `2.530 m/s`, p95 `8.289 m/s`, max `44.719 m/s`. Report vẫn giữ raw metrics
và số mẫu bị loại để audit, không che mất lỗi dữ liệu.

## Hiệu quả tấn công

**Kết luận provisional: không hiệu quả về tạo cơ hội, độ tin cậy trung bình.**

Grounded evidence:

- đội `right` được suy luận là đội bắt đầu pha bóng và tấn công sang trái;
- bóng đi từ `x=+56.095 m` tới xa nhất `x=-9.797 m`;
- progression cực đại `65.892 m`, cho thấy thoát pressing/đưa bóng lên đáng kể;
- chưa chạm final third phía trái (`x <= -17.5 m`);
- không vào penalty area;
- không có shot, vì vậy sequence xG là `0.0`;
- tình huống kết thúc được SoccerNet gắn nhãn `Foul`.

Heuristic score hiện tại là `30/100`. Pha bóng tốt về progression nhưng không tạo
được cơ hội dứt điểm, nên không đạt mục tiêu chance creation. Kết luận chưa được
nâng lên mức chắc chắn cao vì nhãn `Foul` không cho biết đội nào được hưởng quả
phạt; một quả phạt nguy hiểm cho đội tấn công có thể làm thay đổi đánh giá chiến
thuật dù xG trong clip vẫn bằng 0.

## File kết quả

- `data/soccernet-gs/gamestate-2024/valid/SNGS-033/download-manifest.json`
- `data/soccernet-gs/gamestate-2024/valid/SNGS-033/analysis-report.json`
- `data/soccernet-gs/gamestate-2024/valid/SNGS-033/Labels-GameState.json`
- `data/soccernet-gs/gamestate-2024/valid/SNGS-033/img1/*.jpg`
