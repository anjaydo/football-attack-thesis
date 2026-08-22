# Football Attack Effectiveness Thesis

Prototype khởi đầu cho luận văn: từ video bóng đá, tái dựng trạng thái trận đấu,
tính các đặc trưng định lượng và trả lời một đợt tấn công có hiệu quả hay không.

## Phạm vi MVP

MVP hiện tại bắt đầu từ dữ liệu sự kiện có cấu trúc (tọa độ, đường chuyền, cú sút,
xG/xT nếu có). Đây là điểm nối giữa khối computer vision và khối đánh giá:

```text
Video -> detection/tracking/calibration -> events + pitch coordinates
      -> metrics -> effectiveness model -> grounded explanation
```

Prototype đã có:

- schema cho một đợt tấn công;
- schema quan sát cầu thủ/bóng theo từng frame;
- đọc metadata video qua OpenCV tùy chọn;
- tự động tách event stream thành các đợt tấn công;
- các chỉ số progression, final-third/box entry, shot, xG và xT;
- rule-based baseline cho điểm 0-100 và kết luận;
- explanation deterministic chỉ dựa trên dữ liệu đã tính;
- dữ liệu ví dụ và unit tests.

Rule-based score chỉ là baseline kỹ thuật. Trong luận văn, nó sẽ được thay/thêm
bằng mô hình học có giám sát từ nhãn của chuyên gia bóng đá.

## Chạy nhanh

Yêu cầu Python 3.10+.

```powershell
python -m pip install -e .
attack-analysis examples/attack_effective.json
attack-analysis examples/attack_ineffective.json
attack-analysis analyze-events examples/match_events.json
python -m unittest discover -s tests -v
```

Để đọc metadata video thật:

```powershell
python -m pip install -e ".[video]"
attack-analysis video-info path/to/video.mp4
```

Lệnh cũ `attack-analysis file.json` vẫn được hỗ trợ và tương đương với
`attack-analysis evaluate file.json`.

Tài liệu:

- `docs/thesis-kickoff.md`: định hướng nghiên cứu và kế hoạch thực nghiệm;
- `docs/data-contracts.md`: hợp đồng dữ liệu giữa video, tracking, event và scoring.
