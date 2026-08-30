# Kịch bản demo luận văn

Thời lượng mục tiêu: **8–10 phút**. Demo chính dùng một attack FOOTPASS đã cache;
upload video ngoài dataset là phần minh họa `exploratory`, không phải kết quả
định lượng của luận văn.

## Chuẩn bị trước khi trình bày

1. Gắn ổ `D:` và kiểm tra `D:\videos\game_18.mp4`.
2. Kiểm tra `.env` có `OPENAI_API_KEY`; không chiếu nội dung file hoặc key.
3. Từ thư mục project, chạy:

```powershell
python -m unittest discover -s tests -v
python -m streamlit run app/streamlit_app.py --server.port 8504
```

4. Mở <http://localhost:8504>, chờ tab FOOTPASS nạp xong.
5. Chuẩn bị sẵn artifact cache để demo vẫn chạy được nếu GPU hoặc mạng lỗi.

## 0:00–1:00 — Câu hỏi và kiến trúc

Lời nói gợi ý:

> Mục tiêu không chỉ là phát hiện cú sút. Hệ thống chia video thành từng đợt tấn
> công, dựng event và trạng thái không gian, tính xG khi đủ đầu vào, rồi so sánh
> các baseline và LLM để trả lời đợt tấn công hiệu quả đến đâu. Mọi kết luận phải
> dẫn về evidence; nếu chất lượng video không đủ, hệ thống phải từ chối kết luận.

Chỉ vào hai mode:

- **Oracle/assisted:** tactical coordinate và event FOOTPASS, dùng làm upper bound.
- **Predicted/video-only:** model tự suy ra; đây mới là đường end-to-end cần đo độ
  suy giảm và áp dụng quality gate.

## 1:00–2:30 — Chọn một đợt tấn công

Trong tab **FOOTPASS analysis**:

1. Chọn `footpass_oracle`.
2. Chọn match/half `game_18_H1`.
3. Giữ bộ lọc **Chỉ attack (khuyến nghị)** và chọn
   `game_18_H1:sequence-0013`.
4. Cho người xem timeline gồm pass, drive và shot; nhấn mạnh mỗi event có frame,
   thời gian, cầu thủ/role, confidence, source và `evidence_id`.

Lời nói gợi ý:

> Hệ thống phân đoạn possession trước, rồi mới xác nhận chuỗi nào thật sự là
> attack. Effectiveness chỉ được đánh giá khi attack qualification đã PASS.
> Boundary của sequence được giữ cố định để so sánh oracle và predicted.

## 2:30–4:30 — Dynamic calibration và YOLO

Đặt cửa sổ:

- Bắt đầu: `99.68` giây.
- Kết thúc: `106.60` giây.
- Nhấn **Chạy FOOTPASS-assisted calibration + YOLO cho cửa sổ này**.

Kết quả hồi quy mong đợi cho artifact đã xác minh:

- Calibration: `PASS`.
- Frame coverage: khoảng **91,4%**.
- Trong đó: khoảng **78,7% trực tiếp**, **10,9% nội suy giữa hai frame hợp lệ**,
  **1,7% ngoại suy giới hạn**; 15 frame không đủ bằng chứng vẫn bị loại.
- Median holdout error: khoảng **0,20 m**.
- P90 holdout error: khoảng **1,28 m**.

Lời nói gợi ý:

> Coverage không được tăng bằng cách bỏ quality gate. Frame có 7 correspondence
> chỉ được nhận nếu vẫn vượt holdout test; khoảng mất ngắn chỉ được nội suy khi có
> homography hợp lệ ở cả hai đầu. Hệ thống hiển thị riêng phần fit trực tiếp và
> phần temporal propagation để không biến dữ liệu oracle thành một claim video-only.

Phát video overlay và chỉ ra box/track ID, team, pitch coordinate. Nêu rõ YOLO
đang bổ sung quan sát video; event chính ở mode này vẫn đến từ FOOTPASS.

## 4:30–6:00 — Event stream và trained xG

1. Mở phần **Event stream dùng cho phân tích**.
2. Chỉ vào ba gate: assisted calibration, action source và trained xG.
3. Chọn event `shot`; mở inputs gồm tọa độ, distance, angle và pressure/player
   context.
4. Chỉ vào `xg_source` để chứng minh xác suất đến từ model đã huấn luyện, không
   phải con số do LLM tự tạo.

Lời nói gợi ý:

> LLM không tự tính xG. Nó chỉ đọc evidence bundle đã khóa. Nếu shot thiếu tọa độ
> hoặc calibration fail, xG không được đưa vào verdict như bằng chứng chắc chắn.

## 6:00–7:30 — Verdict cho end-user

1. Giữ model OpenAI đã cấu hình.
2. Nhấn **Evaluate selected attack**.
3. Đọc phần trả lời bằng ngôn ngữ tự nhiên trước: verdict ba mức, score 1–5,
   confidence, lý do chính và limitations.
4. Mở từng evidence/source được trích nếu hội đồng muốn kiểm tra.
5. Chỉ mở JSON trong expander **Dữ liệu nghiên cứu/debug** khi được hỏi về schema.

Lời nói gợi ý:

> Câu trả lời này là lớp trình bày. Baseline JSON và source vẫn ở phía dưới để
> audit, nhưng end-user không cần đọc object kỹ thuật. Claim không có evidence ID
> sẽ bị verifier loại trước khi hiển thị.

Nếu OpenAI API lỗi, nói rõ đây là lỗi dịch vụ và trình bày deterministic baseline;
không mô tả baseline đó như kết quả LLM.

## 7:30–8:30 — Chứng minh abstention

Chỉ vào gate **CV-only events**. Với artifact hiện tại, ball tracking chưa đủ ổn
định nên đường chỉ dùng YOLO phải hiện `ABSTAIN`/`insufficient_evidence`.

Lời nói gợi ý:

> Đây là hành vi đúng, không phải demo hỏng. Ball detector có thể thấy bóng ở
> nhiều frame nhưng track liên tục chưa đạt ngưỡng. Vì vậy hệ thống không dùng
> track lỗi để tạo shot hoặc xG giả.

## 8:30–9:30 — Oracle so với predicted

Chuyển sang `footpass_predicted`, giữ cùng match và attack window. So sánh:

- số event và confidence;
- xG/attack features;
- verdict và limitations;
- mức suy giảm khi action/calibration không còn là ground truth.

Không chọn threshold mới trong lúc demo. Threshold phải đến từ validation profile
đã khóa; challenge split không được dùng để tuning.

## 9:30–10:00 — Kết luận và bước tiếp theo

Lời nói gợi ý:

> Demo đã nối video, tracking, dynamic calibration, event stream, trained xG và
> grounded verdict. Phần còn yếu nhất là bóng và calibration video-only. Bước tiếp
> theo là train pitch keypoint/line model trên SoccerNet Calibration; Roboflow chỉ
> dùng auxiliary sau khi audit license và duplicate, rồi báo cáo ablation theo match.

Nguồn dữ liệu và policy chi tiết nằm ở
[`docs/external-calibration-data.md`](docs/external-calibration-data.md).

## Phương án dự phòng

- **GPU bận:** dùng artifact cache, không đổi video/model/window.
- **Video không phát:** dùng file `annotated-browser.mp4` trong run directory;
  đây là bản H.264 dành cho trình duyệt.
- **OpenAI không phản hồi:** hiển thị deterministic baseline và nói rõ LLM chưa
  được đánh giá ở lượt này.
- **Calibration fail ở clip khác:** giữ màn hình `insufficient_evidence` và giải
  thích quality gate; không nới threshold trực tiếp trước hội đồng.
- **Ổ D mất kết nối:** không upload lại footage; dùng artifact cache đã chuẩn bị.

## Checklist trước giờ demo

- [ ] Tests pass.
- [ ] App mở ở port 8504.
- [ ] `game_18_H1:attack-0013` xuất hiện.
- [ ] Window 99.68–106.60 giây đã được cache.
- [ ] `annotated-browser.mp4` phát được.
- [ ] Calibration report có breakdown direct/interpolated/extrapolated.
- [ ] Structured event stream có trained xG provenance.
- [ ] LLM answer hiển thị dạng văn bản; JSON nằm trong expander.
- [ ] CV-only path abstain khi ball tracking gate fail.
- [ ] Không chiếu API key, `.env` hoặc thông tin nhạy cảm.
