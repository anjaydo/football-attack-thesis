# Báo cáo tiến độ — 2026-10-07

Phạm vi: luồng DST H100 của FOOTPASS đã trả JSON val, và mọi chuỗi phân tích kết thúc bằng event `shot` phải có xG từ model train trên StatsBomb.

Tóm tắt:

- DST full pipeline trên H100 đã xong. Chấm lại local bằng `evaluation.py` (τ = 0.15, δ = 12) ra micro-F1 **0.688**, thấp hơn `DST_v1` 0.719, nên **không** lên primary.
- 6 hiệp val decode Attack-v2: 700 possession, 38 attack. 55 chuỗi kết thúc bằng `shot` đều có xG. Không có shot nào thiếu xác suất.
- xG lấy từ `models/xg-v2.joblib` (HistGradientBoosting + Platt), train trên 87 738 cú sút StatsBomb open data, trạng thái `statsbomb_gate_passed_domain_unvalidated`, Brier test 0.0803.
- File JSON H100 từng ghi đè `playbyplay_DST_val.json`. Hash `DST_v1` đã ghim không còn khớp. Bản H100 được tách thành `playbyplay_DST_h100_val.json`.

---

## 1. DST H100

Log server kết thúc `PIPELINE_EXIT:0` lúc 2026-10-06T16:00:55Z. Train DST 15 epoch, rồi ghi 6 hiệp val, 5 776 event. Checkpoint local `FOOTPASS/runs/DST_Baseline_h100/checkpoints/best_model.pt`, SHA-256 `f0acbab02fefc4a0f4c0a747b8f40197b388ba00426a507762e1526a001ddefa`.

Chấm lại trên máy này, cùng `FOOTPASS/evaluation.py`:

| Nguồn | Precision | Recall | Micro-F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `DST_v1` (primary đã đăng ký) | 0.735 | 0.703 | 0.719 | 4267 | 1539 | 1803 |
| `DST_h100` | 0.706 | 0.671 | 0.688 | 4076 | 1700 | 1994 |

Số `DST_h100` trùng log server, kể cả shot: precision 0.564, recall 0.463 (31 TP, 24 FP, 36 FN, 67 GT).

`swap-prediction-source` trả `promoted: false`. `DST_h100` là ablation trong `models/prediction-sources.json`. Không refit `dst-thresholds-v1.json` vì profile đó fit trên DST cũ và nguồn mới không được promote.

Decode Attack-v2 so với oracle decode, biên ±1 giây (`runs/verify_dst_h100_xg.py`):

| Nguồn | Possession | Attack | Possession start / end F1 | Attack start / end F1 |
| --- | ---: | ---: | ---: | ---: |
| Oracle | 755 | 46 | — | — |
| `DST_v1` (lần chấm trước) | 696 | 42 | 0.647 / 0.652 | 0.386 / 0.591 |
| `DST_h100` | 700 | 38 | 0.609 / 0.608 | 0.333 / 0.452 |

DST H100 không cải thiện ranh giới attack. Decoder không bị chỉnh trên các hiệp val này.

### File DST_v1 bị ghi đè

Registry `DST_v1` ghim SHA-256 `34ef3f16233bd2e64cc5336e6f9a6e2c623d724c0517e09b91a99665f03c5741` tại `playbyplay_DST_val.json`. File hiện ở đường dẫn đó có hash `1beba6ae…` và chính là output H100. Bản gốc không còn trong cây thư mục. Đã copy sang `playbyplay_DST_h100_val.json` trước khi đăng ký, để không tiếp tục dùng nhầm tên file.

Chọn `DST_v1` trong UI sẽ báo lệch hash và từ chối chạy. Selectbox predicted mặc định nhảy tới nguồn còn verify được (`DST_h100`) khi primary không còn khớp byte. Muốn dùng lại baseline 0.719 thì cần đặt lại đúng file JSON gốc.

---

## 2. xG bắt buộc khi chuỗi kết thúc bằng shot

Trước đây `footpass-ingest` luôn ghi `xg.status = withheld` và bỏ `--xg-model`. Một chuỗi có event cuối là `shot` có thể trả về mà không có xác suất.

Quy tắc mới, trong `src/attack_analysis/shot_xg.py`:

- Event đóng chuỗi là `shot` hoặc `shot_attempt` thì bản ghi phân tích phải có `xg_probability` trong [0, 1].
- Model là xG-v2. Report phải nhận diện artifact train StatsBomb (`statsbomb_gate_passed_domain_unvalidated` hoặc `validated`).
- Attack-v2 complete dùng `ConfirmedXGRecordV2`. Chuỗi bị censor nhưng vẫn đóng bằng shot vẫn được chấm bằng cùng model, với `confirmed_attack = false`.
- Thiếu model, hoặc model không phải xG-v2 StatsBomb, là lỗi `ShotXGRequiredError`, không còn là field để trống.
- Tọa độ thủ môn / vị trí sút thiếu đi theo missingness mask của xG-v2. Cổng ball F1 không còn xóa xG của một cú sút.

`footpass-ingest` và tab FOOTPASS của Streamlit đều gọi hàm này. Mặc định đọc `models/xg-v2.joblib` (hoặc `XG_V2_MODEL_PATH`).

### Kiểm thử trên DST H100

`runs/verify_dst_h100_xg.py` chạy 6 hiệp predicted, tactical HDF5 `val_tactical_data.h5`, model `xg-v2:sha256:d6a78d180a3d3a05f36eaee6f760e9e3738c138503dfe45197fbb2128c0433ac`.

| | |
| --- | ---: |
| Chuỗi kết thúc bằng shot | 55 |
| Có xG | 55 |
| Attack-v2 complete | 35 |
| Censor nhưng vẫn là shot | 20 |
| xG min / median / max | 0.015 / 0.061 / 0.254 |

`footpass-ingest --prediction-source DST_h100 game_18_H1` ghi `runs/verification-v2/ingest-DST_h100-game_18_H1.json`: `xg.status = scored`, 10/10 shot có xác suất, training source `statsbomb_open_data`, 87 738 dòng.

Model card StatsBomb (không train lại trong đợt này; artifact local đã có):

- 3 461 trận, 9 688 bàn, split nhóm theo trận
- Test Brier 0.0803, ECE₁₀ 0.0040, vượt gate StatsBomb
- Chưa có human-domain validation trên FOOTPASS, nên release xG vẫn `research_only`

Header thuần (`action_type = header`) không thuộc quy tắc bắt buộc này. H1 của game 18 có 1 `headed_attempt` và không nằm trong 10 shot đã chấm.

---

## 3. Test

`$env:PYTHONPATH="src"; py -3 -m unittest discover -s tests -p "test_*.py"` → **338 OK, 1 skip**. Sáu test mới nằm trong `tests/test_shot_xg.py`: shot complete có xG, chuỗi không kết thúc bằng shot không bị ép xG, thiếu model thì lỗi, model không phải StatsBomb thì lỗi, shot bị censor vẫn có xác suất, và tọa độ sút thiếu không làm rơi xG.

---

## 4. Việc còn mở

| # | Việc | Ghi chú |
| --- | --- | --- |
| 1 | Khôi phục byte `DST_v1` | Hash `34ef3f16…5741`. Không suy ra từ file H100. |
| 2 | Gate bóng | Vẫn fail (F1 0.474 / 0.80). xG shot ở trên dùng tọa độ FOOTPASS/HDF5, không dùng ball detector. |
| 3 | Nhãn gold | Chưa có human label để kiểm định xG trên domain FOOTPASS. |
| 4 | Video sync | Vẫn chỉ `game_18` có offset đo được. |
