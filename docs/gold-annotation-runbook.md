# Runbook gán nhãn Attack-v2 (P11)

P11 **không đóng bằng code**. CLI/UI chỉ theo dõi tiến độ. Nhãn nằm ngoài git (`data/` bị gitignore).

Thứ tự bắt buộc:

1. **Development** — `game_18`, `game_24`, `game_47` (intra-rater trên split này).
2. **Train** — `game_6` (pilot), `8`, `16`, `21`, `33`, `48`.
3. **Frozen test** — `game_0`, `game_36`, `game_50` chỉ sau khi khóa threshold trên development. Không mở test để tune.

## 1. Kiểm tra tiến độ (không đọc frozen test)

```powershell
attack-analysis gold-status
```

`p11_status` luôn `open` cho đến khi có nhãn đủ + human-domain validation xG.
`next_step` bắt đầu là `annotate_development` khi chưa có `data/attack-v2-development.jsonl`.

`attack-analysis doctor` cũng in khối `gold` và `operations`.

## 2. Manifest 12 trận

Video Full HD (hoặc processing) phải khớp SHA-256 trong manifest:

```powershell
attack-analysis build-gold-manifest "G:\My Drive\footpass-videos" data\attack-v2-gold-manifest.json
```

Mở tab **Annotation** trên Streamlit, upload file này. Danh sách trận được sắp development → train → test. Game test hiện cảnh báo đỏ.

## 3. Gán nhãn development

- Mỗi frame thuộc đúng một nhãn: `possession`, `attack`, `uncertain`, `dead_ball`.
- Khai báo period inclusive (`development-periods.json`) rồi:

```powershell
attack-analysis validate-gold-annotations data\attack-v2-development.jsonl `
  --period-intervals data\labels\development-periods.json
```

Round 2 (washout ≥ 14 ngày, ~20% thời lượng):

```powershell
attack-analysis gold-intra-rater round-1.jsonl round-2.jsonl
```

## 4. Train, rồi seal test

Train ghi `data/attack-v2-train.jsonl`. Test **chỉ** qua sealed writer:

```powershell
attack-analysis seal-gold-test data\sealed-test `
  --period-intervals data\labels\frozen-test-periods.json
attack-analysis verify-gold-test-seal data\sealed-test
```

`seal-gold-test` yêu cầu đủ cả hai hiệp của đúng `game_0` / `36` / `50`. Đọc nhãn test chỉ dành cho final evaluator (`--allow-frozen-test`).

## 5. Sau khi có nhãn

Vẫn còn P11 cho đến khi xG-v2 được human-domain validate trên FOOTPASS (không dùng StatsBomb làm claim domain). Effectiveness categorical vẫn khóa.
