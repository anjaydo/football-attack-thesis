# Đề cương khởi động luận văn

## Tên đề tài đề xuất

**Đánh giá hiệu quả đợt tấn công trong bóng đá từ video quảng bá bằng tái dựng
trạng thái trận đấu, mô hình giá trị hành động và LLM có ràng buộc bằng chứng.**

Tên tiếng Anh:

**Evaluating Football Attacking Sequences from Broadcast Video using Game State
Reconstruction, Action-Value Models, and Evidence-Grounded LLM Explanations.**

## Câu hỏi nghiên cứu

Câu hỏi chính:

> Từ một đoạn video quảng bá của một đợt tấn công, hệ thống có thể kết luận đợt
> tấn công có hiệu quả hay không, với độ chính xác và khả năng giải thích đến mức
> nào so với đánh giá của chuyên gia?

Các câu hỏi phụ:

1. Các đặc trưng nào từ video đóng góp nhiều nhất: tọa độ, progression, xG, xT,
   khoảng cách đội hình, áp lực hay kết quả cú sút?
2. Lỗi từ khối detection/tracking/calibration ảnh hưởng thế nào đến kết luận cuối?
3. LLM có làm phần giải thích dễ hiểu hơn mà vẫn trung thành với bằng chứng không?
4. Mô hình kết hợp có tốt hơn baseline chỉ dùng kết quả cú sút/xG không?

## Dinh nghia don vi phan tich

Mot **dot tan cong** bat dau khi doi gianh/quyen kiem soat bong va co y dinh dua
bong ve phia khung thanh doi phuong. No ket thuc khi:

- mat quyen kiem soat bong;
- bong ngoai cuoc;
- co cu sut hoac ban thang;
- doi bong tai thiet lap lai trang thai luan chuyen ma khong con tinh tien;
- vuot qua nguong thoi gian quy dinh (de xuat 20 giay cho counter-attack, 40 giay
  cho general attack; nguong nay can duoc kiem chung khi gan nhan).

Nhiem vu duoc mo hinh hoa theo hai cach:

- Classification: `effective` / `ineffective` (co the them `promising` khi gan nhan).
- Regression: diem hieu qua lien tuc tu 0 den 100.

## Dinh nghia hieu qua

Khong dong nhat "hieu qua" voi "ghi ban". Mot dot tan cong co the hieu qua neu
lam tang dang ke xac suat ghi ban hoac tao ra loi the chien thuat ro rang.

Nhan chuyen gia de xuat gom ba muc:

- `effective`: tao ban thang, co hoi chat luong cao, hoac pha vo cau truc phong ngu
  va dua bong vao vung nguy hiem voi quyen kiem soat ro rang;
- `promising`: tinh tien va tao loi the nhung chua tao co hoi du manh;
- `ineffective`: khong tien trien, mat bong som, cu sut chat luong thap hoac khong
  lam tang muc de doa.

Khi bao cao cau tra loi nhi phan, `effective` la Yes; `ineffective` la No;
`promising` duoc xu ly theo threshold da chot truoc va phai co bao cao sensitivity.

## Kien truc he thong

```text
Broadcast video
  -> shot/camera filtering
  -> player + ball detection
  -> multi-object tracking + team assignment
  -> pitch calibration / homography
  -> top-view coordinates and possession events
  -> attack segmentation
  -> xG + xT/VAEP + spatial/tactical features
  -> effectiveness classifier/regressor
  -> structured evidence JSON
  -> LLM explanation constrained by the JSON
```

Nguyen tac quan trong: LLM khong phai bo phan tinh metric. LLM chi nhan JSON da
kiem dinh va dien dat ket luan; moi cau giai thich phai truy nguoc duoc ve feature.

## Du lieu de xuat

### SoccerNet Game State Reconstruction

Dung cho detection, tracking, team/role identification, camera calibration va toa
do top-view. Bo du lieu co 200 sequence moi sequence 30 giay; baseline chinh thuc
co san de tai su dung. Metric CV chinh la GS-HOTA.

### SoccerNet Action Spotting

Dung cho phat hien cac moc nhu shot, goal, foul, corner. Tap du lieu broadcast co
500 tran va 17 lop hanh dong trong ban action-spotting duoc cong bo.

### StatsBomb Open Data / 360

Dung de huan luyen va kiem tra prototype xG, xT/VAEP tren event data truoc khi noi
voi dau ra computer vision. Du lieu JSON gom competition, match, event, lineup va
360 freeze-frame cho mot so tran.

### Tap nhan rieng cho luan van

Can gan nhan khoang 300-800 dot tan cong (tuy nguon luc), moi sequence do it nhat
hai nguoi danh gia. Nhan gom:

- hieu qua 3 muc va diem 1-5;
- loai tan cong: counter, positional, set piece;
- ket qua: goal, shot, box entry, final-third entry, turnover;
- ly do ngan theo rubric co san;
- do chac chan cua nguoi gan nhan.

Phai tach train/validation/test theo **match**, khong tach ngau nhien theo clip,
de tranh ro ri cung doi hinh, tran dau va phong cach camera.

## Dac trung ban dau

- Outcome: goal, shot, shot on target, xG sum, xG cao nhat.
- Progression: quang duong tien ve khung thanh, toc do doc san, so lan vao final third.
- Penetration: box entry, cutback, through ball, khoang cach den goal.
- Pressure/space: so hau ve quanh bong, mat do phong ngu, khoang cach hang thu.
- Possession: so pass, do dai sequence, turnover, tempo.
- Action value: xT gain, VAEP/EPV delta.
- Uncertainty: detection confidence, tracking continuity, calibration reprojection error.

## Baseline va ablation

1. Baseline A: chi dung ket qua cu sut va xG.
2. Baseline B: rule-based score trong prototype hien tai.
3. Model C: Logistic Regression / Gradient Boosting tren feature cau truc.
4. Model D: temporal model tren chuoi event/toa do.
5. Model E: ket hop feature + uncertainty tu CV.
6. LLM ablation: khong LLM, LLM doc feature, LLM/VLM doc video truc tiep.

Ablation toi thieu:

- bo xG;
- bo xT/VAEP;
- bo toa do off-ball;
- bo uncertainty;
- ground-truth coordinates so voi predicted coordinates.

## Danh gia

### Khoi computer vision

- GS-HOTA/HOTA cho game-state reconstruction;
- ball/player detection AP;
- tracking IDF1/HOTA;
- sai so toa do top-view va camera calibration.

### Khoi ket luan hieu qua

- Macro-F1 va balanced accuracy cho classification;
- ROC-AUC/PR-AUC neu dung binary output;
- MAE/Spearman neu du doan diem lien tuc;
- Brier score va calibration curve cho xac suat;
- bootstrap confidence interval theo match.

### Gan nhan va giai thich

- Cohen's kappa (2 annotator) hoac Krippendorff's alpha (nhieu annotator);
- factual consistency: moi claim co duoc metric ho tro hay khong;
- expert usefulness/clarity theo Likert;
- hallucination rate cua LLM.

## Gia thuyet nghien cuu

- H1: feature khong gian va action-value cai thien Macro-F1 so voi xG-only.
- H2: dua uncertainty CV vao model cai thien calibration cua ket qua.
- H3: LLM grounded tren structured evidence tang diem de hieu ma khong lam giam
  factual consistency so voi template explanation.

## Ke hoach 16 tuan

1. Tuan 1-2: literature review, rubric nhan, protocol va data audit.
2. Tuan 3-4: event-data baseline xG/xT/VAEP, attack segmentation.
3. Tuan 5-7: chay SoccerNet GSR baseline, chuan hoa toa do va uncertainty.
4. Tuan 8-9: gan nhan pilot, do agreement, sua rubric.
5. Tuan 10-11: huan luyen effectiveness models va ablation.
6. Tuan 12: noi pipeline video -> features -> prediction.
7. Tuan 13: grounded LLM explanation va hallucination tests.
8. Tuan 14: full evaluation, bootstrap CI, error analysis.
9. Tuan 15-16: viet luan van, reproducibility package va demo.

## Moc hoan thanh dau tien

Moc M1 duoc xem la dat khi:

- 20 dot tan cong mau co schema thong nhat;
- rule-based baseline va xG-only baseline chay duoc;
- rubric duoc hai nguoi thu gan nhan tren 20 clip;
- co bao cao disagreement va ba nhom loi lon nhat;
- mot clip di het duong ong tu toa do/su kien den ket luan va explanation.

## Rui ro va cach giam

- Ball tracking tu broadcast video kho: cho phep giai doan dau dung event/360 data
  lam oracle va thay tung khoi CV sau.
- Khong du compute: tai su dung SoccerNet baseline state/pretrained weights, chi
  fine-tune mot module co dong gop nghien cuu ro.
- Nhan "hieu qua" chu quan: rubric cu the, nhieu annotator, bao cao agreement va
  khong ep consensus cho cac case mo ho.
- LLM hallucination: schema, claim-level verifier va template baseline.
- Data leakage: split theo match/competition va khoa test set truoc khi tuning.

## Tai lieu nen tang

- SoccerNet Game State Reconstruction, CVPR Workshops 2024:
  https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Somers_SoccerNet_Game_State_Reconstruction_End-to-End_Athlete_Tracking_and_Identification_on_CVPRW_2024_paper.html
- SoccerNet tasks and Action Spotting: https://www.soccer-net.org/tasks
- Official GSR baseline: https://github.com/SoccerNet/sn-gamestate
- StatsBomb Open Data: https://github.com/statsbomb/open-data
- VAEP / socceraction: https://github.com/ML-KULeuven/socceraction
- xT versus VAEP comparison:
  https://eos.cs.kuleuven.be/sites/eos.cs.kuleuven.be/files/xt-vs-vaep-aits.pdf
