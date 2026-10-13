Họ tên: Cao Đức Hiếu (Mã học viên: 2A202602701)
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 0.00 | 0.0 | 0.124 | [0.096, 0.153] | 1.31 | 0.106 | 0.145 | 0.099 |
| E1 | LoRA LLM | 33,030,144 | 10.42 | 21.6 | 0.578 | [0.539, 0.617] | 0.65 | 0.543 | 0.614 | 0.535 |
| E2 | + projector | 142,354,432 | 11.85 | 24.8 | 0.605 | [0.566, 0.644] | 0.61 | 0.573 | 0.641 | 0.556 |
| E3 | + LoRA ViT | 149,858,304 | 13.40 | 29.5 | 0.618 | [0.579, 0.657] | 0.59 | 0.589 | 0.654 | 0.567 |
| E4 | rank (r=4 vs r=64) | 8,257,536 / 132,120,576 | 9.85 / 12.10 | 19.2 / 25.4 | 0.537 / 0.582 | [0.496, 0.578] / [0.542, 0.622] | 0.68 / 0.64 | 0.504 / 0.550 | 0.568 / 0.617 | 0.498 / 0.540 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Qwen3-VL-4B có tổng quy mô khoảng 4,44B tham số, phân bổ như sau:
- **Vision Transformer (ViT):** ~415M tham số (24 visual blocks xử lý không gian ảnh và nén frame 640×360 thành đúng 220 visual tokens).
- **Projector:** ~109M tham số, gồm `visual.merger` (~27M) và `visual.deepstack_merger_list` (~82M tham số nối đa tầng tại các block 5, 11, 17 của ViT vào LLM).
- **Large Language Model (LLM):** ~4,0B tham số (36 layers Transformer Decoder).

Phân bố cập nhật qua các thí nghiệm:
- **E1:** Chỉ cập nhật LoRA trên LLM (36 layers × 7 modules self_attn và mlp = 252 modules, rank r=16), tương đương **33.030.144** tham số huấn luyện. ViT và Projector đóng băng 100%.
- **E2:** Cập nhật LoRA LLM (33.03M) cộng thêm mở huấn luyện đầy đủ (full training) toàn bộ Projector ở kiểu dữ liệu float32 (~109M tham số), đạt tổng cộng **142.354.432** tham số huấn luyện. ViT vẫn đóng băng.
- **E3:** Giữ nguyên LoRA LLM và Projector của E2, đồng thời mở thêm LoRA trên ViT (24 blocks × 4 modules = 96 modules, thêm ~7,5M tham số). Tổng số module LoRA đạt 348 modules và tổng số tham số huấn luyện đạt **149.858.304** tham số.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- Tỷ lệ phân tích cú pháp hợp lệ của mô hình gốc (E0): `parse_valid_rate` = **0.624** (62,4%). Có tới 37,6% câu trả lời không thể chuyển đổi thành cấu trúc danh sách JSON đúng chuẩn.
- Hai lỗi phổ biến nhất của mô hình gốc zero-shot:
  1. *Lỗi định dạng văn bản tự do:* Model không tuân thủ yêu cầu "chỉ trả về JSON", mà tự thêm lời chào, giải thích dông dài ở đầu/cuối hoặc bọc markdown ` ```json ` kèm văn bản nhận xét. Đôi khi mô hình rơi vào vòng lặp lặp từ khiến câu trả lời bị cắt cụt (truncated) khi hết `max_new_tokens=256`.
  2. *Lỗi sai thang toạ độ và hallucination (bịa đặt đối tượng):* Mô hình gốc chưa hiểu quy ước grounding 0–1000 của bài toán, thường đoán mò toạ độ theo pixel tuyệt đối hoặc số thực [0, 1]. Đồng thời mô hình bịa thêm box ở các vùng bóng râm mặt đường, dẫn đến FP/ảnh rất cao (1.31) và F1 chỉ đạt 0.124.
- Ví dụ output thô từ mô hình E0:
  ```text
  Based on the image provided, here are the detected pavement damages:
  ```json
  [{"label": "crack", "bbox_2d": [180, 120, 210, 145]}, {"label": "pothole", "bbox_2d": [0.45, 0.62, 0.58, 0.75]}]
  ```
  Note that there is another possible manhole cover near the road curb on the left.
  ```

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- Toạ độ Grounding của Qwen3-VL được chuẩn hoá trên thang nguyên $[0, 1000]$ (tương đối theo W và H của ảnh). Ảnh gốc dashcam có kích thước $640 \times 360$.
- Nếu model dự đoán đúng theo thang 0–1000 nhưng ta lại dùng script chấm so với toạ độ pixel tuyệt đối (hoặc ngược lại):
  Một box ở góc phải ảnh có toạ độ pixel là $[500, 280, 620, 340]$. Trên thang 0–1000, nó tương ứng với $[781, 778, 969, 944]$. Nếu so khớp trực tiếp hai box này, khoảng cách toạ độ bị lệch hoàn toàn khiến phần diện tích giao nhau bằng 0, tức $IoU = 0.0 < 0.5$.
  Khi đó, tất cả dự đoán đều biến thành False Positive ($FP$) và tất cả nhãn thật đều thành False Negative ($FN$). Kết quả là $Precision = 0.0$, $Recall = 0.0$ và **$F1 = 0.0$** dù cú pháp JSON hợp lệ 100%.
- Cách kiểm chứng: Tôi đã kiểm chứng điều này thông qua unit test `test_pixel_vs_norm1000_scale_trap` trong file `tests/test_score.py`. Khi chấm cùng một box dự đoán với cờ `--coord pixel`, kết quả đạt F1 = 1.0 (hoàn hảo); nhưng khi đổi sang cờ `--coord norm1000` mà không đổi thang nhãn, `parse_valid_rate` vẫn là 1.0 nhưng F1 lập tức sụt về 0.0 chính xác.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- Ta chỉ tính loss trên các token của câu trả lời (assistant response, đặt nhãn bằng `input_ids`), còn toàn bộ prompt và 220 visual tokens được mask bằng `-100` vì:
  1. Bài toán là mô hình sinh có điều kiện (conditional generation): mục tiêu của mạng là học cách dự đoán nhãn và toạ độ khi **đã biết trước** ảnh và câu hỏi.
  2. Prompt và 220 visual tokens chiếm hơn 80% tổng chiều dài chuỗi đầu vào. Nếu không mask loss (-100), hàm mất mát Cross-Entropy sẽ dành phần lớn năng lực tối ưu hoá để tái tạo (reconstruct) lại hình ảnh và câu lệnh người dùng thay vì tập trung vào việc định vị bounding box.
  3. Hậu quả nếu tính loss trên cả prompt và ảnh: Gradient bị loãng nghiêm trọng, mô hình hội tụ rất chậm, việc học grounding nhãn/toạ độ bị suy thoái và mô hình dễ bị hallucination hoặc suy giảm chất lượng sinh chuỗi JSON.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- Không thể train full projector nếu ViT bị quantize 4-bit vì: Trong kiến trúc Qwen3-VL, các tầng của projector (`visual.merger` và `visual.deepstack_merger_list`) nằm trực tiếp bên trong module `visual`. Nếu toàn bộ ViT bị lượng tử hoá 4-bit NF4, các trọng số này bị nén thành kiểu số nguyên lượng tử hoá rời rạc. Các trọng số lượng tử này không thể nhận gradient và bộ tối ưu hoá không thể cập nhật liên tục (`requires_grad = True` không có tác dụng).
- Cách xử lý: Sử dụng tham số `llm_int8_skip_modules=["visual", "lm_head"]` trong `BitsAndBytesConfig`. Tham số này giữ nguyên toàn bộ nhánh ViT và Projector ở định dạng FP16 gốc (không bị lượng tử hoá), nhờ đó ta có thể mở và huấn luyện đầy đủ Projector ở E2 và E3.
- Projector phải ở float32 vì: Projector là điểm nghẽn kiến trúc (bottleneck) kết nối và chuyển đổi số chiều biểu diễn từ ViT sang LLM. Dòng gradient tích luỹ từ 36 tầng LLM chảy ngược về qua Projector có biên độ biến thiên rất mạnh. Nếu giữ ở fp16, gradient rất dễ gặp hiện tượng tràn số (overflow) hoặc tắt dần (underflow) dẫn đến loss bị `NaN`/`Inf`. Đưa tham số Projector về float32 đảm bảo độ chính xác số học và ổn định huấn luyện.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- Dữ liệu thu thập từ camera hành trình (dashcam) quay video liên tục, sau đó cắt frame. Hai frame liền kề chỉ cách nhau vài phần mười giây nên góc quay, cảnh vật ven đường, thời tiết và vật thể hư hại gần như trùng lặp nhau.
- Nếu chia ngẫu nhiên (random split), các frame gần như giống hệt nhau sẽ rơi đồng thời vào cả tập train và tập test $\to$ gây **rò rỉ dữ liệu (data leakage)** nghiêm trọng. Mô hình chỉ việc "học vẹt" bối cảnh khung hình quen thuộc thay vì học nhận diện đặc trưng hư hại tổng quát, dẫn đến điểm F1 trên tập test cao giả tạo nhưng hoàn toàn vô dụng trên thực tế.
- Chia theo `session` (phiên quay dựa trên ngày, giờ và tiền tố file) gom toàn bộ các frame của một phiên vào duy nhất một split, đảm bảo mô hình được kiểm thử trên những cung đường hoàn toàn mới.
- Thống kê chi tiết từ `data/splits/meta.json` (dataset version `d10-21e4070301`):
  - **Train:** 1.422 ảnh | 1.074 pothole, 1.750 crack, 691 manhole (tổng 3.515 boxes).
  - **Val:** 276 ảnh | 66 pothole, 470 crack, 103 manhole (tổng 639 boxes).
  - **Test:** 311 ảnh | 121 pothole, 299 crack, 163 manhole (tổng 583 boxes).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- Bảng so sánh 3 cấu hình:
  - **E1 (LoRA LLM):** F1 = **0.578**, CI95 = **[0.539, 0.617]**, VRAM = **10.42 GB**, Thời gian = **21.6 phút**.
  - **E2 (+ Projector):** F1 = **0.605**, CI95 = **[0.566, 0.644]**, VRAM = **11.85 GB**, Thời gian = **24.8 phút**.
  - **E3 (+ LoRA ViT):** F1 = **0.618**, CI95 = **[0.579, 0.657]**, VRAM = **13.40 GB**, Thời gian = **29.5 phút**.
- Đánh giá:
  - Việc mở Projector (E2) giúp F1 tăng +0.027 (+2,7%) so với E1, và thêm LoRA ViT (E3) giúp tăng thêm +0.013 (+1,3%).
  - Tuy nhiên, khi nhìn vào khoảng tin cậy 95% (CI95), khoảng của E2 [0.566, 0.644] và E3 [0.579, 0.657] giao thoa mạnh với E1 [0.539, 0.617]. Về mặt kiểm định thống kê, mức chênh lệch này chưa vượt qua khoảng tin cậy để khẳng định vượt trội hoàn toàn.
  - Xét trên cán cân chi phí tài nguyên: E1 là cấu hình tiết kiệm tài nguyên nhất. E2 là điểm cân bằng hợp lý nhất (tăng nhẹ VRAM và 3 phút huấn luyện để cải thiện khả năng thích ứng của projector). E3 tốn thêm 3 GB VRAM và gần 10 phút tính toán nhưng mức tăng F1 cận biên rất nhỏ, không thực sự đáng đồng tiền bát gạo trên tập dữ liệu nhỏ này.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- Với tập test nhỏ gồm 311 ảnh (chứa 583 box), khoảng tin cậy bootstrap 95% có độ rộng xấp xỉ $\Delta CI95 \approx 0.078$ (khoảng $\pm 0.039$ quanh F1 trung bình).
- Giới hạn kết luận: Khoảng tin cậy rộng làm mờ ranh giới hiệu quả giữa các kiến trúc. Mức chênh lệch F1 giữa E1 (0.578) và E2 (0.605) là 0.027, hoàn toàn nằm gọn trong độ rộng khoảng tin cậy của cả hai bên. Do đó, ta không thể kết luận một cách chắc chắn 100% về mặt ý nghĩa thống kê rằng E2 hay E3 luôn vượt trội hơn E1 trên mọi kịch bản thực tế. Mẫu test nhỏ đòi hỏi chúng ta phải thận trọng khi đánh giá, không thể chỉ nhìn vào một con số F1 điểm (point estimate) mà phải luôn đọc kèm CI95.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- Xếp hạng F1 theo lớp: `crack` (F1 = 0.614 ở E1, 0.654 ở E3) > `manhole` (F1 = 0.535 ở E1, 0.567 ở E3) $\approx$ `pothole` (F1 = 0.543 ở E1, 0.589 ở E3).
- Nguyên nhân:
  - **`crack` đạt kết quả tốt nhất:** Chiếm số lượng lớn nhất trong dataset (1.750 box ở train, 299 box ở test), vết nứt kéo dài tạo tương phản hình thái rõ nét trên bề mặt đường nên mô hình học rất tốt.
  - **`pothole` kém nhất về khả năng định vị:** Ổ gà có diện tích trung vị rất nhỏ (chỉ chiếm ~0.7% diện tích ảnh dashcam). Khi nằm ở khoảng cách xa, ổ gà bị nhạt nhoà và thường bị bỏ sót (False Negative, xem ảnh minh hoạ `report/vis_err_2.png`). Ngoài ra, mô hình dễ nhầm lẫn giữa bóng râm của cây/vệt ướt với ổ gà (xem `report/vis_err_1.png`).
  - **`manhole`:** Có hình dạng tròn/vuông đặc trưng nhưng số lượng mẫu ít nhất (chỉ 163 box ở test). Một số nắp cống chìm cùng màu bê tông hoặc bị bùn đất phủ làm mô hình nhận diện sai lớp thành pothole (xem `report/vis_err_3.png`).
- Các ảnh minh hoạ đã lưu trong thư mục `report/`:
  - True Positives: `report/vis_tp_1.png` (pothole chuẩn), `report/vis_tp_2.png` (crack chuẩn), `report/vis_tp_3.png` (manhole chuẩn).
  - Lỗi sai: `report/vis_err_1.png` (FP do bóng râm), `report/vis_err_2.png` (FN bỏ sót ổ gà nhỏ ở xa), `report/vis_err_3.png` (Lẫn lộn nhãn crack vs pothole).

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- Biến được lựa chọn trong E4: **`rank`** của LoRA ($r=4$ vs $r=64$, cấu hình $\alpha = 2r$) so sánh với cấu hình gốc $r=16$ trên nền tảng E1:
  - **$r=4$ ($\alpha=8$):** 8.257.536 tham số, VRAM = 9.85 GB, Thời gian = 19.2 phút. F1 = **0.537**, CI95 = **[0.496, 0.578]**.
  - **$r=16$ ($\alpha=32$, gốc):** 33.030.144 tham số, VRAM = 10.42 GB, Thời gian = 21.6 phút. F1 = **0.578**, CI95 = **[0.539, 0.617]**.
  - **$r=64$ ($\alpha=128$):** 132.120.576 tham số, VRAM = 12.10 GB, Thời gian = 25.4 phút. F1 = **0.582**, CI95 = **[0.542, 0.622]**.
- Kết quả và phân tích:
  - Khi giảm rank xuống $r=4$, dung lượng học của ma trận thích ứng bị co hẹp, làm F1 giảm rõ rệt từ 0.578 xuống 0.537 (-0.041), khoảng CI95 lệch xuống dưới so với $r=16$.
  - Khi tăng rank lên $r=64$, F1 chỉ tăng nhẹ +0.004 lên 0.582. Khoảng CI95 [0.542, 0.622] gần như bao trùm hoàn toàn khoảng CI95 của $r=16$ [0.539, 0.617], trong khi số tham số tăng gấp 4 lần và tốn thêm gần 2 GB VRAM.
- Kết luận: Việc tăng rank lên 64 không mang lại cải thiện có ý nghĩa thống kê trên tập dữ liệu này. Cấu hình $r=16$ là điểm tối ưu cân bằng giữa hiệu năng và tài nguyên. Kết luận này đáng tin cậy vì được kiểm chứng đồng thời qua cả F1, CI95 và chi phí tính toán.

## Nhìn lại

Qua bài lab, tôi đã hiểu sâu sắc cách hoạt động thực tế của một mô hình Vision-Language Model trong bài toán grounding:
1. *Tầm quan trọng của M-RoPE và phân chia session:* Việc rò rỉ dữ liệu giữa các frame video liên tiếp có thể đánh lừa hoàn toàn các chỉ số đánh giá nếu không gom nhóm theo phiên quay.
2. *Độ nhạy số học của Projector:* Hiểu rõ vì sao Projector phải giữ ở float32 và vì sao thư viện PEFT mặc định không lưu trọng số của các module mở ngoài adapter LoRA (bẫy quên lưu `projector.pt`).
3. *Vấn đề bounding box siêu nhỏ:* Với các đối tượng diện tích < 1% khung hình như ổ gà ở xa, cơ chế nén token thị giác của VLM vẫn gặp giới hạn phân giải không gian nhất định so với các mô hình chuyên biệt như YOLOv8/v11. Đây là bài toán mở thú vị cần kết hợp thêm kỹ thuật cắt lát ảnh độ phân giải cao (high-res cropping/sliding window).
