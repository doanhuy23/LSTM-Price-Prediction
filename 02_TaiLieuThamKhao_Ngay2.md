# 02. TÀI LIỆU THAM KHẢO – NGÀY 2 (07/12/2025)

### 1. Hochreiter & Schmidhuber (1997) – Long Short-Term Memory
- Link: https://www.bioinf.jku.at/publications/older/2604.pdf
- Đóng góp chính: Giới thiệu cell state + 3 cổng (forget, input, output)
- Forget gate: quyết định bỏ bao nhiêu thông tin từ Ct−1
- Input gate + candidate: quyết định thêm thông tin mới
- Output gate: quyết định hidden state ht hiện tại
→ Kết luận: Đây là nền tảng bắt buộc phải trích dẫn trong phần 2.2 của báo cáo đồ án.

### 2. Gers et al. (2000) – Learning to Forget: Continual Prediction with LSTM
- Thêm Peephole connections (cell state nhìn trực tiếp vào các cổng)
- Cải thiện một chút độ chính xác, nhưng tăng tham số
→ Quyết định đồ án: KHÔNG dùng peephole (PyTorch/Keras mặc định không có, giữ đơn giản)

### 3. Tổng quan các nghiên cứu 2020–2024 (paper 2024)
- Link: https://www.sciencedirect.com/science/article/pii/S2405844024023785
- Kết luận từ bảng 2:
  + Tốt nhất hiện nay với Bitcoin: LSTM 2–3 lớp + window 60–90 → RMSE ~300–600 USD
  + Nhiều paper kết hợp thêm Volume, RSI, sentiment Twitter → cải thiện 5–15%
  + GRU thường nhanh hơn LSTM nhưng độ chính xác tương đương hoặc thấp hơn chút

### QUYẾT ĐỊNH CHÍNH THỨC CỦA ĐỒ ÁN (hôm nay xác định luôn để không thay đổi nữa)
- Tài sản dự đoán: **Bitcoin (BTC-USD)**
- Lý do:
  1. Dữ liệu dễ lấy, liên tục 24/7
  2. Biến động cực mạnh → dễ thấy hiệu quả của LSTM so với ARIMA
  3. Có rất nhiều paper để so sánh kết quả
  4. Được hội đồng đánh giá cao vì tính thời sự
- Thời gian dữ liệu: 01/01/2017 → 01/12/2025 (gần 9 năm)
- Dự đoán ngắn hạn: 1–7 ngày tới

### Nhiệm vụ còn lại hôm nay
- [x] Đọc 3 paper trên
- [ ] Tạo file 02_TaiLieuThamKhao_Ngay2.md
- [ ] Commit + push lên GitHub