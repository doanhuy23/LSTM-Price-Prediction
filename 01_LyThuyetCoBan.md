# 01. TÓM TẮT LÝ THUYẾT CƠ BẢN
## Dự đoán chuỗi thời gian tài sản tài chính bằng mạng LSTM

### 1. Chuỗi thời gian (Time Series) và đặc điểm của dữ liệu tài chính
Chuỗi thời gian là tập hợp các quan sát được ghi nhận theo thứ tự thời gian.  
Dữ liệu giá tài sản tài chính (cổ phiếu, crypto, vàng, VN-Index…) có các đặc trưng nổi bật:
- Non-stationary: thống kê (mean, variance) thay đổi theo thời gian
- Xu hướng dài hạn (trend), tính mùa vụ yếu hoặc không có
- Biến động mạnh (high volatility), hiện tượng “volatility clustering”
- Tính tự tương quan (autocorrelation) cao
- Nhiễu rất lớn, chịu ảnh hưởng bởi tin tức, tâm lý đám đông

### 2. Các phương pháp truyền thống và hạn chế
| Phương pháp         | Ưu điểm                          | Hạn chế nghiêm trọng với tài sản tài chính          |
|---------------------|----------------------------------|-----------------------------------------------------|
| ARIMA / SARIMA      | Dễ hiểu, có cơ sở thống kê       | Yêu cầu dữ liệu stationary → cần differencing nhiều lần<br>Không bắt được phi tuyến tính mạnh |
| Exponential Smoothing | Đơn giản, nhanh                  | Chỉ phù hợp chuỗi có trend + mùa vụ rõ ràng         |
| Linear Regression   | Dễ triển khai                    | Không xử lý được tự tương quan và phi tuyến tính    |

→ Các mô hình truyền thống thường cho kết quả rất kém trên dữ liệu crypto và cổ phiếu.

### 3. Mạng nơ-ron hồi quy (RNN) và vấn đề vanishing/exploding gradient
RNN có khả năng “nhớ” thông tin quá khứ nhờ hidden state:
ht = tanh(Wxh·xt + Whh·ht−1)

Tuy nhiên khi lan truyền ngược qua nhiều bước thời gian:
- Gradient bị nhân liên tục với ma trận trọng số < 1 → vanishing gradient → không học được phụ thuộc dài hạn
- Hoặc > 1 → exploding gradient

→ Với chuỗi giá tài chính dài hàng trăm ngày, RNN cơ bản gần như vô dụng.

### 4. LSTM – Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)
LSTM giải quyết triệt để vấn đề trên bằng cơ chế “cổng” (gates) và cell state.

#### 4.1. Kiến trúc một LSTM cell
- Cell state Ct: đường cao tốc mang thông tin qua toàn bộ chuỗi
- 3 cổng điều khiển bằng sigmoid (0–1) và tanh:
  1. Forget Gate ft ∈ (0,1): quyết định bỏ thông tin cũ nào  
     ft = σ(Wf·[ht−1, xt] + bf)
  2. Input Gate it + Candidate C̃t: quyết định thêm thông tin mới nào  
     it = σ(Wi·[ht−1, xt] + bi)  
     C̃t = tanh(WC·[ht−1, xt] + bC)
  3. Output Gate ot: quyết định output hiện tại  
     ot = σ(Wo·[ht−1, xt] + bo)  
     ht = ot ⊙ tanh(Ct)

#### 4.2. Công thức cập nhật cell state (cốt lõi của LSTM)
Ct = ft ⊙ Ct−1 + it ⊙ C̃t   → cộng dồn có chọn lọc

Nhờ phép cộng (additive) thay vì nhân, gradient có đường đi thẳng qua cell state → không bị vanishing.

### 5. Các biến thể phổ biến hiện nay
| Biến thể          | Đặc điểm chính                          | Ưu điểm nổi bật                  |
|-------------------|-----------------------------------------|----------------------------------|
| GRU (2014)        | Gộp Forget + Input gate → chỉ 2 gates   | Ít tham số hơn, train nhanh hơn  |
| Stacked LSTM      | Nhiều lớp LSTM chồng lên nhau           | Bắt được biểu diễn mức cao hơn   |
| Bidirectional LSTM | Chạy cả tiến và lùi                    | Tốt khi có toàn bộ chuỗi (không dùng cho forecasting thực sự) |
| LSTM + Attention  | Thêm cơ chế chú ý                       | Rất mạnh hiện nay                |
| Transformer (2017) | Loại bỏ hoàn toàn recurrence           | State-of-the-art cho chuỗi dài   |

### 6. Tại sao LSTM cực kỳ phù hợp với dự đoán giá tài sản tài chính?
1. Có khả năng học phụ thuộc dài hạn (60–200 ngày trước vẫn ảnh hưởng)
2. Xử lý tốt dữ liệu phi tuyến tính, biến động mạnh
3. Linh hoạt thêm nhiều feature (Volume, RSI, sentiment…)
4. Đã được chứng minh hiệu quả trên rất nhiều bài báo và cuộc thi Kaggle

### 7. Tài liệu tham khảo chính (đã đọc trong Ngày 1)
1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780.
2. Colah’s Blog (2015). Understanding LSTM Networks. https://colah.github.io/posts/2015-08-Understanding-LSTMs/
3. Brownlee, Jason. Deep Learning for Time Series Forecasting (2018)
4. Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM

---  
Ngày hoàn thành: 06/12/2025  
Người thực hiện: [Tên bạn]