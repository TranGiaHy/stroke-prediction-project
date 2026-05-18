# BÁO CÁO PHÂN TÍCH CHUYÊN SÂU VÀ DỰ ĐOÁN NGUY CƠ ĐỘT QUỴ
**Dự án:** Nghiên cứu Phân tích Hệ số Sức khỏe & Tối ưu hóa Mô hình Dự đoán Nguy cơ Đột quỵ  
**Tác giả:** Trần Gia Hy  
**Chuyên ngành:** Khoa học Dữ liệu
**Chương trình:** AI Engineer Foundation Program - Quanskill  

---

## 1. Tóm Tắt Dự Án (Executive Summary)
Dự án này tập trung xây dựng một pipeline (hệ thống) hoàn chỉnh để tiền xử lý, phân tích khám phá (EDA) và chuẩn bị dữ liệu y tế cho bài toán dự đoán nguy cơ đột quỵ. Trọng tâm của dự án là áp dụng các kỹ thuật xử lý dữ liệu chuẩn xác và nền tảng toán học tối ưu nhằm trích xuất các đặc trưng lâm sàng quan trọng, tạo tiền đề vững chắc cho các mô hình Machine Learning.

Báo cáo này tổng hợp toàn bộ quy trình nghiên cứu thực nghiệm xuyên suốt 3 Module cốt lõi:
1. **NumPy Advanced Manipulations:** Tiền xử lý ma trận và thống kê mô tả.
2. **Mathematics for AI:** Ứng dụng Xác suất và tối ưu hóa hàm mất mát (MSE) bằng Gradient Descent từ nền tảng (scratch).
3. **Pandas & Exploratory Data Analysis (EDA):** Làm sạch, kỹ nghệ đặc trưng (Feature Engineering) và xây dựng Hệ thống Dashboard trực quan hóa lỗi biên tập.

---

## 2. Module 1: Xử Lý Biến Số Và Thống Kê Mô Tả Với NumPy

### 2.1. Làm Sạch Và Điền Khuyết Dữ Liệu
Tập dữ liệu y tế ban đầu chứa các giá trị thiếu (Missing Values) tại cột thuộc tính `bmi`. Phương pháp xử lý được áp dụng là **Imputation bằng Giá trị Trung bình (Mean Imputation)** theo từng phân phối cột dữ liệu số. 
- Ma trận số sau đó được chuyển đổi hoàn toàn sang mảng NumPy tối ưu hóa hiệu năng toán học và tính toán vectơ (Vectorized Computation).

### 2.2. Hồ Sơ Thống Kê Định Lượng (Statistical Profiling)
Các chỉ số thống kê đặc trưng của các thuộc tính định lượng (`age`, `avg_glucose_level`, `bmi`) được tính toán thủ công mà không sử dụng các hàm cấp cao có sẵn nhằm kiểm soát phân phối:
- **Hàm tính Mean ($\mu$):** $\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$
- **Hàm tính Độ lệch chuẩn ($\sigma$):** $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$

### 2.3. Chuẩn Hóa Đặc Trưng (Feature Scaling)
Để loại bỏ sự chênh lệch về đơn vị đo lường (Độ tuổi từ 0-82, Đường huyết từ 50-270, BMI từ 10-90), kỹ thuật **Min-Max Scaling** được triển khai theo vectơ hóa:
$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
Kết quả đưa toàn bộ phân phối thuộc tính về dải $[0, 1]$, giúp thuật toán tối ưu hóa Gradient Descent hội tụ nhanh hơn và tránh hiện tượng bùng nổ đạo hàm.

---

## 3. Module 2: Nền Tảng Toán Học Cho AI & Tối Ưu Hóa Gradient Descent

### 3.1. Ước Lượng Rủi Ro Bằng Xác Suất Điều Kiện
Phân tích xác suất Bayes được áp dụng để chứng minh mối tương quan lâm sàng giữa bệnh lý Cao huyết áp (`hypertension`) và biến cố Đột quỵ (`stroke`).
- **Xác suất tiên nghiệm (Prior Probability):** Xác suất một người bất kỳ trong tập dữ liệu bị đột quỵ:
  $$P(\text{stroke} = 1) \approx 4.87\%$$
- **Xác suất hậu nghiệm / điều kiện (Conditional Probability):** Xác suất một người bị đột quỵ khi đã có bệnh lý nền cao huyết áp:
  $$P(\text{stroke} = 1 \mid \text{hypertension} = 1) \approx 13.25\%$$

*Nhận xét:* $P(\text{stroke}=1 \mid \text{hypertension}=1) > 2.7 \times P(\text{stroke}=1)$. Điều này chứng minh trên phương diện toán học rằng cao huyết áp là một trong những yếu tố nguy cơ hàng đầu thúc đẩy biến cố đột quỵ mạch máu não.

### 3.2. Tối Ưu Hóa Hàm Mất Mát Bằng Gradient Descent Từ Đầu
Thuật toán hồi quy tuyến tính được cấu hình để dự đoán xu hướng từ các thuộc tính sức khỏe nền tảng. Quá trình tối ưu hóa trọng số $\mathbf{w}$ được thực hiện qua việc cực tiểu hóa hàm **Mean Squared Error (MSE)**:
$$J(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^{N} \left( h_\mathbf{w}(x^{(i)}) - y^{(i)} \right)^2$$

Đạo hàm riêng (Phần tử Gradient) được cập nhật liên tục qua từng vòng lặp (Epochs):
$$\mathbf{w}_j := \mathbf{w}_j - \alpha \frac{1}{N} \sum_{i=1}^{N} \left( h_\mathbf{w}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

Đồ thị `gradient_descent_convergence.png` cho thấy tốc độ hội tụ ổn định, hàm mất mát trên cả tập huấn luyện (Train) và tập kiểm thử (Test) tiệm cận về điểm cực tiểu toàn cục mà không xảy ra hiện tượng Overfitting.

---

## 4. Module 3: Pandas EDA, Kỹ Nghệ Đặc Trưng VÀ Trực Quan Hóa 

### 4.1. Xử Lý Rác Dữ Liệu Và Mã Hóa Biến Phân Loại
- Loại bỏ dòng chứa dữ liệu giới tính không xác định (`gender = "Other"`) nhằm loại bỏ nhiễu tần suất thấp (Low-frequency anomalies).
- Triển khai **One-Hot Encoding** thông qua hàm `pd.get_dummies()` kết hợp thuộc tính cấu hình `drop_first=True` nhằm triệt tiêu hoàn toàn hiện tượng đa cộng tuyến (Multicollinearity / Dummy Variable Trap) trong các mô hình tuyến tính y sinh.

### 4.2. Kỹ Nghệ Đặc Trưng Định Hướng Y Tế (Healthcare Feature Engineering)
Dự án đã tạo lập đặc trưng phái sinh mang tính tương tác sinh học cao:
- **`glucose_to_bmi_ratio`:** Tỷ lệ giữa chỉ số đường huyết trung bình và chỉ số khối cơ thể. Chỉ số này phản ánh mối tương quan động giữa mức độ hấp thụ đường và thể trạng béo phì.
- **Phân nhóm tuổi (Age Groups):** Phân mảnh theo chuẩn xã hội học: *Child* ($0-18$), *Adult* ($18-60$), *Senior* ($>60$).
- **Phân loại thể trạng (BMI Categories):** Áp dụng nghiêm ngặt theo thang phân loại chuẩn của Tổ chức Y tế Thế giới (WHO): *Underweight*, *Normal*, *Overweight*, *Obese*.

### 4.3. Phân Tích Ma Trận Tương Quan (Tránh Lỗi Trực Quan Hóa)
Trong quá trình hiển thị Ma trận tương quan bằng Seaborn (`sns.heatmap`), hệ thống phát hiện một bug giao diện kinh điển liên quan đến việc sắp xếp tọa độ chữ chú thích (`annot=True`), gây mất số hiển thị ở các hàng bên dưới. 
- **Giải pháp xử lý đột phá:** Trích xuất ma trận giá trị nguyên thủy (`corr_matrix.values`), cấu hình thủ công hệ thống nhãn trục (`xticklabels`, `yticklabels`), đồng thời nhúng trực tiếp vòng lặp lồng Matplotlib (`ax.text()`) để tự động đổi màu chữ tương phản dựa trên trọng số màu nền. Kết quả đồ thị `Correlation_heatmap.png` đạt độ trực quan tuyệt đối, hiện rõ hệ số tương quan của biến mục tiêu `stroke` với tất cả các thuộc tính định lượng.

---

## 5. Thách Thức Mất Cân Bằng Dữ Liệu & Đề Xuất Machine Learning
Phân tích phân phối tần suất lớp mục tiêu (`stroke`) chỉ ra hiện tượng **Mất cân bằng dữ liệu nghiêm trọng (Severe Class Imbalance)**:
- Tỷ lệ ca không đột quỵ (`stroke = 0`) chiếm áp đảo với hơn 95% tổng thể mẫu.
- Tỷ lệ ca đột quỵ thực tế (`stroke = 1`) chỉ chiếm dưới 5%.

### Đề xuất chiến lược xây dựng mô hình ML tiếp theo:
1. **Tiền xử lý mẫu:** Bắt buộc áp dụng thuật toán **SMOTE (Synthetic Minority Over-sampling Technique)** để sinh mẫu nhân tạo cho nhóm thiểu số, hoặc áp dụng kỹ thuật **Under-sampling** có kiểm soát cho nhóm đa số để tránh mô hình bị thiên vị (Bias).
2. **Lựa chọn thuật toán:** Ưu tiên các mô hình cây quyết định phân cấp (Ensemble Learning) như **Random Forest** hoặc **XGBoost**, cấu hình tham số trọng số lớp (`class_weight='balanced'`).
3. **Chỉ số đánh giá:** Tuyệt đối không dùng chỉ số *Accuracy* (Độ chính xác tổng thể). Thay vào đó, dùng **F1-Score**, **Recall** (đảm bảo không bỏ sót ca bệnh thực tế) và chỉ số **AUC-ROC**.

---

## 6. Kết Luận
Dự án mini-project này đã hoàn thiện xuất sắc mục tiêu tiền xử lý, nghiên cứu toán học y sinh và chuẩn hóa toàn diện tập dữ liệu đầu vào. Toàn bộ mã nguồn sạch, cấu trúc thư mục chuẩn hóa kiến trúc dự án và tệp dữ liệu cuối cùng `clean_stroke_data.csv` đã được lưu trữ an toàn, sẵn sàng làm đầu vào tối ưu cho các pipeline huấn luyện Machine Learning chuyên sâu.
