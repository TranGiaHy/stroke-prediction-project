import pandas as pd
import numpy as np
from utils import get_raw_data

def load_and_clean_data():
    """
    Q1: Tải dữ liệu, xử lý giá trị thiếu (NaN) 
        kiểm tra kích thước (shape) và kiểu dữ liệu (dtype) cảu mảng
    """
    print("=== Q1 : Load and Clean Data ===")
    df = get_raw_data()
    # Lấy các đặc trưng số: age, avg_glucose_level, bmi
    num_cols = ['age', 'avg_glucose_level', 'bmi']
    # .values để chuyển thành mảng NumPy
    data = df[num_cols].values

    # Xử lý NaN và thay NaN bằng giá trị trung bình của cột bmi
    col_means = np.nanmean(data, axis=0)
    nan_indices = np.where(np.isnan(data))
    data[nan_indices] = np.take(col_means, nan_indices[1])

    print(f"Shape của mảng: {data.shape}")
    print(f"Dtype của mảng: {data.dtype}\n")
    return data

def compute_statistics(data):
    """
    Q2: Tính Mean, Median, và Std Dev
    """
    print("=== Q2 : Statistics ===")
    # axis=0 để tính toán dọc theo các hàng, cột
    mean_vals = np.mean(data, axis=0)
    median_vals = np.median(data, axis=0)
    std_vals = np.std(data, axis=0)

    features = ['age', 'avg_glucose_level', 'bmi']
    for i, feat in enumerate(features):
        print(f"{feat:18} - Mean: {mean_vals[i]:.2f}, Median: {median_vals[i]:.2f}, Std : {std_vals[i]:.2f}")
    print()

def min_max_normalization(data):
    """
    Q3: Chuẩn hóa Min_Max bằng vector hóa
    """
    print("=== Q3 : Min-Max Normalization ===")
    d_min = np.min(data, axis=0)
    d_max = np.max(data, axis=0)
    # Sử dụng công thức chuẩn hóa để tính X(norm) = X - X(min) / X(max) - X(min)
    scaled_data = (data - d_min) / (d_max - d_min)
    print(f"Min của dữ liệu sau chuẩn hóa: {np.min(scaled_data, axis=0)}")
    print(f"Max của dữ liệu sau chuẩn hóa: {np.max(scaled_data, axis=0)}\n")
    return scaled_data

def filter_patients(data):
    """
    Q4: Lọc bệnh nhân (age > 50 and avg_glucose_level > mean)
    """
    print("=== Q4 : Filter Patients")
    mean_glucose = np.mean(data[:, 1])
    cols = {'age': 0, 'avg_glucose_level': 1, 'bmi': 2}
    mask = (data[:, cols['age']] > 50) & (data[:, cols['avg_glucose_level']] > mean_glucose)
    filtered_data = data[mask]
    print(f"Số lượng bệnh nhân thỏa mãn điều kiện: {filtered_data.shape[0]}\n")

def compute_correlation(data):
    """
    Q5: Tính ma trận tương quan
    """
    print("=== Q5 : Correlation Matrix ===")
    # rowvar=False vì mỗi cột là một biến (đặc trưng), mỗi hàng là một quan sát
    corr_matrix = np.corrcoef(data, rowvar=False)
    print(f"Ma Trận tương quan:\n {corr_matrix} \n")

def computer_pairwise_distance(data):
    """
    Q6: Khoảng cách Euclidean theo cặp (hoàn toàn vector hóa)
    """
    print("=== Q6 : Pairwise Euclidean Distance ===")
    # Để tránh MemoryError với ma trận lớn 
    # Lấy mẫu 100 bệnh nhân đầu tiên
    subset = data[:100]
    # Dùng np.newaxis để mở rộng chiều
    # subset[:, np.newaxis, :] biến mảng (3, 3) thành (3, 1, 3)
    # subset[np.newaxis, :, :] biến mảng (3, 3) thành (1, 3, 3)
    diff = subset[:, np.newaxis, :] - subset[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    print(f"Shape của ma trận khoảng cách (100 bệnh nhân): {distances.shape}\n")
    return distances

def compare_normalizations(data):
    """
    Q7: So sánh Min-Max và Z-score
    """
    print("=== Q7 : Min-Max and Z-score ===")
    d_min = np.min(data, axis=0)
    d_max = np.max(data, axis=0)
    min_max_data = (data - d_min) / (d_max - d_min)

    # Z-score 
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    z_score_data = (data - mean) / std

    print(f"Min-max Data  \n- Mean: {np.mean(min_max_data, axis=0)}, \n- Std : {np.std(min_max_data, axis=0)}")
    print(f"Z-score Data  \n- Mean: {np.mean(z_score_data, axis=0)}, \n- Std : {np.std(z_score_data, axis=0)}\n")

def compute_cosine_similarity(data):
    """
    Q8: Cosine Similarity giữa 2 bênh nhân
    """
    print("=== Q8: Cosine Similarity ===")
    # Lấy bệnh nhân ở vị trí 0 và 1
    p1 = data[0]
    p2 = data[1]

    dot_product = np.dot(p1, p2)
    norm_p1 = np.linalg.norm(p1)
    norm_p2 = np.linalg.norm(p2)
    cos_sim = dot_product / (norm_p1 * norm_p2)
    print(f"Cosine Similarity giữa bệnh nhân 1 và 2: {cos_sim:.4f}\n")

def manual_pca(data):
    """
    Q9: Triển khai PCA thủ công
    """
    print("=== Q9 : Manual PCA ===")
    # Trừ đi mean (Centering)
    centered_data = data - np.mean(data, axis=0)

    # Xây dựng ma trận hiệp phương sai
    cov_matrix = np.cov(centered_data, rowvar=False)

    # Tính eigenvalues và eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sắp xếp giảm dần theo eigenvalues và lấy top 2 components
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_2_eigenvectors = eigenvectors[:, sorted_indices[:2]]

    # Chiếu dữ liệu xuống không gian 2 chiều
    pca_data = np.dot(centered_data, top_2_eigenvectors)

    print(f"Shape của dữ liệu gốc: {data.shape}")
    print(f"Shape của dữ liệu sau PCA: {pca_data.shape}\n")
    return pca_data

def process_in_batches(data, num_batches=5):
    """
    Q10: Xử lý theo lô (Batch processing)
    """
    print("=== Q10 : Batch Processing ===")
    # Chia mảng thành các lô
    batches = np.array_split(data, num_batches)

    for i, batch in enumerate(batches):
        batch_mean = np.mean(batch, axis=0)
        print(f"Lô {i+1} (Size: {batch.shape[0]}) - Vector trung bình: {batch_mean}")
    print()

def main():
    data = load_and_clean_data()
    compute_statistics(data)
    min_max_normalization(data)
    filter_patients(data)
    compute_correlation(data)
    computer_pairwise_distance(data)
    compare_normalizations(data)
    compute_cosine_similarity(data)
    manual_pca(data)
    process_in_batches(data)

if __name__ == "__main__":
    main()
