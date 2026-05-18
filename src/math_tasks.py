import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import get_raw_data

def load_math_data():
    """
    Tải dữ liệu cần thiết cho Module 2, bao gồm cả biến mục tiêu và biến phân loại.
    """
    df = get_raw_data()
    
    # Xử lý NaN cho BMI
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
    df = df[df['gender'] != 'Other']    

    # Lấy ma trận X (đặc trưng) và vector y (mục tiêu)
    X_num = df[['age', 'avg_glucose_level', 'bmi']].values
    y = df['stroke'].values.reshape(-1, 1)
    
    return df, X_num, y

def analyze_matrix_properties(X):
    """
    Q1: Kích thước và Hạng (Rank) của ma trận X.
    """
    print("=== Q1: Matrix Shape and Rank ===")
    shape = X.shape
    rank = np.linalg.matrix_rank(X)
    print(f"Kích thước ma trận X: {shape}")
    print(f"Hạng của ma trận X: {rank}\n")

def compute_linear_regression_analytic(df):
    """
    Q2: Hồi quy tuyến tính bằng Phương trình chuẩn (Normal Equation).
    w = (X^T * X)^-1 * X^T * y
    """
    print("=== Q2: Linear Regression (Analytic Solution) ===")
    X_q2 = df[['age', 'avg_glucose_level']].values
    y = df['stroke'].values.reshape(-1, 1)
    
    # Thêm cột bias (số 1) vào X 
    X_b = np.c_[np.ones((X_q2.shape[0], 1)), X_q2]
    
    # Tính toán trọng số tối ưu
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    print(f"Bias (b): {theta_best[0][0]:.5f}")
    print(f"Trọng số Age (w1): {theta_best[1][0]:.5f}")
    print(f"Trọng số Glucose (w2): {theta_best[2][0]:.5f}\n")
    return X_b, y, theta_best

def compute_probabilities(df):
    """
    Q3: Xác suất cơ sở và Xác suất có điều kiện.
    """
    print("=== Q3: Probabilities ===")
    p_stroke = df['stroke'].mean()
    
    hyper_df = df[df['hypertension'] == 1]
    p_stroke_given_hyper = hyper_df['stroke'].mean()
    
    print(f"P(stroke=1) = {p_stroke:.4f}")
    print(f"P(stroke=1 | hypertension=1) = {p_stroke_given_hyper:.4f}\n")

def compute_mse_gradient(X_b, y, theta):
    """
    Q4: Tính Gradient của MSE đối với trọng số W.
    """
    print("=== Q4: MSE Gradient ===")
    n = len(y)
    predictions = X_b.dot(theta)
    errors = predictions - y
    gradient = (2/n) * X_b.T.dot(errors)
    
    print(f"Gradient tại điểm tối ưu:\n {gradient}\n")

def compute_variance_covariance(df):
    """
    Q5: Phương sai và Hiệp phương sai.
    """
    print("=== Q5: Variance & Covariance ===")
    glucose = df['avg_glucose_level'].values
    age = df['age'].values
    
    var_glucose = np.var(glucose)
    cov_matrix = np.cov(glucose, age)
    cov_glucose_age = cov_matrix[0, 1]
    
    print(f"Phương sai (Variance) của Glucose: {var_glucose:.2f}")
    print(f"Hiệp phương sai (Covariance) giữa Glucose và Age: {cov_glucose_age:.2f}\n")

def compute_eigen_decomposition(X):
    """
    Q6: Ma trận hiệp phương sai và Eigenvalues/Eigenvectors.
    """
    print("=== Q6: Covariance Matrix & Eigen decomposition ===")
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    print(f"Eigenvalues:\n{eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}\n")

def perform_svd(X):
    """
    Q7: Singular Value Decomposition (SVD).
    """
    print("=== Q7: SVD Decomposition ===")
    X_centered = X - np.mean(X, axis=0)
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
    
    print(f"Kích thước U: {U.shape} -> Biểu diễn mối quan hệ giữa các bệnh nhân.")
    print(f"Kích thước Sigma (S): {S.shape} -> Mức độ quan trọng của các đặc trưng ẩn.")
    print(f"Kích thước V^T: {VT.shape} -> Trục tọa độ mới để giảm chiều dữ liệu.\n")

def optimize_with_gradient_descent(X_b, y, learning_rate=0.01, n_iterations=1000):
    """
    Q8: Tối ưu bằng Gradient Descent. Đã áp dụng chuẩn hóa Z-score để tránh Exploding Gradient.
    """
    print("=== Q8: Gradient Descent ===")

    # Chia Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_b, y, test_size=0.2, random_state=42)
    n_train = len(y_train)
    n_test = len(y_test)

    # Tính toán Mean/Std trên tập train
    X_train_scaled = np.copy(X_train)
    X_test_scaled = np.copy(X_test)

    X_train_features = X_train_scaled[:, 1:]
    X_test_features = X_test_scaled[:, 1:] 
    
    mean_train = np.mean(X_train_features, axis=0)
    std_train = np.std(X_train_features, axis=0)

    # Chuẩn hóa
    X_train_scaled[:, 1:] = (X_train_features - mean_train) / std_train
    X_test_scaled[:, 1:] = (X_test_features - mean_train) / std_train

    # Thực hiện Gradient Descent trên tập train
    theta = np.random.randn(X_train_scaled.shape[1], 1)
    loss_history = []
    
    for iteration in range(n_iterations):
        predictions = X_train_scaled.dot(theta)
        errors = predictions - y_train
        
        mse = (1/n_train) * np.sum(errors**2)
        loss_history.append(mse)
        
        gradients = (2/n_train) * X_train_scaled.T.dot(errors)
        theta = theta - learning_rate * gradients
    
    # Đánh giá mô hình trên tập Test
    test_predictions = X_test_scaled.dot(theta)
    test_mse = (1/n_test) * np.sum((test_predictions - y_test)**2)


    print(f"Trọng số sau {n_iterations} vòng lặp GD:\n{theta}")
    print(f"MSE tập Train: {loss_history[-1]:.4f}\n")
    print(f"MSE tập Test: {test_mse:.4f}\n")
    
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history[:50])
    plt.title("Hội tụ của MSE trong Gradient Descent (Train Set)")
    plt.xlabel("Iterations")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig('outputs/figures/gradient_descent_convergence.png')
    plt.close()

def apply_bayes_theorem(df):
    """
    Q9: Định lý Bayes với glucose bins.
    """
    print("=== Q9: Bayes Theorem ===")
    bins = [0, 100, 125, np.inf]
    labels = ['Normal', 'Prediabetic', 'Diabetic']
    df['glucose_bin'] = pd.cut(df['avg_glucose_level'], bins=bins, labels=labels)
    
    p_stroke = df['stroke'].mean()
    
    for label in labels:
        p_bin = len(df[df['glucose_bin'] == label]) / len(df)
        stroke_cases = df[df['stroke'] == 1]
        p_bin_given_stroke = len(stroke_cases[stroke_cases['glucose_bin'] == label]) / len(stroke_cases)
        
        p_stroke_given_bin = (p_bin_given_stroke * p_stroke) / p_bin if p_bin > 0 else 0
        print(f"P(Stroke=1 | {label}) = {p_stroke_given_bin:.4f}")
    print()

def compare_regularization_methods(X_b, y):
    """
    Q10: So sánh L2 (Ridge) và L1 (Lasso). Sửa lỗi Gradient Descent cho L1.
    """
    print("=== Q10: L1 & L2 Regularization ===")
    lambda_reg = 100 
    
    # Chia Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_b, y, test_size=0.2, random_state=42)
    n_train = len(y_train)

    # Ridge (L2) - Dùng công thức đại số trên tập Train
    I = np.eye(X_train.shape[1])
    I[0, 0] = 0 
    theta_ridge = np.linalg.inv(X_train.T.dot(X_train) + lambda_reg * I).dot(X_train.T).dot(y_train)
    print(f"Trọng số Ridge (L2):\n{theta_ridge}")
    
    # Lasso (L1) - Gradient Descent với tập train đã chuẩn hóa đúng chuẩn
    X_train_scaled = np.copy(X_train)
    X_train_features = X_train_scaled[:, 1:]

    mean_train = np.mean(X_train_features, axis=0)
    std_train = np.std(X_train_features, axis=0)

    X_train_scaled[:, 1:] = (X_train_features - mean_train) / std_train

    theta_lasso = np.random.randn(X_train_scaled.shape[1], 1)
    learning_rate = 0.01 # Dùng lr lớn hơn sau khi chuẩn hóa
    
    for _ in range(1000):
        predictions = X_train_scaled.dot(theta_lasso)
        errors = predictions - y_train
        gradients = (2/n_train) * X_train_scaled.T.dot(errors)
        
        l1_penalty = lambda_reg * np.sign(theta_lasso)
        l1_penalty[0] = 0 # Không phạt bias
        
        theta_lasso = theta_lasso - learning_rate * (gradients + l1_penalty)
        
    print(f"Trọng số Lasso (L1):\n{theta_lasso}")

def main():
    df, X_num, y = load_math_data()
    analyze_matrix_properties(X_num)
    X_b, y_q2, theta_best = compute_linear_regression_analytic(df)
    compute_probabilities(df)
    compute_mse_gradient(X_b, y_q2, theta_best)
    compute_variance_covariance(df)
    compute_eigen_decomposition(X_num)
    perform_svd(X_num)
    optimize_with_gradient_descent(X_b, y_q2)
    apply_bayes_theorem(df.copy())
    compare_regularization_methods(X_b, y_q2)

if __name__ == "__main__":
    main()