import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import get_raw_data

# Đảm bảo thư mục lưu ảnh tồn tại
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

def load_and_explore():
    """
    Q1: Tải tập dữ liệu vào Pandas DataFrame và hiển thị head(), info(), describe().
    """
    print("=== Q1: Load & Explore Data ===")
    df = get_raw_data()
    
    print("1. Trích xuất 5 dòng đầu tiên (head):")
    print(df.head(), "\n")
    
    print("2. Thông tin cấu trúc dữ liệu (info):")
    df.info()
    print("\n")
    
    print("3. Thống kê mô tả (describe):")
    print(df.describe(), "\n")
    
    return df

def clean_data(df):
    """
    Q2: Làm sạch dữ liệu: xử lý giá trị thiếu ở cột bmi và loại bỏ dữ liệu không hợp lệ.
    """
    print("=== Q2: Clean Data ===")
    # Điền giá trị thiếu của bmi bằng giá trị trung bình
    bmi_mean = df['bmi'].mean()
    df['bmi'] = df['bmi'].fillna(bmi_mean)
    
    # Loại bỏ các hàng có giới tính là "Other" 
    df = df[df['gender'] != 'Other']
    
    print(f"Số lượng missing values sau khi làm sạch:\n{df.isnull().sum()}\n")
    return df

def analyze_smoking(df):
    """
    Q3: Nhóm theo smoking_status, tính tổng số bệnh nhân và tỷ lệ đột quỵ.
    """
    print("=== Q3: Smoking Status Analysis ===")
    smoking_analysis = df.groupby('smoking_status').agg(
        Total_Patients=('stroke', 'count'),
        Stroke_Rate=('stroke', 'mean')
    ).reset_index()
    
    smoking_analysis['Stroke_Rate'] = (smoking_analysis['Stroke_Rate'] * 100).round(2).astype(str) + '%'
    print(smoking_analysis, "\n")

def feature_engineering_ratio(df):
    """
    Q4: Tạo đặc trưng mới glucose_to_bmi_ratio và phân tích tương quan.
    """
    print("=== Q4: Glucose to BMI Ratio ===")
    df['glucose_to_bmi_ratio'] = df['avg_glucose_level'] / df['bmi']
    
    correlation = df['glucose_to_bmi_ratio'].corr(df['stroke'])
    print(f"Hệ số tương quan giữa glucose_to_bmi_ratio và stroke: {correlation:.4f}\n")
    return df

def visualize_distributions(df):
    """
    Q5: Vẽ biểu đồ phân phối tuổi (histogram) và so sánh số ca đột quỵ (bar chart).
    """
    print("=== Q5: Visualizations ===")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='stroke', palette='Set2')
    plt.title('Stroke Counts (0 = No, 1 = Yes)')
    plt.xlabel('Stroke')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/Age_stroke_dist.png')
    plt.close()
    print("- Đã lưu Age_stroke_dist.png\n")

def work_type_pivot(df):
    """
    Q6: Tạo pivot table với work_type làm index, stroke làm columns.
    """
    print("=== Q6: Work Type Pivot Table ===")
    pivot = pd.pivot_table(df, values='id', index='work_type', columns='stroke', aggfunc='size', fill_value=0)
    print(pivot, "\n")

def age_groups(df):
    """
    Q7: Phân nhóm tuổi (Child, Adult, Senior) và phân tích tỷ lệ đột quỵ.
    """
    print("=== Q7: Age Groups Analysis ===")
    bins = [0, 18, 60, 100]
    labels = ['Child', 'Adult', 'Senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    
    age_analysis = df.groupby('age_group', observed=False)['stroke'].mean().reset_index()
    age_analysis['stroke_rate'] = (age_analysis['stroke'] * 100).round(2).astype(str) + '%'
    print(age_analysis[['age_group', 'stroke_rate']], "\n")
    return df

def bmi_categories(df):
    """
    Q8: Phân loại BMI (Underweight, Normal, Overweight, Obese).
    """
    print("=== Q8: BMI Categories Analysis ===")
    bins = [0, 18.5, 24.9, 29.9, 100]
    labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df['bmi_category'] = pd.cut(df['bmi'], bins=bins, labels=labels)
    
    bmi_analysis = df.groupby('bmi_category', observed=False)['stroke'].mean().reset_index()
    bmi_analysis['stroke_rate'] = (bmi_analysis['stroke'] * 100).round(2).astype(str) + '%'
    print(bmi_analysis[['bmi_category', 'stroke_rate']], "\n")
    return df

def advanced_visualizations(df):
    """
    Q9: Biểu đồ tương quan (Heatmap) và Boxplot của avg_glucose_level theo stroke.
    """
    print("=== Q9: Advanced Visualizations ===")    
    plt.figure(figsize=(10, 8))
    
    num_cols = df.select_dtypes(include=[np.number]).drop(columns=['id'], errors='ignore')
    corr_matrix = num_cols.corr().astype(float)
    corr_vals = corr_matrix.values
    
    ax = sns.heatmap(
        corr_vals,               
        annot=False, 
        cmap='coolwarm', 
        vmin=-1, vmax=1,                 
        linewidths=0.5,
        xticklabels=corr_matrix.columns, 
        yticklabels=corr_matrix.columns
    )
    
    for i in range(corr_vals.shape[0]):
        for j in range(corr_vals.shape[1]):
            val = corr_vals[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}",
                    ha='center', va='center', color=text_color, 
                    fontsize=12, fontweight='bold')

    plt.title('Correlation Heatmap (Numeric Features)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig('outputs/figures/Correlation_heatmap.png', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.boxplot(x='stroke', y='avg_glucose_level', data=df, palette='Set1')
    plt.title('Avg Glucose Level grouped by Stroke')
    plt.tight_layout()
    plt.savefig('outputs/figures/Glucose_boxplot.png', dpi=300)
    plt.close()
    
    print("-> Saved: Correlation_heatmap.png and Glucose_boxplot.png")

def encode_categorical(df):
    """
    Q10: Mã hóa biến phân loại (One-Hot Encoding) chuẩn bị cho ML.
    """
    print("=== Q10: Feature Encoding ===")
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'age_group', 'bmi_category']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df_encoded = df_encoded.drop(columns=['id'])
    
    print(f"Kích thước bộ dữ liệu ban đầu: {df.shape}")
    print(f"Kích thước bộ dữ liệu sau mã hóa: {df_encoded.shape}")
    print("Các cột mới được tạo ra (mẫu):")
    print(list(df_encoded.columns)[-5:]) 
    print("\nDữ liệu đã sẵn sàng để đưa vào các mô hình Machine Learning!")
    
    df_encoded.to_csv('data/processed/clean_stroke_data.csv', index=False)
    
    return df_encoded

def main():
    print("=== BẮT ĐẦU MODULE 3: PANDAS ===")
    df = load_and_explore()
    df = clean_data(df)
    analyze_smoking(df)
    df = feature_engineering_ratio(df)
    visualize_distributions(df)
    work_type_pivot(df)
    df = age_groups(df)
    df = bmi_categories(df)
    advanced_visualizations(df)
    df_final = encode_categorical(df)

if __name__ == "__main__":
    main()