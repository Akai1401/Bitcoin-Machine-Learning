import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv('crypto-com-chain.csv')

# 2. In ra tên các cột trong dữ liệu
print(f"Các cột có trong dữ liệu: {df.columns.tolist()}")

# 3. Kiểm tra kiểu dữ liệu của các cột
print("\nKiểu dữ liệu của các cột:")
print(df.dtypes)

# 4. In 5 dòng đầu tiên của dữ liệu
print("\n5 dòng đầu tiên của dữ liệu:")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# 5. In thống kê mô tả của dữ liệu
print("\nThống kê mô tả của dữ liệu:")
print(df.describe().to_markdown(numalign="left", stralign="left"))

# 6. Lọc các cột cần thiết 'total_volume', 'market_cap', và 'price'
df = df[['total_volume', 'market_cap', 'price']]

# 7. Loại bỏ các dòng có giá trị null
df.dropna(inplace=True)

# 8. Chia dữ liệu thành biến đầu vào (X) và biến đầu ra (y)
X = df[['total_volume', 'market_cap']]
y = df['price']

# 9. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 10. Chuẩn hóa dữ liệu bằng StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 11. Khởi tạo mô hình Neural Network với MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(50, 20),  # Kiến trúc 2 tầng, với 50 và 20 nodes
                     activation='relu',
                     solver='adam',
                     alpha= 1.0,  # Tăng regularization để tránh overfitting
                     learning_rate_init=0.001,  # Giữ nguyên learning rate
                     max_iter=500,  # Giới hạn số lần lặp
                     random_state=42
                    )

# 12. Huấn luyện mô hình
model.fit(X_train_scaled, y_train)

# 13. Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test_scaled)

# 14. Đánh giá mô hình sử dụng R²
r2 = r2_score(y_test, y_pred)
print(f'\nHệ số xác định R²: {r2}')

# 15. In ra hệ số dự đoán (weights) của mô hình
# In các hệ số của mô hình Neural Network
print("\nCác hệ số của mô hình Neural Network:")

# In các hệ số chệch (bias) của mỗi lớp
for i, intercept in enumerate(model.intercepts_):
    print(f"Intercept (Bias) của lớp {i}: {intercept}")

# In các trọng số (weights) giữa các lớp
for i, coef in enumerate(model.coefs_):
    print(f"Trọng số (Weights) giữa lớp {i} và lớp {i+1}:")
    print(coef)

# 16. Tạo biểu đồ phân tán giữa `price` và `total_volume`, `price` và `market_cap`
plt.figure(figsize=(12, 5))

# Biểu đồ phân tán cho `price` vs `total_volume`
plt.subplot(1, 2, 1)
plt.scatter(df['total_volume'], df['price'], color='blue')
plt.title('Biểu đồ phân tán giữa Giá và Tổng khối lượng giao dịch')
plt.xlabel('Tổng khối lượng giao dịch')
plt.ylabel('Giá')

# Biểu đồ phân tán cho `price` vs `market_cap`
plt.subplot(1, 2, 2)
plt.scatter(df['market_cap'], df['price'], color='green')
plt.title('Biểu đồ phân tán giữa Giá và Vốn hóa thị trường')
plt.xlabel('Vốn hóa thị trường')
plt.ylabel('Giá')

# 17. Hiển thị biểu đồ
plt.tight_layout()
plt.show()

# Hàm dự đoán giá sử dụng mô hình Neural Network
def predict_price(total_volume, market_cap):
    """
    Dự đoán giá dựa trên tổng khối lượng giao dịch và vốn hóa thị trường.

    Args:
        total_volume: Tổng khối lượng giao dịch.
        market_cap: Vốn hóa thị trường.

    Returns:
        Giá trị dự đoán của đồng Bitcoin.
    """
    # Tạo DataFrame từ dữ liệu đầu vào với tên cột phù hợp
    input_data = pd.DataFrame([[total_volume, market_cap]], columns=['total_volume', 'market_cap'])
    
    # Chuẩn hóa dữ liệu đầu vào trước khi dự đoán
    input_scaled = scaler.transform(input_data)
    
    # Dự đoán giá
    price = model.predict(input_scaled)[0]
    
    print(f'Với total_volume = {total_volume} và market_cap = {market_cap}, giá trị dự đoán của đồng Bitcoin là: {price}')
    return price


# Chọn một số giá trị mẫu cho total_volume và market_cap
sample_total_volumes = [18353216.27, 5000000, 10000000]
sample_market_caps = [3530339614, 1000000000, 2000000000]

# Dự đoán giá cho từng cặp giá trị mẫu và in ra kết quả
for volume, cap in zip(sample_total_volumes, sample_market_caps):
    predicted_price = predict_price(volume, cap)

