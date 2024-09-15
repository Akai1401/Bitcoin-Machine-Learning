import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. Read the CSV file into a DataFrame `df`.
df = pd.read_csv('crypto-com-chain.csv')

# 2. Print the column names.
print(f"Các cột có trong dữ liệu: {df.columns.tolist()}")

# 3. Check the data types of all columns.
print("\nKiểu dữ liệu của các cột:")
print(df.dtypes)

# 4. Print the first 5 rows.
print("\n5 dòng đầu tiên của dữ liệu:")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# 5. Print descriptive statistics of the data.
print("\nThống kê mô tả của dữ liệu:")
print(df.describe().to_markdown(numalign="left", stralign="left"))

# 6. Filter the necessary columns `total_volume`, `market_cap`, and `price`.
df = df[['total_volume', 'market_cap', 'price']]

# 7. Drop rows with null values.
df.dropna(inplace=True)

# 8. Split data into features `X` (`total_volume`, `market_cap`) and target `y` (`price`).
X = df[['total_volume', 'market_cap']]
y = df['price']

# 9. Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 10. Initialize the Lasso model.
model = Lasso()

# 11. Fit the model on the training data
model.fit(X_train, y_train)

# 12. Predict on the test set
y_pred = model.predict(X_test)

# 13. Evaluate the model using R-squared and print the result
r2 = r2_score(y_test, y_pred)
print(f'\nHệ số xác định R²: {r2}')

# 14. Print the coefficients of the Lasso model
print("\nCác hệ số của mô hình Lasso:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# 15. Import matplotlib.pyplot
# This was already imported at the beginning of the code

# 16. Create scatter plots of `price` vs `total_volume` and `price` vs `market_cap`
plt.figure(figsize=(12, 5))

# Scatter plot for `price` vs `total_volume`
plt.subplot(1, 2, 1)
plt.scatter(df['total_volume'], df['price'])

# 17. Add titles and labels to the plots
plt.title('Biểu đồ phân tán giữa Giá và Tổng khối lượng giao dịch')
plt.xlabel('Tổng khối lượng giao dịch')
plt.ylabel('Giá')

# Scatter plot for `price` vs `market_cap`
plt.subplot(1, 2, 2)
plt.scatter(df['market_cap'], df['price'])
plt.title('Biểu đồ phân tán giữa Giá và Vốn hóa thị trường')
plt.xlabel('Vốn hóa thị trường')
plt.ylabel('Giá')

# 18. Display the plots
plt.tight_layout()
plt.show()
# Tạo hàm dự đoán giá
def predict_price(total_volume, market_cap):
  """
  Dự đoán giá dựa trên tổng khối lượng giao dịch và vốn hóa thị trường.

  Args:
      total_volume: Tổng khối lượng giao dịch.
      market_cap: Vốn hóa thị trường.

  Returns:
      Giá trị dự đoán của đồng Bitcoin.
  """
  price = model.coef_[0] * total_volume + model.coef_[1] * market_cap + model.intercept_
  print(f'Với total_volume = {total_volume} và market_cap = {market_cap}, giá trị dự đoán của đồng Bitcoin là: {price}')
  return price

# Chọn một số giá trị mẫu cho total_volume và market_cap
sample_total_volumes = [18353216.27
, 5000000, 10000000]
sample_market_caps = [3530339614
, 1000000000, 2000000000]

# Dự đoán giá cho từng cặp giá trị mẫu và in ra kết quả
for volume, cap in zip(sample_total_volumes, sample_market_caps):
    predicted_price = predict_price(volume, cap)