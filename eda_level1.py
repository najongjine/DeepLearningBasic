import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

# 1. 데이터 가져오기 (level1.py와 동일)
x_train = torch.FloatTensor([[1], [2], [3], [4], [5]])
y_train = torch.FloatTensor([[12], [22], [32], [42], [52]])

# 시각화를 위해 numpy/pandas 데이터로 변환
x_data = x_train.numpy().flatten()
y_data = y_train.numpy().flatten()

df = pd.DataFrame({
    'Cows': x_data,
    'Feed_kg': y_data
})

# 2. EDA 시각화 설정
plt.figure(figsize=(12, 5))

# (1) Scatter Plot: 소 마리 수와 사료 양의 상관관계
plt.subplot(1, 2, 1)
plt.scatter(df['Cows'], df['Feed_kg'], color='blue', marker='o', s=100)
plt.title('Scatter Plot: Cows vs Feed')
plt.xlabel('Number of Cows')
plt.ylabel('Feed (kg)')
plt.grid(True, linestyle='--', alpha=0.7)

# (2) Box Plot: 데이터 분포 확인
plt.subplot(1, 2, 2)
plt.boxplot([df['Cows'], df['Feed_kg']], labels=['Cows', 'Feed_kg'])
plt.title('Box Plot: Data Distribution')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

# 3. 결과 저장 및 출력
save_path = 'eda_result.png'
plt.savefig(save_path)
print(f"EDA result saved to {os.path.abspath(save_path)}")

# 4. 데이터 통계 요약 출력
print("\n--- Data Summary ---")
print(df.describe())
