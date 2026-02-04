import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import os

# 1. 데이터 생성 (비선형 & 다변량: 경력, 학력 -> 연봉)
# Experience: 0~20년, Education: 0(고졸), 1(대졸), 2(대학원)
np.random.seed(42)
num_samples = 200

exp = np.random.uniform(0, 20, num_samples)
edu = np.random.randint(0, 3, num_samples)

# 비선형 관계 생성: 연봉 = 기본급 + (경력^1.2 * 150) + (학력^2 * 800) + (경력*학력*50) + 노이즈
salary = 3000 + (exp**1.2 * 150) + (edu**2 * 800) + (exp * edu * 50) + np.random.normal(0, 200, num_samples)

# 시각화를 위한 Pandas DataFrame
df = pd.DataFrame({
    'Experience': exp,
    'Education': edu,
    'Salary': salary
})
print(df.head())

# 2. EDA (데이터 시각화)
print("--- EDA 시작 ---")
fig = plt.figure(figsize=(15, 6))

# (1) 3D Scatter Plot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(df['Experience'], df['Education'], df['Salary'], c=df['Salary'], cmap='viridis')
ax.set_title('3D EDA: Exp & Edu vs Salary')
ax.set_xlabel('Experience (Years)')
ax.set_ylabel('Education (Level)')
ax.set_zlabel('Salary ($)')

# (2) 2D Scatter Plot (Experience vs Salary)
ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(df['Experience'], df['Salary'], alpha=0.5, color='orange')
ax2.set_title('Exp vs Salary')
ax2.set_xlabel('Experience')
ax2.set_ylabel('Salary')

# (3) Box Plot (Education vs Salary)
ax3 = fig.add_subplot(2, 2, 4)
df.boxplot(column='Salary', by='Education', ax=ax3)
ax3.set_title('Salary by Education Level')
plt.suptitle('') # 기본 타이틀 제거

plt.tight_layout()
plt.savefig('level2_eda.png')
print("EDA 결과가 level2_eda.png로 저장되었습니다.")

# 3. 데이터 전처리 (PyTorch 텐서 변환)
# 입력 데이터: [Experience, Education] shape=(N, 2)
x_data = torch.FloatTensor(df[['Experience', 'Education']].values)
y_data = torch.FloatTensor(df[['Salary']].values)

# 정규화 (학습 효율을 위해)
x_mean, x_std = x_data.mean(dim=0), x_data.std(dim=0)
y_mean, y_std = y_data.mean(), y_data.std()

x_train = (x_data - x_mean) / x_std
y_train = (y_data - y_mean) / y_std

# 4. 모델 정의 (은닉층이 있는 신경망)
model = nn.Sequential(
    nn.Linear(2, 32),   # 입력 2개 (Exp, Edu) -> 뉴런 32개
    nn.ReLU(),          # 비선형성 추가
    nn.Linear(32, 16),  # 뉴런 32개 -> 16개
    nn.ReLU(),
    nn.Linear(16, 1)    # 최종 출력 1 (연봉)
)

# 5. 학습 설정
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 6. 학습 시작
print("\n--- 딥러닝 학습 시작 ---")
for epoch in range(4001):
    prediction = model(x_train)
    loss = criterion(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}/4000, Loss: {loss.item():.5f}")

# 7. 결과 확인 및 시각화 (예측값 vs 실제값)
model.eval()
with torch.no_grad():
    y_pred_norm = model(x_train)
    # 정규화 해제
    y_pred = y_pred_norm * y_std + y_mean

# 최종 결과 시각화 (3D Prediction Surface)
fig_res = plt.figure(figsize=(10, 8))
ax_res = fig_res.add_subplot(111, projection='3d')

# 데이터 포인트
ax_res.scatter(df['Experience'], df['Education'], df['Salary'], color='blue', alpha=0.3, label='Actual Data')

# 예측 표면 생성
exp_range = np.linspace(0, 20, 20)
edu_range = np.array([0, 1, 2])
EXP, EDU = np.meshgrid(exp_range, edu_range)

# 표면상의 점들에 대해 예측 수행
grid_points = np.stack([EXP.flatten(), EDU.flatten()], axis=1)
grid_tensor = torch.FloatTensor(grid_points)
grid_tensor_norm = (grid_tensor - x_mean) / x_std

with torch.no_grad():
    z_pred_norm = model(grid_tensor_norm)
    z_pred = z_pred_norm * y_std + y_mean
    Z = z_pred.numpy().reshape(EXP.shape)

ax_res.plot_surface(EXP, EDU, Z, color='red', alpha=0.5, label='Model Prediction')
ax_res.set_title('Deep Learning Prediction Surface')
ax_res.set_xlabel('Experience')
ax_res.set_ylabel('Education')
ax_res.set_zlabel('Salary')

plt.savefig('level2_result.png')
print("최종 학습 결과가 level2_result.png로 저장되었습니다.")

# 8. 임의의 값 테스트
test_input = torch.FloatTensor([[10, 1]]) # 경력 10년, 대졸(1)
test_input_norm = (test_input - x_mean) / x_std
with torch.no_grad():
    test_pred_norm = model(test_input_norm)
    test_pred = test_pred_norm * y_std + y_mean
    print(f"\n[테스트] 경력 10년, 대졸(1)의 예측 연봉: {test_pred.item():.2f}")
