import torch
import torch.nn as nn
import torch.optim as optim

# 1. 데이터 정의 (소 마리 수 -> 사료 kg)
x_train = torch.FloatTensor([[1], [2], [3], [4], [5]])
y_train = torch.FloatTensor([[12], [22], [32], [42], [52]])

# 2. 모델 정의 (딥러닝: 은닉층 추가)
# nn.Linear(1, 1) 한 줄을 아래와 같이 '깊게' 바꿉니다.
model = nn.Sequential(
    nn.Linear(1, 10), # 입력층 -> 은닉층1 (뉴런 10개로 뻥튀기)
    nn.ReLU(),        # 활성화 함수 (비선형성 부여)
    
    nn.Linear(10, 10),# 은닉층1 -> 은닉층2 (뉴런 10개 유지)
    nn.ReLU(),        # 활성화 함수
    
    nn.Linear(10, 1)  # 은닉층2 -> 출력층 (최종 결과 1개로 압축)
)

# 3. 학습 설정 (Optimizer)
# 딥러닝은 파라미터가 많아서 학습률(lr)을 조금 조절하거나 Adam을 쓰기도 합니다.
# 여기서는 그대로 SGD를 쓰되, 학습이 조금 더 오래 걸릴 수 있어 epoch를 넉넉히 잡습니다.
optimizer = optim.Adam(model.parameters(), lr=0.01) 

# 4. 학습 시작
print("--- 딥러닝 학습 시작 ---")
for epoch in range(3001): # 층이 깊어지면 학습이 더 오래 걸립니다 (2000 -> 3000)
    prediction = model(x_train)
    loss = torch.mean((prediction - y_train) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.5f}")

# 5. 결과 확인
test_val = torch.FloatTensor([[10]]) 
pred = model(test_val)

print("\n--- 결과 확인 ---")
print(f"소 10마리일 때 AI 예측값: {pred.item():.2f}kg (정답: 102kg)")

# 주의: 딥러닝 모델은 내부가 복잡해서 Weight, Bias를 딱 하나로 찍어서 보여주기 어렵습니다.
# 대신 '결과값'으로 성능을 판단합니다.