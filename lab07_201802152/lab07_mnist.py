import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import csv


"""
    pytorch에서 제공하는 data loader가 있다. 이를 이용하면 손 쉽게 batch를 구성할 수 있다.
    x, y를 set로 묶고자.
    일반적으로 data를 한 번 섞어준다. 
    예상이 필요한 값과 정답으로 묶어줌
    option을 true를 주면 랜덤하게 쌍을 묶어주고 섞어준다.
    
    맞는지 틀린지 정답을 도출해 낼 때 채점 방식을 지정해 준다.3333
    
    loss:실제로 정답을 채점하는 방향
"""

def read_data(train_path, test_path):
    # train_path에 있는 파일을 읽음
    train = pd.read_csv(train_path)
    # 결과값
    y_train = train['label']
    y_list = np.zeros(shape=(y_train.size, 10))
    # i : 인덱스
    # print(y_train)

    for i, y in enumerate(y_train):
        y_list[i][y] = 1
    y_train = y_list
    # label열 삭제
    del train['label']
    # 0 ~ 1 사이의 값으로 scaling
    x_train = train.to_numpy() / 255
    # test.csv 파일을 읽음
    test = pd.read_csv(test_path)
    # 0 ~ 1 사이의 값으로 scaling
    x_test = test.to_numpy() / 255
    return x_train, y_train, x_test

# 상속을 해야 구현 가능.
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.f1 = nn.Linear(784, 512)
        self.f2 = nn.Linear(512, 256)
        self.f3 = nn.Linear(256, 10)
        """
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
        """
    # sigmoid: 0~1 값으로 표현
    # torch.sigmoid(self.fc3(x))
    def forward(self, x):
        x1 = torch.relu(self.f1(x))
        x2 = torch.relu(self.f2(x1))
        x3 = self.f3(x2)
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        """
        return x3

# pred: prediction
# mse: 틀린 차이를 리턴
def get_acc(pred, answer):
    correct = 0
    # p에서 가장 큰 값이 정답.
    for p, a in zip(pred, answer):
        pv, pi = p.max(0)
        av, ai = a.max(0)
        if pi == ai:
            correct += 1
    return correct / len(pred)

# 학습 부분
def train(x_train, y_train, batch, lr, epoch):
    model = MNISTModel()
    model.train() # 이 모델을 학습을 하는데 사용하겠다는 뜻
    #mse는 0이 나와야 좋은 것. 빼기를 하기 때문
    loss_function = nn.MSELoss(reduction="mean") # 모든 결과를 더할거냐? 평균값을 구할 것이냐? 채점하는 거
    optimizer = optim.Adam(model.parameters(), lr=lr) # 틀린 것을 다시 공부 하는 것 학습하는 것
    # data 처리
    # x_train을 torch에 맞는 tensor로 바꿔줌. 그것을 float 형태로
    x = torch.from_numpy(x_train).float()
    y = torch.from_numpy(y_train).float()
    # zip: set의 형태로 사용 가능
    data_loader = torch.utils.data.DataLoader(list(zip(x, y)), batch, shuffle=True)
    epoch_loss = []
    epoch_acc = [] # accuracy
    for e in range(epoch):
        total_loss = 0
        total_acc = 0 # accuracy
        # data : batch data?
        for data in data_loader: # 알아서 학습이 됨
            # forward 부분이 실행된다
            x_data, y_data = data

            # forward 문제 풀이
            pred = model(x_data)# prediction

            # backward 채점 및 학습
            loss = loss_function(pred, y_data) # prediction and right result

            optimizer.zero_grad() # 이전의 학습 결과를 reset 한다
            loss.backward() # 학습 한다.

            optimizer.step() # update

            total_loss += loss.item()
            total_acc += get_acc(pred, y_data)

        epoch_loss.append(total_loss / len(data_loader))
        epoch_acc.append(total_acc / len(data_loader))
        print("Epoch [%d] Loss: %.5f\tAcc: %.5f" % (e + 1, epoch_loss[e], epoch_acc[e]))
    return model, epoch_loss, epoch_acc


def test(model, x_test, batch):
    model.eval() # 학습이 아닌 평가 모드로

    x = torch.from_numpy(x_test).float()

    # shuffle을 하면 안됨 -> 정담을 순서대로 해야하기 때문
    # epoch이 없기 때문에 바로 돌리면 됨
    data_loader = torch.utils.data.DataLoader(x, batch, shuffle=False)

    preds = []
    for data in data_loader:
        pred = model(data) # 예측된 결과 값을 가져옴
        for p in pred:
            pv, pi = p.max(0)
            preds.append(pi.item())

    return preds

# cvs 파일로 저장
def save_pred(pred_path, preds):
    with open(pred_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['ImageId', 'Label'])
        index = 1
        for i in range(len(preds)):
            tmp = []
            tmp.append(index)
            tmp.append(preds[i])
            csv_writer.writerow(tmp)
            index += 1


# 그래프를 출력하는 함수
def draw_graph(data):
    plt.plot(data)
    plt.show()



if __name__ == '__main__':
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    save_path = 'data/my_submission.csv'

    # batch = 196
    batch = 64
    lr = 0.001
    #epoch = 10
    epoch = 45

    x_train, y_train, x_test = read_data(train_path, test_path) # 데이터를 가공

    model, epoch_loss, epoch_acc = train(x_train, y_train, batch, lr, epoch)


    #bath : 학습을 더 빨리 하기 위해
    preds = test(model, x_test, batch)
    save_pred(save_path, preds)

    draw_graph(epoch_loss)
    draw_graph(epoch_acc)
