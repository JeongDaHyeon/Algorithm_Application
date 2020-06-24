import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

def make_vocab(vocab_path, train=None):
    vocab = {}
    # 파일이 존재하면 파일을 읽어 옴
    if os.path.isfile(vocab_path):
        file = open(vocab_path, 'r', encoding='utf-8')
        for line in file.readlines():
            line = line.rstrip()
            key,value = line.split('\t')
            vocab[key] = value
        file.close()
    else:
        count_dict = defaultdict(int)
        for index, data in tqdm(train.iterrows(),desc='make vocab',total=len(train)):
            sentence = data['Phrase'].lower()
            tokens = sentence.split(' ')
            for token in tokens:
                count_dict[token] +=1

        file = open(vocab_path,'w',encoding='utf-8')
        file.write('[UNK]\t0\n[PAD]\t1\n')
        vocab = {'[UNK]':0,'[PAD]':1}
        # sorted : python dictionary를 정렬하는 함수
        for index,(token,count) in enumerate(sorted(count_dict.items(),reverse=True,key=lambda item: item[1])):
            vocab[token] = index+2
            file.write(token + '\t' + str(index + 2) + '\n')
        file.close()

    return vocab

def read_data(train, test, vocab, max_len):
    x_train = np.ones(shape=(len(train),max_len)) # train data 크기에 맞게 배열을 생성 padding으로채워줌
    # 0: unknown token   1: padding token -> unknown은 드물다
    for i, data in tqdm(enumerate(train['Phrase']), desc='make x_train data', total=len(train)):
        data = data.lower() # 소문자로 변환
        tokens = data.split(' ') # 공백으로 자름
        for j, token in enumerate(tokens):
            if j == max_len:  # 50이 넘어가면 문장을 자름
                break
            x_train[i][j] = vocab[token]

    x_test = np.ones(shape=(len(test),max_len))
    for i, data in tqdm(enumerate(test['Phrase']), desc='make x_test data', total=len(test)):
        data = data.lower()
        tokens = data.split(' ')
        for j, token in enumerate(tokens):
            if j == 50:
                break
            if token not in vocab.keys():
                x_test[i][j] = 0

            else:
                x_test[i][j] = vocab[token]

    y_train = train['Sentiment'].to_numpy()

    return x_train, y_train,x_test

class RNN(nn.Module):
    # input_size: input 크기
    # output_size: label의 개수
    # num_layer : hidden
    def __init__(self,input_size, embed_size, hidden_size, output_size, num_layers=1, bidirec=False, device='cuda'):
        super(RNN,self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if bidirec:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.device = device

        self.embed = nn.Embedding(input_size,embed_size,padding_idx=1) # embeding을 알아서 해줌
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True,bidirectional=bidirec) # 초기값을 0으로 설정해줌
        self.linear = nn.Linear(hidden_size*self.num_directions,output_size) # 5개로 size를 축소시켜 주는 마지막 부분

    def init_hidden(self,batch_size):
        hidden = torch.zeros(self.num_layers * self.num_directions, batch_size,self.hidden_size).to(self.device)
        cell = torch.zeros(self.num_layers * self.num_directions,batch_size,self.hidden_size).to(self.device)
        return hidden,cell

    def forward(self,inputs):
        embed = self.embed(inputs)
        hidden, cell = self.init_hidden(inputs.size(0)) #초기화

        output,(hidden,cell) = self.lstm(embed,(hidden,cell))

        hidden = hidden[-self.num_directions:]
        hidden = torch.cat([h for h in hidden],1)
        output = self.linear(hidden)

        return output

# pred: prediction
# mse: 틀린 차이를 리턴
def get_acc(pred, answer):
    correct = 0
    # p에서 가장 큰 값이 정답.
    for p, a in zip(pred, answer):
        pv, pi = p.max(0)
        # one-hot encoding 이 아니기때문에 바로 넣어 주면 됨
        # av, ai = a.max(0)
        if pi == a:
            correct += 1
    return correct / len(pred)

def train(x,y,max_len, embed_size, hidden_size, output_size, batch_size, epochs, lr, device,model=None):
    x = torch.from_numpy(x).long()
    y = torch.from_numpy(y).long()
    if model is None:
        model = RNN(max_len,embed_size,hidden_size,output_size,device=device)
    model.to(device)
    model.train() # 학습 모드로 변경

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # data를 batch 만큼 잘라 준다
    data_loader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size, shuffle=True) #데이터섞음
    loss_total = []
    acc_total = []
    # epoch만큼 돈다.
    for epoch in trange(epochs):
        epoch_loss = 0
        epoch_acc =0
        for batch_data in data_loader:
            # to: GPU를 사용하겠다!
            x_batch, y_batch = batch_data
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # forward
            pred = model(x_batch)

            loss = loss_function(pred,y_batch)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step() #업데이트

            epoch_loss +=loss.item()
            epoch_acc +=get_acc(pred,y_batch)
        epoch_loss /= len(data_loader)
        epoch_acc /= len(data_loader)
        loss_total.append(epoch_loss)
        acc_total.append(epoch_acc)

    # 알아서 저장됨
    torch.save(model,'model.out')

    return model,loss_total,acc_total

def test(model,x, batch_size,device):
    model.to(device)
    model.eval() #평가모드
    x = torch.from_numpy(x).long()
    data_loader = torch.utils.data.DataLoader(x, batch_size ,shuffle=False)

    predict=[]
    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        pred = model(batch_data)
        for p in pred:
            pv, pi = p.max(0)
            predict.append(pi.item())

    return predict


# 그래프를 출력하는 부분
def draw_graph(data):
    plt.plot(data)
    plt.show()

# 파일을 저장하는 부분
def save_submission(pred):
    data = {
        "PhraseId": np.arange(156061, len(pred) + 156061),
        "Sentiment": pred
    }
    df = pd.DataFrame(data)
    df.to_csv('data/my_submission.csv', mode='w', index=False)


if __name__ == '__main__':
    train_path = 'data/train.tsv'
    test_path = 'data/test.tsv'
    vocab_path = 'data/vocab.txt'

    train_data = pd.read_csv(train_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')
    vocab = make_vocab(vocab_path, train_data)

    # 훈련을 한 번에 하지 않고 중간에 저장하고 다시 학습을 진행
    # model을 저장하고 불러오기를
    # 알아서 model이 불러와 짐
    # model = torch.load('model.out')
    model = None
    # 0번째 GPU를 사용하겠다는 뜻
    device = torch.device('cuda:0')
    # cpu를 사용하겠다는 뜻
    # device = torch.device('cpu')

    # parameter들을 setting 하는 부분
    # RNN은 과거의 특징들을 사용할 수 있음 -> data 길이가 너무 길면 잘 반영이 되지 않음
    # device = torch.device('cpu')
    max_len=50 # 문장의 최대 길이
    input_size = len(vocab) # vocab의 size
    embed_size = 50
    hidden_size = 100
    output_size = 5 # 5개를 뽑아 내야 한다
    batch_size = 1024
    epochs = 10
    lr = 0.001

    x_train, y_train, x_test = read_data(train_data,test_data,vocab,max_len)

    model, loss_total, acc_total = train(x_train, y_train, input_size, embed_size, hidden_size, output_size, batch_size, epochs, lr, device, model)

    draw_graph(loss_total)
    draw_graph(acc_total)
    predict = test(model, x_test, batch_size, device)
    save_submission(predict)