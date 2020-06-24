import numpy as np
import codecs
import sklearn.metrics
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler




#main 함수
if __name__ == '__main__':
    data_list = [] #data를 저장할 list
    #파일을 읽음
    with codecs.open('seoul_tax.txt', 'r', 'utf-8') as seoul:
        first_line = seoul.readline()
        lines = seoul.readlines()
        for line in lines:
            line = line.rstrip() #개행 문자 제거
            data_list.append(line.split('\t')) # 토큰으로 분리




    data_np = np.array(data_list)
    #0번째 열 제거
    data_np = np.delete(data_np, [0], 1)

    data_list = list()
    for col in data_np:
        col = [float(x) for x in col] # float로 변환
        data_list.append(col) # data_list에 저장

    # 함수로 cosine 함수 계산
    cosine_distance = sklearn.metrics.pairwise.cosine_distances(data_list)
    cosine_distance = np.array(cosine_distance)

    #그래프 출력
    plt.pcolor(cosine_distance)
    plt.colorbar()
    plt.show()

    # 맨하탄 거리를 계산하여 저장할 배열
    manhattan_distance = [[0.0]*25 for i in range(25)]

    data_list = np.array(data_list)
    # 두 값의 차이의 절댓값을 배열에 저장
    for i in range(25):
        for j in range(25):
            manhattan_distance[i][j] = sum(abs(data_list[i] - data_list[j]))

    # 맨하탄 거리를 계산한 결과값 그래프로 출력
    plt.pcolor(manhattan_distance)
    plt.colorbar()
    plt.show()

    # 유클라디안 거리를 계산한 후 결과값을 저장할 배열
    euclidean_distance = [[0.0] * 25 for i in range(25)]
    # data_list = np.array(data_list)
    for i in range(25):
        for j in range(25):
            # 내적의 결과값의 제곱근을 구함
            euclidean_distance[i][j] = np.sqrt(np.dot(data_list[i] - data_list[j], data_list[i] - data_list[j]))
    
    #유클라디안 거리를 계산한 결과값을 그래프로 출력
    plt.pcolor(euclidean_distance)
    plt.colorbar()
    plt.show()

    # 데이터를 정규화할 함수
    scaler = MinMaxScaler()

    #데이터를 정규화
    data_list[:] = scaler.fit_transform(data_list[:])

    # 함수로 코사인 거리 계산
    cosine_distance = sklearn.metrics.pairwise.cosine_distances(data_list)
    cosine_distance = np.array(cosine_distance)

    # 정규화한 코사인 거리를 출력
    plt.pcolor(cosine_distance)
    plt.colorbar()
    plt.show()


    data_list = np.array(data_list)

    # 맨하탄 거리 결과값을 저장할 배열
    manhattan_distance = [[0.0] * 25 for i in range(25)]

    # 맨하탄 거리: 두 벡터값의 차이의 절대값
    for i in range(25):
        for j in range(25):
            manhattan_distance[i][j] = sum(abs(data_list[i] - data_list[j]))

    # 정규화된 맨하탄 거리 출력
    plt.pcolor(manhattan_distance)
    plt.colorbar()
    plt.show()

    # 정규화한 유클라디안 거리를 저장할 배열
    euclidean_distance = [[0.0] * 25 for i in range(25)]
    
    # 유클라디안 거리를 계산
    for i in range(25):
        for j in range(25):
            euclidean_distance[i][j] = np.sqrt(np.dot(data_list[i] - data_list[j], data_list[i] - data_list[j]))
    
    #정규화한 그래프 거리를 출력
    plt.pcolor(euclidean_distance)
    plt.colorbar()
    plt.show()