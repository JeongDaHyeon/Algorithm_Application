import codecs
import random
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#euclidean 거리
# 짧은 거리를 출력 -> centers
def get_euclidean(data, center):
    # 좌표값을 주면 거리를 계산하여 리턴.
    # 내적의 값에 제곱근을 취함
    result = np.sqrt(np.dot(data - center, data - center))
    return result


def draw_graph(data, labels):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow')
    plt.show()

#정규화 하는 부분
def norm(data):
    scaler = MinMaxScaler()
    data[:] = scaler.fit_transform(data[:])
    return data

# 데이터를 읽고 필요한 부분만 추출
def read_data():
    data = []
    with codecs.open('covid-19.txt', 'r', 'utf-8') as covid:
        first_line = covid.readline()
        lines = covid.readlines()
        for line in lines:
            line = line.rstrip()
            data.append(line.split('\t'))
    data = np.array(data)
    data = np.delete(data, [0, 1, 2, 3, 4, 7], 1)
    data = np.array(data, dtype=np.float)

    return data

def dbscan(data):
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=0.1, min_samples=2).fit(data)
    draw_graph(data, clustering.labels_)

def hierarchical(data):
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=8).fit(data)
    draw_graph(data, clustering.labels_)

class KMeans:
    #data: 데이터
    #n: 군집수
    def __init__(self, data, n):
        self.data = data
        self.n = n
        self.cluster = OrderedDict() # 군집화된 구조를 체계적으로 나타내고 있음

    # 75개 중에 중복을 제외한 n개의 데이터를 뽑아냄
    # 이거는 초기에 실행되고 끝.
    def init_center(self):
        index = random.randint(0, self.n)
        index_list =[]
        for i in range(self.n):
            # 위에서 뽑은 index가 index_list에 실제로 들어 있는지를 판단
            while index in index_list:
                # 중복된 값이 있으면 새로운 값을 뽑아 옴
                index = random.randint(0, self.n)
            index_list.append(index)
            # key: 중심
            self.cluster[i] = {'center': self.data[index], 'data':[]}

    # 가장 가까운 center 값에 mapping 해줌
    # euclidean 거리를 이용하여 cluster dict 채움
    def clustering(self, cluster):
        updated_data = [[] for i in range(self.n)]
        for i in range(len(self.data)):
            temp_distance = []
            for j in range(self.n):
                temp_distance.append(get_euclidean(self.data[i], self.cluster[j]['center']))
            updated_data[temp_distance.index(min(temp_distance))].append(data[i])

        for i in range(self.n):
            self.cluster[i]['data'] = updated_data[i]

        return self.cluster


    # 센터값을 모두 업데이트.
    def update_center(self):
        new_center = []
        for i in range(self.n):
            new_center.append(np.average(self.cluster[i]['data'], axis = 0))
        return new_center

    # update_center을 한 다음 clustering 다시 진행
    # prev_centers와 다르면 다시 진행. 같으면? 끝~
    """
    prev_centers를 채움
    -> clustering 진행
    -> update_center : new centers 리턴
    -> 만약 다르다? 다시 진행
    -> 같다? 끝!
    """
    def update(self):
        again = True
        while again:
            again = False
            # 여기서 clustering 다시 함.
            prev_center = []  # ndarray의 list
            for j in range(self.n):
                prev_center.append(self.cluster[j]['center'])

            self.cluster = self.clustering(self.cluster)
            new_center = self.update_center()
            # 같은 집합인지 비교
            for a in range(self.n):
                if((~np.equal(prev_center[a], new_center[a]).all())):
                    again = True

            for k in range(self.n):
                self.cluster[k]['center'] = new_center[k]

    # 외부에서 실행하는 부분
    def fit(self):
        self.init_center()
        self.cluster = self.clustering(self.cluster)

        # 여기서 계속 돌아갈 수 있게
        self.update()

        result, labels = self.get_result(self.cluster)
        draw_graph(result, labels)

    def get_result(self, cluster):
        result = []
        labels = []
        for key, value in cluster.items():
            for item in value['data']:
                labels.append(key)
                result.append(item)

        return np.array(result), labels


if __name__ == '__main__':
    data = read_data()
    data = norm(data)
    hierarchical(data)
    dbscan(data)
    model = KMeans(data, 8)
    model.fit()