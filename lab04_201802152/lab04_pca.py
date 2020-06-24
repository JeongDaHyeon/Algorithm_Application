import codecs
import copy
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# 데이터를 읽는 함수
def read_data():
    data = []
    with codecs.open('seoul_student.txt', 'r', 'utf-8') as student:
        first_line = student.readline()
        lines = student.readlines()

        for line in lines:
            line = line.rstrip()
            data.append(line.split('\t'))

    data = np.array(data)
    data = np.array(data, dtype=np.float) # 데이터를 실수형으로 변환
    return data

# 정규화를 하는 함수
def norm(data):
    scaler = MinMaxScaler()
    data[:] = scaler.fit_transform(data[:])
    return data

def draw_graph(data):
    plt.figure()
    plt.scatter(data[:,0], data[:,1], cmap='rainbow')
    plt.show()


def draw_pca_graph(data):
    plt.figure()
    plt.scatter(data, [0] * len(data), cmap='rainbow')
    plt.yticks(np.arange(-0.008, 0.008, 0.002))
    plt.show()
    

# 라이브 러리를 이용하여 그래프를 그리는 함수
def sklearn_pca(data, dim):
    pca = PCA(n_components=dim)
    data = pca.fit_transform(data)
    data = data.flatten()
    draw_pca_graph(data)
    print("sklearn")


# 키의 평균값, 몸무게의 평균값 리턴
def get_expected_value(data):
    height = 0.0
    weight = 0.0

    for i in range(len(data)):
        height += data[i][0]
        weight += data[i][1]

    height /= len(data)
    weight /= len(data)

    return height, weight

# ex: 키 값의 평균
# ey: 몸무게 값의 평균
def get_covariance_matrix(x, y, ex, ey):
    print('covariance matrix')

# 정렬하는 함수
def e_sort(val, vec, dim):
    eig_pairs = [(np.abs(val[i]), vec[:, i]) for i in range(len(vec))]
    eig_pairs.sort(key=lambda x: x[0], reverse= True)

    return_vector = []

    for i in range(dim):
        return_vector.append(eig_pairs[i])

    return return_vector



def pca(data, dim):
    mh, mw = get_expected_value(data)
    sub = [[mh] * len(data), [mw] * len(data)]
    sub = np.array(sub)
    data = np.subtract(data, sub.T)
    cov_data = np.divide(np.matmul(data.T, data), len(data)-1) # covariance matrix
    w, v = LA.eig(cov_data)
    sort_vec = e_sort(w, v, dim)

    product_mat = []
    for i in range (dim):
        product_mat.append((sort_vec[i][1]).tolist())

    product_mat = np.array(product_mat)
    data = np.matmul(data, product_mat.T)
    data = data.flatten()
    draw_pca_graph(data)



if __name__ == '__main__':
    reduce_dim = 1
    data = read_data()
    data = norm(data)
    draw_graph(data)
    sklearn_pca(copy.deepcopy(data), reduce_dim)
    pca(copy.deepcopy(data), reduce_dim)
