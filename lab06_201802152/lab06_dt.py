import pandas as pd
import pydot
from sklearn.tree import export_graphviz, DecisionTreeClassifier
import csv
from sklearn import tree


def readData(file_name):
    data = pd.read_csv(file_name)
    return data

# 데이터를 전처리 하는 함수
def process_data(data):
    # 같이 묶기 곤란한 항목 제거
    del data['PassengerId']
    del data['Name']
    del data['Cabin']
    del data['Ticket']

    # 숫자로 변경할 수 있는 항목은 수치화
    change_numerical_value(data, 'Sex')
    change_numerical_value(data, 'Embarked')
    # 평균값으로 데이터를 채워줌
    data = data.fillna(data.mean())
    # 변경한 데이터 리턴
    return data

# 항목을 수치화 해주는 것
# data: 데이터
# label: 수치화 할 항목
def change_numerical_value(data, label):
    data[label] = pd.get_dummies(data[label])

def fill_nan(data):
    data.fillna(data.mean())

# 그래프를 그리는 함수
# dt: Decision Tree 객체
# features: 항목
def draw_graph(dt, features):
    export_graphviz(dt, out_file="dt.dot", class_names=['No', 'Yes'], feature_names=features, impurity=False, filled=True)
    (graph, ) = pydot.graph_from_dot_file('dt.dot', encoding='utf-8')
    graph.write_png('dt.png')

# 데이터를 cvs 파일로 저장하는 함수
def save_data_cvs(data):
    with open('titanic.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['PassengerId', 'Survived'])
        index = 892
        for i in range(len(data)):
            tmp = []
            tmp.append(index)
            tmp.append(data[i])
            spamwriter.writerow(tmp)
            index += 1;

if __name__ == '__main__':
    # 데이터를 읽음
    train_data = readData('data/train.csv')
    test_data = readData('data/test.csv')
    # 데이터를 전처리
    train_data = process_data(train_data)
    test_data = process_data(test_data)
    test = test_data.copy()
    #print(train_data.isnull().sum())
    #print(test_data.isnull().sum())
    # features: 데이터의 항복
    features = test_data.columns.values
    X_train = train_data.drop("Survived", axis=1)
    Y_train = train_data["Survived"]
    decision_tree = DecisionTreeClassifier()
    #Build a decision tree classifier from the training set (X_train, Y_train)
    decision_tree.fit(X_train, Y_train)
    # test_data로 예상값 Y_pred 추측
    Y_pred = decision_tree.predict(test_data)
    dt = tree.DecisionTreeClassifier()
    dt.fit(test, Y_pred)
    draw_graph(dt, features)# 그래프를 출력
    save_data_cvs(Y_pred) # 데이터를 저장
"""
    with open('titanic.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['PassengerId','Survived'])
        index = 892
        for i in range(len(Y_pred)):
            tmp = []
            tmp.append(index)
            tmp.append(Y_pred[i])
            spamwriter.writerow(tmp)
            index += 1;
"""

