import numpy as np
import matplotlib.pyplot as plt
import codecs



data_list = []

with codecs.open('seoul.txt', 'r', 'utf-8') as seoul:
    first_line = seoul.readline() # 첫번째 줄
    lines = seoul.readlines() #두번째 줄~마지막 줄
    for line in lines:
        line = line.rstrip() # 개행문자 제거
        data_list.append(line.split('\t')) #한줄씩 split 되어서 들어감

total = [] # 계
men = [] # 남자
women = [] #여자

data_np = np.array(data_list)
data_np = np.delete(data_np, [0,1], 1) #계싼을 위해 삭제 1: 열을 삭제, 0: 행을 삭제

temp = data_np.tolist() #python 리스트로 다시 변환.


# 경우 별로 나눔
for i in range(len(temp)):
    if i % 3 == 0: # 계일 때
        total.append(temp[i])
    elif i % 3 == 1: # 남자일 때
        men.append(temp[i])
    else: # 여자 일 때
        women.append(temp[i])

# 전치 행렬로 변환
totalT = [[element for element in t] for t in zip(*total)]
menT = [[element for element in t] for t in zip(*men)]
womenT = [[element for element in t] for t in zip(*women)]

total = []
men = []
women = []

# 합을 추가
for col in totalT:
    col = [int(x) for x in col]
    total.append(sum(col))

for col in menT:
    col = [int(x) for x in col]
    men.append(sum(col))

for col in womenT:
    col = [int(x) for x in col]
    women.append(sum(col))

#계 총합 평균 분산 출력
print('계 : ', end='')
for i in range(len(total)):
    print(total[i], '\t', end='')
print('\n계 총합 : ', sum(total))
print('계 평균 : ', int(np.mean(total)))
print('계 분산 : ', int(np.var(total)), '\n')

#남자 총합 평균 분산 출력
print('남자 : ', end='')
for i in range(len(men)):
    print(men[i], '\t', end='')
print('\n남자 총합 : ', sum(men))
print('남자 평균 : ', int(np.mean(men)))
print('남자 분산 : ', int(np.var(men)), '\n')

#여자 총합 평균 분산 출력
print('여자 : ', end='')
for i in range(len(women)):
    print(women[i], '\t', end='')
print('\n여자 총합 : ', sum(women))
print('여자 평균 : ', int(np.mean(women)))
print('여자 분산 : ', int(np.var(women)), '\n')

#계 그래프 출력
plt.bar(range(101), total)
plt.show()

#남자 그래프 출력
plt.bar(range(101), men)
plt.show()

#여자 그래프 출력
plt.bar(range(101), women)
plt.show()