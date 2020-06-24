import numpy as np
import copy

# n : score
# n을 기준으로 뽑은 5개의 random seed을 저장한 배열을 리턴하는 함수
def random_seed(n):
    return_seed = [] # random seed를 저장할 배열
    n = int(n * 10) # 정수로 변환
    np.random.seed(n) # n을 기준으로 random seed를 추출하겠다고 설정
    # 5개의 random seed를 추출
    for i in range(5):
        # return_seed에 random seed를 저장
        return_seed.append(np.random.random())
    return return_seed # random seed를 저장한 배열 리턴

# greedy search
def greedy_search(score):
    return_score = []
    # 배열을 초기화
    # [[], score] 형태로 만듦
    return_score.append([])
    # score 값을 저장
    return_score.append(score)
    # 10회 동안 점수를 계산
    for i in range(10):
        # tmp : 점수로 계산한 random_seed의 return값 저장
        tmp = random_seed(return_score[1])
        index = np.argmax(tmp) # index: 가장 점수가 높은 값의 index
        return_score[0].append(index) # 인덱스를 저장
        return_score[1] *= tmp[index] # 점수를 update
    print('greedy search')
    print(return_score) # 결과값 출력

# beam search
def beam_search(score):
    return_score = [] # 결과값을 저장할 배열
    rand = random_seed(score) # random seed 계산
    rand_ = sorted(rand) # 배열을
    # random seed 1, 2, 3 번째 점수들을 구하여 저장
    # score을 가지고 큰 seed 값 3개를 추출
    # range(5) : element가 5개 이기 때문
    for i in range(5):
        # index를 저장하는 배열
        tmp = []
        # 제일 큰 값, 두 번째로 큰 값, 세 번째로 큰 값
        if(rand[i] == rand_[-1] or rand[i] == rand_[-2] or rand[i] == rand_[-3]):
            # tmp에 배열을 저장
            tmp.append([])
            # index를 저장
            tmp[0].append(rand.index(rand[i]))
            # 점수 값을 저장
            tmp.append(rand[i])
            # tmp를 return_score에 저장
            return_score.append(tmp)
    # print(return_score)
    # 처음에 1번 계산했기 때문에 9번 반복
    for i in range(9):
        # 점수를 저장하는 배열
        scores = []
        # 계산한 15개의 항목을 저장
        # [[], index]가 15개 존재
        tmp_score = []
        # 15개의 값을 생성
        for j in range(3):
            # random seed를 계산하여 저장
            tmp_seed = random_seed(return_score[j][1])
            for k in range(5):
                # return_score에서 copy
                # tmp = [[], score]
                tmp = copy.deepcopy(return_score[j])
                # 점수 값을 계산
                tmp[1] = tmp[1] * tmp_seed[k]
                # tmp에 index를 저장
                tmp[0].append(k)
                # scores에 계산한 점수를 저장
                scores.append(tmp[1])
                # updata한 배열을 저장
                tmp_score.append(tmp)
        # 점수를 정렬
        scores = sorted(scores)
        # return_score 배열의 모든 요소 삭제
        return_score.clear()
        # return_score 배열 update
        for j in range(15):
            # 배열의 크기가 3개이면 중지
            if(len(return_score) == 3):
                break
            # 점수가 상위 3개의 값이면 return_score에 저장
            if(tmp_score[j][1] == scores[-1] or tmp_score[j][1] == scores[-2] or tmp_score[j][1] == scores[-3]):
                return_score.append(tmp_score[j])
    print('beam search')
    # 출력
    for i in range(3):
        print(return_score[i])
    

if __name__ == '__main__':
    # score
    score = 1
    # greedy search
    greedy_search(score)
    # beam search
    beam_search(score)
