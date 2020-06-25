import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.fftpack import dct
import warnings

# 시간축 상의 data
# 어떤 frequency가 있는지 분석
# time domain -> frequency domain
# 음향에서 nosing 처리에 사용할 수 있음


# WARNING들을 무시
warnings.filterwarnings("ignore")


# 정현파 조합
# sample의 개수
N = 1024
# 1초에 44100개의 측정
# 1 / 44100 s 간격으로 측정을 함
T = 1.0 / 44100.0
# 697 HZ
f1 = 697
# 1209 HZ
f2 = 1209
# 0과 N*T 사이에 균일한 간격을 N개의 구간으로 동등하게 분할
t = np.linspace(0.0, N*T, N)
# sine 함수
y1 = 1.1 * np.sin(2 * np.pi * f1 * t)
y2 = 0.9 * np.sin(2 * np.pi * f2 * t)
y = y1 + y2 # 두 개의 주파수를 합침

plt.subplot(311)
plt.plot(t, y1)
plt.title(r"$1.1\cdot\sin(2\pi\cdot 697t)$")

plt.subplot(312)
plt.plot(t, y2)
plt.title(r"$0.9\cdot\sin(2\pi\cdot 1209t)$")
# 두 개의 주파수를 합친 그래프
plt.subplot(313)
plt.plot(t, y)
plt.title(r"$1.1\cdot\sin(2\pi\cdot 697t) + 0.9\cdot\sin(2\pi\cdot 1209t)$")
plt.tight_layout()
plt.show() # 그래프를 출력


# 고속 푸리에 변환
y2 = np.hstack([y, y, y])

plt.subplot(211)
plt.plot(y2)
plt.axvspan(N, N * 2, alpha=0.3, color='green') # 특정 구간을 강조
plt.xlim(0, 3 * N)

plt.subplot(212)
plt.plot(y2)
plt.axvspan(N, N * 2, alpha=0.3, color='green') # 특정 구간을 강조
plt.xlim(900, 1270) # x축 제한

plt.show()


# 신호에 담겨 있는 주파수를 분석
yf = fft(y, N) # 고속 푸리에 변환
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

plt.stem(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.xlim(0, 3000)

plt.show()



# DCT (Discrete Cosine Transform) : 결과물이 실수로 출력
# DFT (Discrete Fourier Transform) : 결과물이 복소수로 출력
dct_type = 2
yf2 = dct(y, dct_type, N)


# DFT 실수부, 허수부 출력
# 실수부 출력
plt.subplot(311)
plt.stem(np.real(yf))
plt.title("DFT REAL NUM")

# 허수부 출력
plt.subplot(312)
plt.stem(np.imag(yf))
plt.title("DFT IMAGINARY NUM")

# DCT 결과 출력
plt.subplot(313)
plt.stem(np.abs(yf2))
plt.title("DCT")

plt.tight_layout()
plt.show()
