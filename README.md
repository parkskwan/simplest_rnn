# simplest_rnn

```
이해하기 쉽게 NAVER(네이버)의 실제 적용 사례를 예로 들어 RNN(LSTM)를 설명 하겠습니다. 그리고 프로그램은 세상에서 가장 단순한 예제로 
sequnce-to-sequence (many-to-many) 그리고 sequence to value (many-to-one)를 설명 할 것이므로 겁먹지 말고 천천히 따라 하시면 
RNN(LSTM)의 본질을 이해 하실 수 있습니다. 잊지 마세요, 새로운 지식을 얻기 위해서는 노력이 필요하다는 것을

나는 충분히 LSTM의 필요성을 알고 있다고 생각하시는 독자께서는 시계열 예측으로 이동 하셔도 됩니다. 

Jupyter Notebook으로 결과가 같이 출력된 화일로 다시 올려 드릴 테니 걱정하지 마시고 그냥 쭉 읽어 보세요. 
```

# 개요

### NAVER(네이버) 서비스엔 어떤 AI 녹아 있나

하루에도 몇 번씩 방문하게 되는 네이버 서비스 안에는 어떤 인공지능(AI) 기술이 녹아 있을까? 그동안 네이버 AI 서비스 중 가장 주목을 받은 것은 번역앱인 파파고였다. 하지만 네이버 측은 특정 서비스 자체가 중요한 게 아니란 입장이다. 서비스 전반에 걸쳐 AI를 활용하기 위해 노력 중이라고 네이버 관계자가 설명했다. 이제 AI는 거창한 어떤 것이라기보다는 일상 생활에서 직접 사용될 때 더 큰 의미를 갖는 기술이 됐다. 네이버는 검색과 쇼핑 전반에 걸쳐 AI 기반 기술을 녹여내는 작업을 진행 중이다.

#### ■ 원하는 이미지 검색 돕는 '쿼리 어시스턴트'

네이버 검색 공식 블로그에 따르면 우선 네이버 모바일앱을 통한 이미지 검색에는 '쿼리 어시스턴트'라는 기술이 숨어있다. 예를 들어 네이버 모바일앱에서 '식탁'을 키워드로 이미지 검색을 하면 검색어 입력창 바로 아랫쪽에 '2인용', '4인용', '6인용', '리폼', '매트'라는 키워드가 관련 이미지와 함께 뜬다. 

이러한 키워드를 표시하기 위해 네이버는 AI 기반 기술 중 하나인 딥러닝 기법을 썼다. 우선 검색어나 검색어와 관련된 위치 중심으로 노출되는 이미지 목록을 모은다. 그 뒤 이들 중 시각적으로 유사한 것들만 따로 모아 분류하는 '클러스터링'이라는 작업을 수행한다.

다음으로는 딥러닝 방법을 통해 분류된 이미지들을 비지도학습 방법으로 분석한다. 이미지들 간 유사성이 어느 정도인지에 따라 정보를 다시 묶어주고, 관련 텍스트를 이용해 적절한 키워드를 정확도에 따라 추천한다.

#### ■ 여행지에서 뉴스까지 맞춤 정보 추천하는 AI

또 다른 기술로는 해외 도시 관련 정보 추천서비스에 적용된 코나(ConA)와 사용자 맞춤 뉴스 추천 서비스인 에어스(AiRS)를 들 수 있다.

네이버에 따르면 맥락인식 AI(Context recognition AI)의 줄임말인 코나는 고베, 발리, 시카고, 칭다오, 마카오, 오코하마, 밀라노 등을 포함해 76개 도시, 1만3천여개 명소, 맛집, 쇼핑센터 등 관심장소(POI)를 대상으로 도시정보, 리뷰, 가볼만한 곳, 코스추천 등 정보를 제공하고 있다.

관련 장소에 대해 사용자에게 필요한 정보를 제공하기 위해 AI 기술 중 회선신경망(CNN), 장단기기억기술(LSTM)을 썼다. 사용자가 블로그, 지식iN 등을 통해 올린 여행지, 식당 등에 대한 콘텐츠(UGC) 빅데이터를 모아 해당 장소에 가는 목적이나 분위기 등과 같은 테마를 자동으로 추출해낸다. 그 뒤에는 인기 테마를 뽑아내기 위해 각종 UGC에서 CNN과 LSTM을 활용, 추출된 후보 테마가 들어간 문장의 문맥을 분석하고 여행지별 적합한 테마를 서로 연결시키는 태깅 작업을 한다.

CNN은 인간의 시신경이 사물을 받아들이는 방식을 차용한 AI 기술로 코나에서는 각 여행지에 대한 UGC 내에 리뷰 문장 중 사전에 정해진 후보 테마 키워드가 등장하지 않지만 문맥 상 해당 의미를 포함한 테마를 뽑아낸다. 예를 들어 '오드리햅번이 영화 로마의 휴일을 찍은 스페인광장'이라는 말에서 '촬영지'라는 테마를 추출해 해당 장소와 태깅한다. 이를 통해 '로마 스페인 광장'의 대표 키워드인 '분수', '젤라또', '아이스크림', '나들이', '촬영지'를 자동으로 추출해 네이버 모바일앱에서 여행지에 대한 보다 유용한 정보를 얻을 수 있게 돕는다.

LSTM은 특정 위치에 나타나는 단어의 종류를 인식한다. 사용자에 따라 다양하게 표현되는 장소별 후보 테마 키워드를 풍부하게 하는 역할을 맡았다.

CNN과 LSTM이 상호작용해 여행지별 테마를 태깅하고, 태깅된 데이터는 다양한 테마의 여행정보를 찾는 사용자들에게 맞춤형 여행정보를 묶어서 보여준다.



# 시계열 예측이란?

```
RNN이 기존 신경망과 가장 다른 점은 순서열을 처리할 수 있다는 점이다. RNN에 입력 벡터 순서열 x1,x2,…,xn 을 차례대로 입력하면 상태 순서열 
s1,s2,…,sn이 내부적으로 생성되고 출력으로는 출력 순서열 o1,o2,…,on 이 나온다.

만약 원하는 결과가 출력 순서열 o1,o2,…,on 이 target 순서열 y1,y2,…,yn 과 같아지는 것이라면 입력 순서열 길이와 출력 순서열 길이가 같은 
특수한 경우의 sequnce-to-sequence (many-to-many) 예측 문제가 되고 순서열의 마지막 출력 on 값이 yn값과 같아지는 것만 목표라면 단순한 
sequence to value (many-to-one) 문제가 된다.
```

# Keras를 사용한 RNN 구현

```
개인적으로는 RNN(LSTM)의 경우는 C++로 구현해서 사용하는 것을 추천한다. 그러나 CNN는 개인적으로 만들어 사용하는 것을 추천하지는 않는다. 
이유는 알파고 때문인지는 모르겠으나 CNN은 굉장히 신경써서 만들고 테스트한 것이 분명한 것 같다. 그러나 RNN(LSTM)은 상대적으로 그렇지 
못한 것 같다. 이유는 시계열을 충분히 딥하게 하면 제대로 학습이 되지 않는 경우가 많기 때문 입니다. 개인적인 생각입니다.
```

이제 부터는 모두가 좋아하는 파이썬용 신경망 패키지인 Keras를 사용해서 RNN을 구현 하겠습니다. Keras는 tensorflow 를 사용하여 신경망을 구현해 줄 수 있도록 하는 고수준 라이브러리다.

Keras 는 다양한 형태의 신경망 구조를 블럭 형태로 제공하고 있으며 SimpleRNN, LSTM, GRU 와 같은 RNN 구조도 제공한다. 



Keras 에서 RNN 을 사용하려면 입력 데이터는 (nb_samples, timesteps, input_dim) 크기를 가지는 ndim=3인 3차원 텐서(tensor) 형태이어야 한다.

```
    nb_samples: tngth, tlngth (자료의 수)
     timesteps: window = sgmt (순서열의 길이)
     input_dim: x 벡터의 크기 (시계열 데이터의 종류, 지금 예제의 경우는 1)
```

여기에서는 단일 시계열이므로 input_dim = 1 이고 스텝 크기의 순서열을 사용하므로 timesteps = window.



### 시계열 예측 문제

풀어야 할 문제는 다음과 같은 시계열을 입력으로 다음 스텝의 출력을 예측하는 단순한 시계열 예측 문제이다. 

##### ■ 헤더 부분

```
%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
```

##### ■ 시계열 전체 데이터 만들고 확인 하기 위해 출력 및 plot 하기 

```
lngth = 50; sgmt = 3;

t = np.linspace(0, 10*np.pi, lngth)
s = .04 * t * np.cos(2 * 0.125 * t)

#print('t = ', t); print('s = ', s)

plt.figure(figsize=(10, 5), frameon=False)
plt.plot(t, s, 'ro-')
plt.title("Plot of the Time Series Input Dataset")
plt.show()
```

##### ■ sequnce-to-value (many-to-one) 데이터 만들기

```
window = sgmt; tngth = lngth - window;

x_train = np.zeros((tngth, window,1))
y_train = np.zeros((tngth))

#print(x_train.shape); print(y_train.shape)

I, J, K = x_train.shape

for i in range(I):
    for j in range(J):
        x_train[i,j,0] = s[i+j]
    y_train[i] = s[i+window]

```

##### ■ sequnce-to-value (many-to-one) 데이터 plot 하기

여기에서는 0 <= i < tngth 까지 변화 시켜 plot 해 가면서 rnn(lstm)이 어떤 형식의 입출력 자료를 사용하는지 충분히 생각 해야 합니다.

```
plt.figure(figsize=(10, 8), frameon=False)    

plt.subplot(311)
plt.plot(t, s, 'c-')

i = 0
plt.plot(t[i:window+i], x_train[i].flatten(), 'bo-', label="train_input sequence")
plt.plot(t[window], y_train[i], 'ro', label="target")
plt.legend()
plt.title("First sample sequence")


plt.subplot(312)
plt.plot(t, s, 'c-')


i = 1
plt.plot(t[i:window+i], x_train[i].flatten(), 'bo-', label="train_input sequence")
plt.plot(t[window+i], y_train[i], 'ro', label="target")
plt.legend()
plt.title("Second sample sequence")


plt.subplot(313)
plt.plot(t, s, 'c-')

i = tngth-1
plt.plot(t[i:window+i], x_train[i].flatten(), 'bo-', label="train_input sequence")
plt.plot(t[window+i], y_train[i], 'ro', label="target")
plt.legend()
plt.title("Third sample sequence")

plt.tight_layout()
plt.show()   
```
##### ■ Simple RNN MLP 만들기

```
np.random.seed(0)
model = Sequential()

model.add(SimpleRNN(tngth, input_shape= (sgmt,1)))

model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
```

##### ■ Simple RNN 계산 그리고 history loss plot
```
history = model.fit(x_train, y_train, epochs=100, verbose=0)

plt.figure(figsize=(10, 10), frameon=False)
plt.plot(history.history["loss"])
plt.title("Loss")
plt.show()
```

##### ■ Simple RNN 으로 훈련된 자료를 이용하여 예측결과 plot 하기

```
plt.figure(figsize=(15, 10), frameon=False)
plt.plot(t,s,'co-', label="origin")
plt.plot(t[sgmt:], y_train, 'ro-', label="target")
plt.plot(t[sgmt:], model.predict(x_train[:]), 'bs-', label="output")

plt.legend()
plt.title("After training")
plt.show()
```


##### ■ sequnce-to-sequence (many-to-many) 데이터 만들기

입력(X_train) 순서열과 출력(Y_train) 순서열의 크기는 같아야 한다는 것만 주의 하면됩니다.

```
window = sgmt; tngth = lngth - window; tlngth = tngth - window + 1

X_train = np.zeros((tlngth, window,1))
Y_train = np.zeros((tlngth, window,1))

print(X_train.shape); print(Y_train.shape)

I, J, K = X_train.shape

for i in range(I):
    for j in range(J):
        X_train[i,j,0] = s[i+j]
        Y_train[i,j,0] = s[i+window+j]

```

##### ■ sequnce-to-sequence (many-to-many) 데이터 plot 하기

여기도 역시 0<= i < tlngth 변화 시켜 plot 해 가면서 rnn(lstm)이 어떤 형식의 입출력 자료를 사용하는지 충분히 생각 해야 합니다.

```
plt.figure(figsize=(10, 8), frameon=False)    

plt.subplot(311)
plt.plot(t, s, 'c-')

i = 0;
plt.plot(t[i:window+i], X_train[i].flatten(), 'bo-', label="train_input sequence")
plt.plot(t[window+i:2*window+i], Y_train[i].flatten(), 'ro-', label="target")
plt.legend()
plt.title("First sample sequence")


plt.subplot(312)
plt.plot(t, s, 'c-')

i = 1;
plt.plot(t[i:window+i], X_train[i].flatten(), 'bo-', label="train_input sequence")
plt.plot(t[window+i:2*window+i], Y_train[i].flatten(), 'ro-', label="target")
plt.legend()
plt.title("Second sample sequence")


plt.subplot(313)
plt.plot(t, s, 'c-')

i = tlngth-1;
plt.plot(t[i:window+i], X_train[i].flatten(), 'bo-', label="train_input sequence")
plt.plot(t[window+i:2*window+i], Y_train[i].flatten(), 'ro-', label="target")
plt.legend()
plt.title("Third sample sequence")

plt.tight_layout()
plt.show()   
```

##### ■ Seq2Seq Simple RNN MLP 만들기

SimpleRNN 클래스 생성시 return_sequences 인수를 True로 하면 출력 순서열 중 마지막 값만 출력하는 것이 아니라 전체 순서열을 3차원 텐서 형태로 출력하므로 sequence-to-sequence 문제로 풀 수 있습니다. 

다만 이 경우에는 다음에 오는 Dense 클래스 객체를 TimeDistributed wrapper를 사용하여 3차원 텐서 입력을 받을 수 있게 확장해 주어야 합니다.

```
from keras.layers import TimeDistributed

model2 = Sequential()
model2.add(SimpleRNN(tlngth, return_sequences=True, input_shape=(window, 1)))
model2.add(TimeDistributed(Dense(1)))
model2.compile(loss='mse', optimizer='adam')
```

##### ■ Seq2Seq Simple RNN 계산 그리고 history loss plot
```
history2 = model2.fit(X_train, Y_train, epochs=100, verbose=0)

plt.figure(figsize=(10, 10), frameon=False)
plt.plot(history2.history["loss"])
plt.title("Loss")
plt.show()
```
##### ■ Seq2Seq Simple RNN 으로 훈련된 자료를 이용하여 예측결과 plot 하기

```
plt.figure(figsize=(10, 8), frameon=False)    

plt.subplot(311)
#plt.plot(t, s, 'c-')

i = 0;
plt.plot(t[i:window+i], X_train[i].flatten(), 'bo-', label="train_input sequence")
plt.plot(t[window+i:2*window+i], Y_train[i].flatten(), 'ro-', label="target")
plt.plot(t[window+i:2*window+i], model2.predict(X_train[i:i+1,:,:]).flatten(), 'gs-', label="output sequence")
plt.legend()
plt.title("First sample sequence")


plt.subplot(312)
#plt.plot(t, s, 'c-')

i = 1;
plt.plot(t[i:window+i], X_train[i].flatten(), 'bo-', label="train_input sequence")
plt.plot(t[window+i:2*window+i], Y_train[i].flatten(), 'ro-', label="target")
plt.plot(t[window+i:2*window+i], model2.predict(X_train[i:i+1,:,:]).flatten(), 'gs-', label="output sequence")
plt.legend()
plt.title("Second sample sequence")


plt.subplot(313)
#plt.plot(t, s, 'c-')

i = tlngth-1; 
plt.plot(t[i:window+i], X_train[i].flatten(), 'bo-', label="train_input sequence")
plt.plot(t[window+i:2*window+i], Y_train[i].flatten(), 'ro-', label="target")
plt.plot(t[window+i:2*window+i], model2.predict(X_train[i:i+1,:,:]).flatten(), 'gs-', label="output sequence")
plt.legend()
plt.title("Third sample sequence")

plt.tight_layout()
plt.show()   
```




작성중.....
