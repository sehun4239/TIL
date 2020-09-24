## 특정 예제를 통해서 Linear Regression을 적용해보자

* 날씨 예보로 온도에 따른 오존량을 예측해보자

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from my_library.machine_learning_library import numerical_derivative

def numerical_derivative(f,x):
    # f : 미분하려고 하는 다변수 함수
    # x : 모든 변수를 포함하고 있는 ndarray (차원 상관없이)
    
    delta_x = 1e-4
    derivative_x = np.zeros_like(x)  # 미분한 결과를 저장하는 ndarray
    
    # iterator를 이용해서 입력변수 x에 대해 편미분을 수행
    it = np.nditer(x, flags=['multi_index'])
    
    while not it.finished:
        idx = it.multi_index    # iterator의 현재 index를 추출 (tuple)
        
        # 현재 칸의 값을 어딘가에 잠시 저장해야한다.
        tmp = x[idx]
        
        x[idx] = tmp + delta_x
        fx_plus_delta = f(x)   # f(x + delta_x)
        
        x[idx] = tmp - delta_x
        fx_minus_delta = f(x)   # f(x - delta_x)
        
        derivative_x[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta_x)
        
        x[idx] = tmp     # 데이터를 원상 복구
        
        it.iternext()
        
    return derivative_x

# 1. Raw Data Loading
df = pd.read_csv('./data/ozone.csv')
display(df)

# 2. Data Preprocessing(데이터 전처리)
#    - 결측치 처리 => 삭제 ! (결측치가 10% 미만이면 삭제해도 무방)
#                  - 10% 이상이면 값을 변경 (평균, 최대, 최소등)
#    - 이상치 처리(outlier)
#      - 이상치를 검출하고 변경하는 작업
#    - 데이터 정규화 작업
#    - 학습에 필요한 컬럼을 추출, 새로 생성.

# 필요한 column(Temp, Ozone)만 추출해서 결측값 제거

training_data = df[['Temp','Ozone']]
print(training_data.shape)  # (153, 2)

training_data = training_data.dropna(how='any')
print(training_data.shape)  # (116, 2)

# 3. Training Data Set
x_data = training_data['Temp'].values.reshape(-1,1)
t_data = training_data['Ozone'].values.reshape(-1,1)

# 4. 지금 우리는 Simple Linear Regression
#    y = Wx + b => 우리가 구해야 하는 W.B를 정의
W = np.random.rand(1,1)
b = np.random.rand(1)

# 5. loss function 정의
def loss_func(x,t):
    
    y = np.dot(x,W) + b
    
    return np.mean(np.power((t-y),2))   # 최소제곱법에 대한 코드

# 6. 학습종료 후 예측값을 알아오는 함수
def predict(x):
    return np.dot(x,W) + b

# 7. 프로그램에서 필요한 변수들 정의
learning_rate = 1e-5
f = lambda x : loss_func(x_data,t_data)

# 8. 학습을 진행 !
for step in range(30000):
    
    W -= learning_rate * numerical_derivative(f, W)
    b -= learning_rate * numerical_derivative(f, b)
    
    if step % 3000 == 0 :
        print('W: {}, b: {}, loss: {}'.format(W,b,loss_func(x_data, t_data)))
        
# 9. 그래프로 확인해보아요

plt.scatter(x_data,t_data)
plt.plot(x_data, np.dot(x_data,W) + b, color='r')
plt.show()
```

* 위의 방법보다 sklearn을 이용하면 훨씬 효율이 좋게 예측이 가능하다.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# 1. Raw Data Loading
df = pd.read_csv('./data/ozone.csv')

# 2. Data Preprocessing(데이터 전처리)

training_data = df[['Temp','Ozone']]
print(training_data.shape)  # (153, 2)

training_data = training_data.dropna(how='any')
print(training_data.shape)  # (116, 2)

# 3. Training Data Set
x_data = training_data['Temp'].values.reshape(-1,1)
t_data = training_data['Ozone'].values.reshape(-1,1)

# 4. sklearn을 이용해서 linear regression model 객체를 생성
model = linear_model.LinearRegression()

# 5. Training Data Set을 이용해서 학습을 진행!
model.fit(x_data,t_data)
    
# 6. W와 b값을 알아내야 한다.
print('W:{}, b:{}'.format(model.coef_,model.intercept_))

# 7. 그래프로 확인
plt.scatter(x_data,t_data)
plt.plot(x_data,np.dot(x_data,model.coef_) + model.intercept_, color='r')
plt.show()

# 8. 예측
predict_val = model.predict([[62]])
print(predict_val)
```



### 차이가 발생하는 이유 ?

* 데이터 전처리가 잘 안돼서 그렇다.

```python
# 학습이 잘 되기 위해서는 데이터의 전처리가 필수다 => 2가지 정도 얘기해보자
# 결측치 처리: 일단 여기서는 삭제처리로 끝낼 것이다.
# 이상치처리와 데이터의 정규화를 해보자

# 이상치 처리(Outlier)
# - Z-score (분산, 표준편차를 이용하는 검출방식 - 통계기반)
# - Turkey Outlier (4분위 값을 이용하는 이상치 검출방식)
# 이상치(Outlier)는 속성의 값이 일반적인 값보다 편차가 큰 값을 의미
# 즉, 데이터 전체 패턴에서 동떨어져 있는 관측치를 지칭
# 평균뿐아니라 분산에도 영향을 미치기 때문에 결국은 데이터 전체의 안정성을 저해함
# 그래서 이상치는 반드시 처리해야 하고 이것을 검출하고 처리하는데
# 상당히 많은 시간이 소요되는게 일반적이다.

# 독립변수(온도)에 있는 이상치를 지대점이라고 하고
# 종속변수(오존량)에 있는 이상치를 outlier라고 한다.

# Turkey outlier를 이용해서 처리해보자
# boxplot이라는걸 이용해서 확인해보자

# boxplot을 사용할 때 이상치를 분류하는 기준은 IQR value를 사용
# IQR value = 3사분위값 - 1사분위값
# 1사분위값 - 1.5 IQR 보다 작으면 outlier
# 3사분위값 + 1.5  IQR 보다 크면 outlier

import numpy as np
import matplotlib.pyplot as plt

data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,22.1])

fig = plt.figure() # 새로운 그림(figure)을 생성

fig_1 = fig.add_subplot(1,2,1)  # 1행 2열의 subplot의 위치가 1번 위치
fig_2 = fig.add_subplot(1,2,2)  # 1행 2열의 subplot의 위치가 2번 위치

fig_1.set_title('Original Data Boxplot')
fig_1.boxplot(data)

# numpy로 사분위수를 구하려면 percentile() 함수를 이용한다.
print(np.mean(data))  # 평균
print(np.median(data))  # 중위수, 2사분위, 미디언
print(np.percentile(data,25)) # 1사분위
print(np.percentile(data,50)) # 2사분위
print(np.percentile(data,75)) # 3사분위

# 이상치를 검출하려면 IQR value 필요
IQR_val = np.percentile(data,75) - np.percentile(data,25)

upper_fense = np.percentile(data,75) + 1.5 * IQR_val
lower_fense = np.percentile(data,25) - 1.5 * IQR_val

print('upper_fense : {}'.format(upper_fense))
print('lower_fense : {}'.format(lower_fense))

# 데이터중에 이상치를 출력하고 이상치를 제거한 데이터로 boxplot을 그려보세요
outlier = data[(data > upper_fense) | (data < lower_fense)]
print(outlier)

fig_2.set_title('Remove Outlier BoxPlot')
fig_2.boxplot(data[(data <= upper_fense) & (data >= lower_fense)])
plt.show()
```

* 이것 역시 scipy를 이용하면 훨씬 쉽게 작성 가능하다.

```python
from scipy import stats

data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,22.1])

zscore_threshold = 2 # zscore outlier 임계값 ( 일반적 2)

outliers = data[np.abs(stats.zscore(data)) > zscore_threshold]
print(outliers)

data = data[np.isin(data, outliers, invert=True)]
print(data)
```

* 이상치 처리 외에도 정규화를 통해 조정해줘야한다.

```python
# 데이터가 가진 feature들의 scale이 심하게 차이나는 경우 이를
# 조정해 줘야 한다 => 정규화 (Normalization)

# 집에 대한 feature가 2개가 있다.
# 하나는 방의 개수 ( 1, 2, 3, 4, 5, .... 20) 1 ~ 20 숫자차이는 크지 않지만
# 집의 연식(월) (12,24,.....,240)     12~240 숫자 차이가 크다

# 각 feature들에 대해 동일한 scale을 적용할 필요가 있다. (0~1사이로)

# Min-Max Normalization (정규화)
# Standardization - Z-Score Normalization (표준화)

# Min-Max Normalization
# 데이터를 정규화 하는 가장 일반적인 방법 => 모든 feature들에 대해 최소 0 ~ 최대 1로 scaling (변환)

# Z-Score Normalization(Standardization)
# 평균과 분산을 이용하여 scaling => 이상치에 덜 민감, 하지만 모든 feature에 대해 동일한 척도 x
```

* 이 둘을 적용시키면 아까의 코드는 다음과 같이 변한다.

```python
# 1. Raw Data Loading
df = pd.read_csv('./data/ozone.csv')

training_data = df[['Temp','Ozone']]
print(training_data.shape)  # (153, 2)

training_data = training_data.dropna(how='any')
print(training_data.shape)  # (116, 2)

# 이상치 처리
zscore_threshold = 2.0
# outlier를 출력
# Temp에 대한 이상치(지대점)을 확인
outliers = training_data['Temp'][np.abs(stats.zscore(training_data['Temp'])) > zscore_threshold]

training_data = training_data.loc[~training_data['Temp'].isin(outliers)]
print(training_data.shape)

# 정규화 처리 ! sklearn을 이용
# 독립변수와 종속변수의 scaler 객체를 각각 생성

scaler_x = MinMaxScaler()  # MinMaxScaler 클래스의 객체를 생성
scaler_t = MinMaxScaler()  # MinMaxScaler 클래스의 객체를 생성

scaler_x.fit(training_data['Temp'].values.reshape(-1,1))
scaler_t.fit(training_data['Ozone'].values.reshape(-1,1))

training_data['Temp'] = scaler_x.transform(training_data['Temp'].values.reshape(-1,1))
training_data['Ozone'] = scaler_t.transform(training_data['Ozone'].values.reshape(-1,1))

# print(training_data['Temp'].values)
# print(training_data['Ozone'].values)

# Training Data Set
x_data = training_data['Temp'].values.reshape(-1,1)
t_data = training_data['Ozone'].values.reshape(-1,1)

# Weight & bias
W = np.random.rand(1,1)
b = np.random.rand(1)

# loss function
def loss_func(x,t):
    y = np.dot(x,W) + b
    return np.mean(np.power((t-y),2))

def predict(x):
    
    return np.dot(x,W) + b

# learning_rate
learning_rate = 1e-4
f = lambda x : loss_func(x_data,t_data)

# 학습

for step in range(900000):
    W -= learning_rate * numerical_derivative(f,W)
    b -= learning_rate * numerical_derivative(f,b)
    
    if step % 300000 == 0:
        print('W:{}, b:{}, loss:{}'.format(W,b,loss_func(x_data,t_data)))
        
print(predict(62))
```

* 원하는 값을 알아보기 위해서는 원래 scale로 돌리는 것이 필요하다.

```python
predict_data = np.array([62])
scaled_predict_data = scaler_x.transform(predict_data.reshape(-1,1))
print(scaled_predict_data)

scaled_result = predict(scaled_predict_data)
result = scaler_t.inverse_transform(scaled_result)
print(result)
```





## Tensorflow

* Tensorflow를 이용한 linear Regression을 알아보자
* Tensorflow를 설치 (1.x, 2.x 버젼이 있다. 공부때는 1.x 딥러닝때 2.x 적용하자)
  * pip install tensorflow==1.15

```python
import tensorflow as tf

print(tf.__version__)

node = tf.constant('Hello World') # Node를 생성했다.

# 우리가 만든 Graph를 실행하기 위해서 Session이 필요
sess = tf.Session()

# runner인 session이 생성되었으니 이걸 이용해서 node를 실행해보자
print(sess.run(node))

# node를 2개 만들자
node1 = tf.constant(10, dtype=tf.float32)
node2 = tf.constant(20, dtype=tf.float32)

node3 = node1 + node2

sess = tf.Session()
print(sess.run([node3,node2]))

# placeholder를 이용
# 2개의 수를 입력으로 받아서 덧셈연산을 수행

import tensorflow as tf

node1 = tf.placeholder(dtype=tf.float32)      # scalar 형태의 값 1개를 실수로 받아드리는 placeholder

node2 = tf.placeholder(dtype=tf.float32)

node3 = node1 + node2

sess = tf.Session()

sess.run(node3, feed_dict={ node1 : 20, node2 : 40})
```

* Linear Regression에 적용하자

```python
import tensorflow as tf

# 1. Raw Data Loading
# 2. Data Preprocessing
# 3. Training Data Set
x_data = [2,4,5,7,10]
t_data = [7,11,13,17,23]

# 4. Weight & bias
W = tf.Variable(tf.random.normal([1]), name='weight') # W = np.random.rand(1,1)
b = tf.Variable(tf.random.normal([1]), name='bias')   # b = np.random.rand(1)

# 5. Hypothesis, Simple Linear Regression Model
H = W * x_data + b

# 6. loss function
loss = tf.reduce_mean(tf.square(t_data-H))

# 7. train node 생성
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

# 8. 실행준비 및 초기화작업 (변수가 있으면 초기화가 필요하다)
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # 초기화 작업

for step in range(30000):
    _, W_val, b_val = sess.run([train,W,b])
    
    if step % 3000 == 0:
        print('W:{},b:{}'.format(W_val, b_val))
        
print(sess.run(H))
```

