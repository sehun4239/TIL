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

