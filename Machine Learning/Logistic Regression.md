## Logistic Regression

* 지금까지는 선형 회귀였지만 다른 회귀에 대해 알아보자.

* 보통 Machine learning에는 regression과 classification이 있다.

* Classification은 정확도 측정이 가능하며 training data의 특성과 분포를 파악한 후 미지의 입력 데이터에 대해 어떤 종류의 값으로 분류될 수 있는지 예측한다.

* Classification 구현을 위해서는 많은 알고리즘이 있다. (SVM, Naive Bayse ...) 이번에는 logistic regression에 대해 알아보자

  * logistic regression은 정확도가 상당히 높다.
  * Deep learning의 기본 component로 활용된다.

* 동작방식은 Linear Regression을 이용해서 training data set의 특성과 분포를 파악해서 직선을 찾는다. (2차원 대상, 3차원은 평면/ 4차원은 초평면 ...)

* 그 직선을 기준으로 데이터를 분류한다. (0 or 1)

* 다만 분류(classification)작업을 하기에 문제가 있다. (직선이기 때문에 문제가 된다)

  => 이를 해결하기 위해 Sigmoid 함수 (S 모양의 곡선으로 변환)을 도입한다.

![Logistic Regression3_lossfunction](../markdown-images/Logistic Regression3_lossfunction.JPG)

* 위는 sigmoid 함수를 나타낸다. 그렇다면 최소제곱법은 저렇게 나타내진다. 하지만

![Logistic Regression3_lossfunction2](../markdown-images/Logistic Regression3_lossfunction2.JPG)

* Sigmoid 함수를 최소제곱법으로 활용하면 W값에 따라 local minimum이 걸릴 수 있다. 이를 해결하기 위해서 Cross Entropy를 활용한다.![Logistic Regression3_lossfunction3](../markdown-images/Logistic Regression3_lossfunction3.JPG)



### tensorflow를 이용한 구현

```python
import tensorflow as tf
import numpy as np

# training data set
x_data = np.array([[1,0],
                  [2,0],
                  [5,1],
                  [2,3], 
                  [3,3],
                  [8,1],
                  [10,0]])

t_data = np.array([[0],
                   [0],
                   [0],
                   [1],
                   [1],
                   [1],
                   [1]])

# placeholder
X = tf.placeholder(shape=[None,2], dtype=tf.float32)
T = tf.placeholder(shape=[None,1], dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([2,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# Hypothesis(Logistic Model)
logits = tf.matmul(X,W) + b     # Linear Regression Hypothesis
H = tf.sigmoid(logits)

# loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=T))

######### Linear Regression에서 사용했던 코드 그대로 사용!
# train node 생성
train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# Session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습을 진행
for step in range(30000):
    _,W_val,b_val,loss_val = sess.run([train,W,b,loss], feed_dict={X : x_data, T : t_data})
    
    if step % 3000 == 0 :
        print('w:{},b:{},loss:{}'.format(W_val,b_val,loss_val))
    
# 예측 !!
print(sess.run(H, feed_dict={X : [[4,2]] }))
```



## 예제를 통해 배운 3가지 방법을 정리하자

```python
# %reset          # 메모리 초기화

# Logistic Regression을 python, tensorflow, sklearn으로 각각 구현
# 처음은 독립변수가 1개인 걸로 하자

import numpy as np
import tensorflow as tf
from sklearn import linear_model

# 수치미분함수 들고와서 쓰자
###########################
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

##########################
# Raw Data Loading + Data Preprocessing
# 이번 예제에는 이 과정 필요 없음

# Training Data Set
# 지도학습을 하고 있기 때문에 독립변수와 종속변수 (label)로 구분해서 데이터 준비
# 어떤 경우에는 이 두객를 아예 분리해서 제공하는 경우도 있다. 참고

x_data = np.arange(2,21,2).reshape(-1,1)
t_data = np.array([0,0,0,0,0,0,1,1,1,1])

###########################
# python 구현부터 해보자

# Weight & bias
W = np.random.rand(1,1)
b = np.random.rand(1)

# 위에서 정의한 W와 b의 값을 구해야 한다.
# 이 값만 구하면 우리의 최종 목적인 model을 완성할 수 있다.

# loss function(손실함수) - 우리 모델의 예측값과 들어온 t_data(정답)
# 입력으로 들어온 x_data와 W,b 값을 이용해 예측값 계산, t_data(정답)을 비교함
def loss_func(input_obj):     # input_obj : W와 b를 같이 포함하는 ndarray [W1 W2 b]
    
    num_of_bias = b.shape[0]   # num_of_bias : 1
    input_W = input_obj[:-1*num_of_bias].reshape(-1,num_of_bias)  # W 생성
    input_b = input_obj[-1*num_of_bias]
    
    # 우리 모델의 예측값은 : (linear regression model(Wx+b) => sigmoid를 적용)
    z = np.dot(x_data, input_W) + input_b
    y = 1 / ( 1 + np.exp(-1 * z) )
    
    delta = 1e-7 # 굉장히 작은값을 이용해서 프로그램으로 로그연산시 
                 # 무한대로 발산하는 것 방지
    
    # cross entropy
    return -np.sum(t_data*np.log(y+delta) + ((1-t_data)*np.log(1-y+delta)))

# learning rate
learning_rate = 1e-4

# 학습
for step in range(30000):
    input_param = np.concatenate((W.ravel(), b.ravel()), axis=0)  # [W1 W2 b]
    derivative_result = learning_rate * numerical_derivative(loss_func, input_param)
    
    num_of_bias = b.shape[0]
    
    W = W - derivative_result[:-1*num_of_bias].reshape(-1,num_of_bias) # [[W1] [W2]]
    b = b - derivative_result[-1*num_of_bias:]
    
# predict => W,b를 다 구해서 우리의 logistic Regression model을 완성
def logistic_predict(x):   # 공부한 시간이 입력
    z = np.dot(x,W) + b
    y = 1 / ( 1 + np.exp(-1*z) )
    
    if y < 0.5 :
        result = 0
    else:
        result = 1
        
    return result, y

study_hour = np.array([[13]])
result = logistic_predict(study_hour)
print('######## python 결과값 ##########')
print('공부시간 : {}, 결과 : {}'.format(study_hour,result))
```



```python
### sklearn으로 구현해보자

# logistic regression model 생성
model = linear_model.LogisticRegression()

# training data set을 이용해서 학습
model.fit(x_data,t_data.ravel())       # x는 2차원, label은 1차원 vector로 넣어야함

study_hour = np.array([[13]])
predict_val = model.predict(study_hour)
predict_prob = model.predict_proba(study_hour)

print('######## sklearn 결과값 ##########')
print('공부시간 : {}, 결과 : {},{}'.format(study_hour,predict_val,predict_prob))
```



```python
# Tensorflow

# placeholder
X = tf.placeholder(dtype=tf.float32) # 독립변수가 1개인 경우 shape을 명시하지 않음
T = tf.placeholder(dtype=tf.float32) # t_data

# Weight & bias
W = tf.Variable(tf.random.normal([1,1]), name='weight')
W = tf.Variable(tf.random.normal([1]), name='bias')

# Hypothesis
logit = W * X + b  # matrix 곱연산 하지 않나요 ??
H = tf.sigmoid(logit)

# loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(30000):
    sess.run(train, feed_dict={X:x_data, T:t_data})
    
study_hour = np.array([13])
result = sess.run(H,feed_dict={X:study_hour})
print('######## tensorflow 결과값 ##########')
print('공부시간 : {}, 결과 : {}'.format(study_hour,result))
```



* 예제

```python 
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import linear_model
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 

# Multi Variable Logistic Regression
# 독립변수가 2개 이상인 logistic regression

# 학습하는 데이터는 GRE(Graduate Record Examination)와
# GPA 성적 그리고 Rank에 대한 대학원 합격/불합격 정보

# 내 성적 [600. 3.8 1.] 의 결과
# 첫번째는 sklearn으로 구현하기

# Raw Data Loading 
df = pd.read_csv('./data/admission/admission.csv')

# 결측치 확인
print(df.isnull().sum())     # 결측치가 없음 -> dropna 안해도됨

# 이상치를 확인해서 있으면 제거
# fig = plt.figure()
# fig_admit = fig.add_subplot(1,4,1)
# fig_gre = fig.add_subplot(1,4,2)
# fig_gpa = fig.add_subplot(1,4,3)
# fig_rank = fig.add_subplot(1,4,4)

# fig_admit.boxplot(df['admit'])
# fig_gre.boxplot(df['gre'])
# fig_gpa.boxplot(df['gpa'])
# fig_rank.boxplot(df['rank'])

# fig.tight_layout()
# plt.show()

# 확인했더니 이상치가 있어서 제거하려한다.
zscore_threshold = 2.0

for col in df.columns:
    outlier = df[col][np.abs(stats.zscore(df[col])) > zscore_threshold]
    df = df[~df[col].isin(outlier)]
    
print(df.shape)        # (382, 4)

# Training Data Set
x_data = df.drop('admit', axis=1, inplace=False).values
t_data = df['admit'].values.reshape(-1,1)


# 정규화를 진행해야한다.
scaler_x = MinMaxScaler()
scaler_x.fit(x_data)
norm_x_data = scaler_x.transform(x_data)   # for python, tensorflow

# sklearn을 이용한 구현
model = linear_model.LogisticRegression()
model.fit(x_data,t_data.ravel())
print('###### sklearn 결과 ######')
my_score = np.array([[600, 3.8, 1]])
predict_val = model.predict(my_score)  # 0 or 1
predict_proba = model.predict_proba(my_score)  # 불합격/합격 확률
print(my_score, predict_val, predict_proba)


# Tensorflow

# placeholder
X = tf.placeholder(shape=[None, 3], dtype=tf.float32)  # 독립변수의 데이터를 받기위한 placeholder
T = tf.placeholder(shape=[None, 1], dtype=tf.float32)  # 종속변수(label)의 placeholder

# Weight & bias
W = tf.Variable(tf.random.normal([3,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# hypothesis
logit = tf.matmul(X,W) + b
H = tf.sigmoid(logit)

# loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(30000):
    _, W_val, b_val, loss_val = sess.run([train,W,b,loss], 
                                         feed_dict={X:norm_x_data, T:t_data})
    
    if step % 3000 == 0:
        print('W:{},b:{},loss:{}'.format(W_val,b_val,loss_val))
        
my_score = np.array([[600,3.8,1]])
scaled_my_score = scaler_x.transform(my_score)

result = sess.run(H,feed_dict={X:scaled_my_score})
print('######## tensorflow 결과값 ##########')
print('점수 : {}, 결과 : {}'.format(my_score,result))
```

