## 성능평가 지표

* 예측한 모델이 얼마나 정확한지에 대해서는 아래와 같은 지표들로 평가가 가능하다. 크게 3가지로 분류된다.

![1005_2](../markdown-images/1005_2.JPG)

![1005_3](../markdown-images/1005_3.JPG)

![1005_4](../markdown-images/1005_4.JPG)



### 다음으로 몇 가지 용어에 대해 이해를 하자

* Learning rate는 적절한 경우 모델 예측에 도움이 되지만 크거나 작을 경우 아래와 같은 문제가 일어날 수 있다. (Overshooting / Local minima)![1005_5](../markdown-images/1005_5.JPG)

* Normalization은 여러가지가 있지만 우리는 주로 Min-Max Normalization을 사용했다.
* underfitting은 학습이 너무 빈약하여 잘 이루어져 있지 않을 경우를 뜻하며 overfitting은 training data에 대해 너무 학습이 되어 오히려 실제 데이터에 적용이 안되는 경우를 말한다.

![1005_6](../markdown-images/1005_6.JPG)

* Overfitting을 해결하는 방법은 아래와 같은 방법이 있다.

![1005_7](../markdown-images/1005_7.JPG)

* 전체 학습데이터를 training data와 testing data로 나누어서 성능을 평가한다. 모델에 개선작업이 필요하면 traning data의 일부를 validation data set으로 이용한다.

![1005_8](../markdown-images/1005_8.JPG)

* 만약 데이터의 양이 작으면 cross validation을 이용하여 수행한다.![1005_9](../markdown-images/1005_9.JPG)





## Multinomial

Multinomial classification도 앞에서 수행한 Binary classification과 비슷하다. 다만 항의 개수에 맞게 logistic regression을 수행하는 점이 다르다.![1005_10_multinomial](../markdown-images/1005_10_multinomial.JPG)

![1005_11_multinomial](../markdown-images/1005_11_multinomial.JPG)

![1005_12_multinomial](../markdown-images/1005_12_multinomial.JPG)

![1005_13_multinomial](../markdown-images/1005_13_multinomial.JPG)



* 위의 내용을 특정 예제를 통해서 코드로 구현해보자

```python
# BMI 지수로 학습해보자. => 키와 몸무게를 가지고 저체중, 정상 과체중 비만을 판단하는 지수
# BMI = 자신의 몸무게(kg) / 키의 제곱(m)
#      18.5 이하 => 저체중
#      18.5 ~ 23 => 정상
#      23 ~ 25 => 과체중
#      25 ~ => 비만
# 우리가 하려는 건 식이 아니라 BMI 지수를 조사한 데이터가 있다.
# 이걸 학습해서 예측을 통해 나의 BMI 지수를 알아보자
# 단 제공하는 데이터는 4가지가 아니라 3가지 분류로 되어있다.

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('./data/bmi/bmi.csv', skiprows=3)

# display(df)

# 결측치 확인
# df.isnull().sum()        # 결측치 없음

# 이상치 확인
zscore = 1.8

# 이상치를 확인
# df.loc[np.abs(stats.zscore(df['height'])) >= zscore, :] # height의 이상치는 없다
# df.loc[np.abs(stats.zscore(df['weight'])) >= zscore, :] # weight의 이상치는 없다
# df.loc[np.abs(stats.zscore(df['label'])) >= zscore, :] # label의 이상치는 없다

# Data Split
# Train, Test 두 부분으로 분할. 분리하는 비율은 7:3으로 분리
# 나중에 Train부분은 k-fold cross validation을 진행
x_data_train, x_data_test, t_data_train, t_data_test = \
train_test_split(df[['height','weight']],df['label'],test_size = 0.3, random_state=0) # 14000 / 6000

# Normalization
scaler = MinMaxScaler()  # scaler 객체를 생성
scaler.fit(x_data_train) # scaler 객체에 최대 최소와 같은 정보가 들어간다. (fit 처리)

x_data_train_norm = scaler.transform(x_data_train)
x_data_test_norm = scaler.transform(x_data_test)

del x_data_train       # 혼동 방지를 위해 변수를 삭제
del x_data_test

# sklearn 구현은 매우매우 간단 - model 생성하고 학습진행
model = LogisticRegression()
model.fit(x_data_train_norm, t_data_train)

# 우리 model의 정확도를 측정해야한다.
# cross validation
kfold = 10
kfold_score = cross_val_score(model,x_data_train_norm, t_data_train, cv=kfold)
print('### cross validation ###')
print('score : {}'.format(kfold_score))
print('평균: {}'.format(kfold_score.mean()))

# 최종모델평가
predict_val = model.predict(x_data_test_norm)  # 테스트 데이터로 예측값을 구해요
acc = accuracy_score(predict_val, t_data_test)

print('우리 Model의 최종 Accuracy : {}'.format(acc))

# Predict

height = 188
weight = 78
my_state = [[height,weight]]
my_state_val = model.predict(scaler.transform(my_state))
print(my_state_val)
```



* accuracy를 확인하기위한 방법을 알아보자
  * classification_report
  * confusion_matrix

```python
from sklearn.metrics import classification_report

y_true = [0, 1, 2, 2, 2]   # 정답
y_pred = [0, 0, 2, 2, 1]   # 우리 model이 예측한 값

target_name = ['thin', 'normal', 'fat']

print(classification_report(y_true, y_pred, target_names = target_name))
```

```python
from sklearn.metrics import confusion_matrix

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]

confusion_matrix(y_true,y_pred)
```



* 이를 통해서 Tensorflow로 MNIST 예제를 다시 구현해보고 accuracy를 확인하자

```python
# Tensorflow 1.15버전을 가지고 MNIST예제를 구현
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns           # confusino matrix를 heatmap을 통해서 그래프 출력
from sklearn.preprocessing import MinMaxScaler   # Normalization
from sklearn.model_selection import train_test_split   # train, test 분리
from sklearn.model_selection import KFold         # Cross Validation
from sklearn.metrics import classification_report, confusion_matrix

# 1. Raw Data Loading
df = pd.read_csv('/content/drive/My Drive/MachineLearning/train.csv')
# display(df.head(), df.shape)      # (42000, 785)

# 2. 결측치와 이상치 처리 => 결측치를 찾고 만약 결측치가 있으면 수정하자 이상치는 (scipy zscore이용)
# 근데 MSNIST에는 없네

# 3. 사용하는 데이터가 이미지 데이터다 => 어떤 이미지인지 한번 확인해보자
#    df에서 label column은 제외하고 pixel 데이터만 들고오자
img_data = df.drop('label', axis=1, inplace=False).values
# 이미지들의 pixel 데이터만 ndarray로 추출(2차원) => 화면에 출력
fig = plt.figure()  # 출력할 전체 화면을 지칭하는 객체를 가져온다.
# fig안에 subplot을 만들거다. 저장할 list 만든다.
fig_arr = list()

for n in range(10):
  fig_arr.append(fig.add_subplot(2,5,n+1))
  fig_arr[n].imshow(img_data[n].reshape(28,28), cmap='Greys', interpolation='nearest')

plt.tight_layout()
plt.show()

# 4. Data Split
#    데이터는 크게 3부분으로 나누어야 한다.
#    일단 2부분으로 나누자 (train용, test용)
#    여기서 train용이라고 되어 있는 데이터를 다시 2부분으로 분리 (train, validation)
#    train : 학습용 ,  validation : 모델 수정용도의 데이터 셋
x_data_train, x_data_test, t_data_train, t_data_test = \
train_test_split(df.drop('label',axis=1), df['label'], test_size=0.3, random_state=0)

#5. norm
scaler = MinMaxScaler()
scaler.fit(x_data_train)
x_data_train_norm = scaler.transform(x_data_train)
x_data_test_norm = scaler.transform(x_data_test)

# 6 one hot
sess = tf.Session()
t_data_train_onehot = sess.run(tf.one_hot(t_data_train, depth=10))
t_data_test_onehot = sess.run(tf.one_hot(t_data_test, depth=10))

#########################
# Tensorflow 구현

# 1. placeholder
X = tf.placeholder(shape = [None,784], dtype=tf.float32)
T = tf.placeholder(shape = [None,10], dtype=tf.float32)

# 2. Weight,bias
W = tf.Variable(tf.random.normal([784,10]), name='weight')
b = tf.Variable(tf.random.normal([10]), name='bias')

# 3. Model(Hypothesis)
logit = tf.matmul(X,W) + b
H = tf.nn.softmax(logit)

# 4. loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logit, labels = T))

# 5. Optimizer를 이용한 train (Optimizer는 loss값을 줄이는 알고리즘)
train = tf.train.GradientDescentOptimizer(learning_rate = 1e-1).minimize(loss)

# parameter setting (기본적으로 2개는 설정)
num_of_epoch = 100
batch_size = 100

# 7. 학습진행
def run_train(sess ,train_x, train_t,):
  print('###학습시작###')
  sess.run(tf.global_variables_initializer())
  total_batch = int(train_x.shape[0] / batch_size)
  for step in range(num_of_epoch):
    
    for i in range(total_batch):
      batch_x = train_x[i*batch_size:(i+1)*batch_size]
      batch_t = train_t[i*batch_size:(i+1)*batch_size]
      _, loss_val = sess.run([train,loss], feed_dict={X:batch_x, T:batch_t})

    if step % 10 == 0:
      print('Loss: {}'.format(loss_val))
  print('###학습끝###')

# Accuracy
predict = tf.argmax(H, 1)     # [[0.1 0.3 0.2 .... 0.1]]


# sklearn을 이용해서 classification_report를 출력
target_name = ['num 0', 'num 1', 'num 2', 'num 3', 'num 4', 'num 5', 'num 6', 'num 7', 'num 8', 'num 9']
# train 데이터로 학습하고 train 데이터로 성능평가를 해보자
run_train(sess,x_data_train_norm,t_data_train_onehot)
print(classification_report(t_data_train, sess.run(predict, feed_dict={X:x_data_train_norm}), target_names= target_name))

```

