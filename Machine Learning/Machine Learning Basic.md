# Machine Learning



## Machine Learning Introduction

먼저 3가지 용어부터 정리하자

AI (Artificial Intelligence) : 인간이 가지고 있는 학습능력, 응용력, 추론능력을
컴퓨터를 통해서 구현하고자 하는 가장 포괄적인 개념

Machine Learning : AI를 구현하는 하나의 방법론. 데이터를 이용해서 데이터의 특성과 패턴을 학습하고 
그 결과를 바탕으로 미지의 데이터에 대한 미래결과를 예측하는 프로그래밍 기법
이런 Machine Learning을 구현하기 위한 방법
Regression, SVM(Support Vector Machine), Random Forest, Decision Tree,
Neural Network(신경망), Clustering, Reinforcement Learning 등등등...

Data Mining :데이터간의 상관관계나 새로운 속성(feature)을 찾는 것이 주 목적인 작업

Deep Learning : Machine Learning의 한 부분, Neural Network를 이용해서 학습하는 알고리즘의 집합
(CNN, RNN, LSTM, GAN, ...)

##################################################################

Machine Learning이 왜 필요할까 ??
Machine Learning이라는 개념은 1960년대 개념이 만들어졌다. 
Machine Learning은 Explicit program의 한계때문에 고안되었다.
Explicit program은 rule based program이라고 한다. 이런 Explicit program을 이용하면 왠만한 프로그램은 다
구현이 가능하다. 그런데 Explicit programming으로 할 수 없는 프로그램들이 있다.
Rule이 너무 많아서 즉, 조건이나 규칙이 너무 많아서 프로그램으로 표현하기 힘든거다.
예1) 스팸메일을 걸러주는 필터프로그램 -> 대출이라는 글자를 찾아서 메일을 spam처리하는 경우 -> 대~출은 못거름
=> 대~출을 등록 => 대^^출은 못 거름 .... => 프로그램으로 만들기 어렵다.
예2) 자율주행시스템 => 너무 많은 조건을 생각해서 차량을 운행해야 하기 때문에 어렵다.
예3) 바둑 

Explicit 프로그램의 한계때문에 Machine Learning 개념이 1960년대에 도입 되었다.
Machine Learning : 프로그램 자체가 데이터를 기반으로 학습을 통해 배우는 능력을 가지는 프로그램을 지칭




Machine Learning의 Type

Machine Learning은 학습방법에 따라서 크게 4가지로 분류 !!
- 지도학습 (Supervised Learning)
- 비지도학습 (Unsupervised Learning)
- 준지도학습 (SemiSupervised Learning)
- 강화학습 (Reinforcement Learning)

이 4개중에 우리가 관심이 있어하는 학습방법은 지도학습 (Supervised Learning)
우리가 해결해야 하는 현실세계의 대부분의 문제가 지도학습문제

### - 지도학습 (Supervised Learning) - classification algorithm
지도학습은 학습에 사용되는 데이터(data)와 그 정답(label)을 이용해서
데이터의 특성과 분포를 학습하고 미래 결과를 예측하는 방법.

지도 학습은 어떤 종류의 미래값을 예측하느냐에 따라
Regression (회귀)
Classification (분류) - binary classification
                      - multinomial classification

### - 비지도학습 (Unsupervised Learning) - clustering algorithm
비지도학습은 학습에 사용되는 데이터는 ... label이 없는 데이터가 사용된다. 이 부분이 지도학습과의 차이
비지도학습은 정답(label)이 없는 데이터만을 이용하기 때문에 입력값 자체가 가지고 있는 특성과 분포를 
이용해서 Grouping하는 Clustering(군집화)하는데 사용된다.

### - 준지도학습 = 지도학습 + 비지도학습
데이터의 일부분만 label이 제공되는 경우

### - 강화학습 = 위에서 말한 3가지 방식과는 완전히 다른 학습 방법
Agent, Environment, Action, Reward 개념을 이용
게임쪽에서 많이 사용되는 학습방법 => 바둑
Google 알파고의 메인 알고리즘이 바로 이 강화학습.



## Machine Learning 학습방법 중 Supervised Learning을 알아보자

지도학습은 입력값(x)와 Label이라고 표현되는 정답(t)를 포함하는 Training Data Set을 이용해서 학습을 진행

학습된 결과를 바탕으로 => Predictive model을 만든다. => 미지의 데이터에 대해 미래값을 예측하는 작업을 진행

공부시간에 따른 시험점수 산출(예측) => 점수로
공부시간에 따른 시험 합격여부 예측 => 합격/불합격
카드 사용패턴에 따른 도난신용카드 판별
공부시간에 따른 시험 Grade예측 (A/B/C/D/F)