---
layout: post
title: Attention Is All You Need-Papers Review
date: 2022-03-20 00:43:00
tags: [nlp, BERT, attention, transformer]
description: 'Attention is all you need' review
comments: true
published: true
---


### 1. 기존의 자연어 처리

- Method : Sequence modeling이었던 RNN, CNN, LSTM과 encoder-decoder의 구조로 처리
- Problem : training 데이터 내에서 병렬 처리를 배제시켰기 때문에 문장이 길어지게 되면 메모리 제약으로 인해 훈련되기 힘든 상황이었음. 따라서 이를 해결하기 위해 factorization tricks와 conditional computation으로 효율성을 향상하였음. 그러나 근본적 문제는 해결X

![Untitled](https://user-images.githubusercontent.com/94058241/159128091-a1d267bc-407e-4b42-8460-2e161367c7d3.png)

### 2. Attention

Sequence modeling and transduction models의 필수적 요소가 되었는데 input or output sequence에서의 거리에 관계없이 그 단어들의 종속성을 모델링할 수 있음.

![Untitled](https://user-images.githubusercontent.com/94058241/159128010-626bc9c3-932e-4cf5-84d4-482d4b8d64f6.png)

Attention이 돌아가기 위해서는 input값은 transformer의 형태로 처리되는데 이는 기존의 encoder-decoder의 구조를 거쳐 ‘self-attention’ 과 ‘point-wise’를 통해 처리된다. 이 2가지 개념이 Transformer와 관련하여 가장 중요한 개념이다.

본격적으로 Attention에 대해 알아보자.
Attention은 query와 key-value쌍을 output을 산출할 때에 mapping 하는 것으로 설명할 수 있는데 query, key, value 모두 벡터의 형태이며 output 시 가중 합계로 계산된다. 즉 각 값(value)에 할당된 가중치는 해당 key로 query한다. 아래 이미지를 보며 자세히 그 과정이 어떻게 이루어지는지에 대해 설명하도록 하겠다.

![Untitled 2](https://user-images.githubusercontent.com/94058241/159128027-16b69a26-0f08-4659-aa45-38951ef8854a.png)

‘Thinking’이라는 단어를 self-attention 계산을 해보도록 하겠다. 우리는 가장 먼저 이 단어에 대해 입력 문장에서의 점수를 매겨야 한다. 이 점수는 특정 위치에서 단어를 인코딩할 때, 입력된 문장의 다른 부분에 얼마나 집중할 것인지를 결정한다.(=가중치) 이 점수는 query와 key를 이용하여 매겨지게 된다. 그러고서 key 벡터 차원의 제곱근으로 나누는데 논문 기준 그리고 default값이 8이다. 8로 나눈 다음 softmax를 하여 이 위치에서 각 단어가 얼마나 표현될 수 있는지에 대한 확률값을 얻는다. 그런 후 각각의 value 벡터에 이 점수를 곱하면 가중치 벡터가 나오는데 이 값들을 합산한다. 이러면 ‘thinking’이라는 단어의 self-attention layer의 출력을 생성한 것이다.

![Untitled 3](https://user-images.githubusercontent.com/94058241/159128035-a27e321e-a0ba-4eab-b85d-4c06f9b87963.png)

![Untitled 4](https://user-images.githubusercontent.com/94058241/159128042-95b99fd2-cd1d-49f9-9c9a-f0452570cde2.png)

이러한 방식으로 embedding layer의 차원을 query, key, value에 맞게 조정할 수 있는 값들이 준비되었고 아래와 같은 계산 방식으로 차원을 조정한다.

![Untitled 5](https://user-images.githubusercontent.com/94058241/159128046-5ecd733c-b3b8-4cdd-b1bc-f7a33793f177.png)

### 3. 그럼 왜 하필 attention 이냐?......

결론을 가장 먼저 얘기하자면, 계산 복잡도가 이전 모델들 보다 덜 복잡하고, 해석 가능한 모델을 생성할 수 있다.

![Untitled 6](https://user-images.githubusercontent.com/94058241/159128052-cbcd968d-735b-4606-8dfa-e595c7dd206e.png)

레이어 별 복잡도는 self-attention이 차원 기준으로 가장 낮다. 왜냐하면 모든 위치를 순차적으로 연결하여 실행되기 때문이다.
또한 다른 영역에서도 O(1) 임을 논문에서 밝히고 있는데 RNN, CNN과 비교했을 때 가장 복잡도가 낮다. 특히 아까 기존 모델의 문제점으로 학습이 진행될수록 이전 문장에 대해 학습했던 memory가 소실되는 것을 언급하였는데 self-attention의 경우 그 길이가 아주 짧은 편이다.(O(1)) 따라서 학습을 하더라도 memory 소실이 가장 적다.

### 4. Attention 개념의 등장 이후 적용해본 실험들

- Machine Translation : BLEU 점수가 41.8점을 달성하였다.

![Untitled 7](https://user-images.githubusercontent.com/94058241/159128059-5ad5686f-7c51-4031-aaee-9937990d2365.png)

- Model Variations

![Untitled 8](https://user-images.githubusercontent.com/94058241/159128067-08f583b5-8180-44c2-ab95-a4aeb3b7930c.png)

- English Constituency Parsing이 더욱 수월하게 학습됨

![Untitled 9](https://user-images.githubusercontent.com/94058241/159128083-101ce21d-d1ee-4914-809c-f1a709d20cd0.png)

⇒ encoder-decoder의 구조에서 사용되는 레이어들(RNN, CNN)들을 self-attention 방식으로 대체한 시퀀스 변환 모델인 transformer은 RNN, CNN 기반으로 한 구조보다 훨씬 빠르게 학습할 수 있었다.
