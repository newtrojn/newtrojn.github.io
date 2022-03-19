---
layout: post
title: NLG SubTask
date: 2022-03-19 19:59:00
tags: [nlp, subtask, nlg, GAN, txt_summarization]
description: nlg papers keyword & concept
comments: true
published: true
---
# Text Summarization, GAN

### 1.  GAN(Generative Adversarial Network), 생성적 적대 신경망

: 컴퓨터가 새로운 데이터를 생성할 수 있는 신경망
※ 본 내용은  ‘머신 러닝 교과서 with 파이썬, 사이킷런, 텐서플로(개정 3판)’ 의 내용을 정리함.

1. Introduction
    - GAN의 주요 목적 : 훈련 데이터셋과 동일한 분포를 가진 새로운 데이터 합성 → GAN의 원본 형태는 레이블 데이터가 필요하지 않기에 비지도 학습 범주로 속함.

1. AutoEncoder & Variational AutoEncoder 소개 및 GAN과의 관계
    - AutoEncoder
    훈련 데이터를 압축하고 해제할 수 있음. Encoder 신경망과 Decoder 신경망 2개가 연결되어 구성.
        
        ![Untitled](Week1-3%20%E1%84%80%E1%85%AA%20b74eb/Untitled.png)
        
        인코딩된 벡터 z : 잠재 벡터(latent vector) or 잠재 특성 표현.  → 입력 샘플의 차원보다 작음.  
        ⇒  데이터 압축 기능
        
        아래 이미지는 다층 신경망처럼 은닉층이 없는 AutoEncoder 이지만, 여러 은닉층을 추가하여 심층 AutoEncoder 생성 가능(효과적인 데이터 압축과 재구성 함수 학습 가능)
        
        AutoEncoder에 훈련되고 나면 저차원 공간의 압축된 버전에서 이 입력을 재구성  가능. 
        = 새로운 데이터 생성 불가
        
        생성 모델은 잠재 표현에 해당하는 랜덤한 벡터 z에서 새로운 샘플 y를 생성할 수 있음. 
        AutoEncoder에서 생성 모델로 바라보면 Decoder 신경망이 생성 모델과 비슷하다는 것을 알 수 있음.
        
2. [GAN 구조](https://developers.google.com/machine-learning/gan/gan_structure)
    
    ![Untitled](Week1-3%20%E1%84%80%E1%85%AA%20b74eb/Untitled%201.png)
    
- The Discirminator  : 한국어로 ‘판별자’라고 부름. 실제 데이터와 생성자에 의해 생성된 데이터를 구별하는데 쓰임. 판별자(D)에 대해 이 값을 최대화하고 생성자(G)를 최소화 해야함.
- The Generator : 판별자의 피드백을 통합하여 가짜 데이터를 생성하는 방법을 배움. 이를 통해 판별자가 출력을 실제로 분류하도록 하는 방법 배움.

- VAE (Variational AutoEncoder) : AutoEncoder을 생성 모델로 일반화한 방법 중 하나
Encoder 신경망에서 잠재 벡터 분포의 두 요소 평균과 분산을 계산. 
VAE를 훈련하는 동안 이 평균과 분산을 표준 정규 분포에 맞추도록 조정함.
Decoder 신경망에서 가우시안 분포에서 랜덤하게 샘플링한 z 벡터 주입해서 새로운 샘플  y생성.
    
    ![                                      Latent Space에 있는 분포도를 보면 표준 정규 분포의 모형임](Week1-3%20%E1%84%80%E1%85%AA%20b74eb/Untitled%202.png)
    
                                          Latent Space에 있는 분포도를 보면 표준 정규 분포의 모형임
    
    ![Untitled](Week1-3%20%E1%84%80%E1%85%AA%20b74eb/Untitled%203.png)
    

### 2.  NLG extractive summarization task

[Papers with Code - Neural Extractive Text Summarization with Syntactic Compression](https://paperswithcode.com/paper/neural-extractive-text-summarization-with)

1. 문제 정의, task가 해결하고자 하는 문제
문서의 요약 추출물을 생성하려고 할 때에 neural network를 활용하여 extraction modul과 compression modul의 장점들을 조합하여 neural extractive systems, additional flexibility from compression, interpretability 향상 시키기
2. 데이터셋 소개(대표적인 데이터셋1개)
    1.  task를 해결하기 위해 사용할 수 있는데 데이터셋이 무엇인가?
    CNN/Daily Mail : CNN 및 Daily Mail 웹사이트의 뉴스 기사에서 바탕으로 질문으로 생성되어 컴퓨터 또는 시스템이 이 질문에 답할 것으로 예상되는 구절로 생성되었음.
    Training Pairs : 286,817, Validation Pairs : 13,368 , Test Pairs : 11,487
    2. 데이터 구조

```
{'id': '0054d6d30dbcad772e20b22771153a2a9cbeaf62',
 'article': '(CNN) -- An American woman died aboard a cruise ship that docked at Rio de Janeiro on Tuesday, the same ship on which 86 passengers previously fell ill, according to the state-run Brazilian news agency, Agencia Brasil. 
The American tourist died aboard the MS Veendam, owned by cruise operator Holland America. Federal Police told Agencia Brasil that forensic doctors were investigating her death. 
The ship's doctors told police that the woman was elderly and suffered from diabetes and hypertension, according the agency. 
The other passengers came down with diarrhea prior to her death during an earlier part of the trip, the ship's doctors said. 
The Veendam left New York 36 days ago for a South America tour.'
'highlights': 'The elderly woman suffered from diabetes and hypertension, ship's doctors say .Previously, 86 passengers had fallen ill on the ship, Agencia Brasil says .'}
```

```
한국어 ver.
기사 : 브라질 국영 통신사 Agencia Brasil에 따르면 화요일 리우데자네이루에 정박한 유람선에서 86명의 승객이 질병에 걸렸던 유람선에서 미국인 여성이 사망했다.
미국인 관광객은 크루즈 운영사 홀랜드 아메리카가 소유한 MS Veendam에서 사망했습니다. 연방 경찰은 Agencia Brasil에 법의학 의사가 그녀의 죽음을 조사하고 있다고 말했습니다.
선박의 의사들은 경찰에 그 여성이 고령이며 당뇨병과 고혈압을 앓고 있다고 말했습니다.
다른 승객들은 여행 초반에 그녀가 사망하기 전에 설사를 했다고 배의 의사들이 말했습니다.

요약(하이라이트) : '노인 여성은 당뇨병과 고혈압을 앓고 있었다고 배의 의사는 말했습니다. 이전에는 86명의 승객이 배에서 병에 걸렸다고 Agencia Brasil은 말했습니다.'
```

1. SOTA 모델 소개(대표적인 모델 2개 이상)
    1. HAHSum
    문장 수준의 추출 텍스트 요약에 특화된 알고리즘.
    BERT는 문장 간의 관계 모델링 전문이기에 선택된 대상의 요약을 위한 의미를 고려하는 기능이 없고, 문장과 문장 사이의 종속성을 고려하지 않음.
    HAHSum(Hierarchical Attentive Heterogeneous Graph for Text Summarization, 텍스트 요약을 위한 계층주의 이질적 그래프)의 경우 중복 인식 그래프로 문장 표현을 반복적으로 개선하고, 메시지 전달을 통해 레이블의 종속성을 전달.
    
    ![Untitled](Week1-3%20%E1%84%80%E1%85%AA%20b74eb/Untitled%204.png)
    

b. MatchSum
gold summary와 문서에서 추출된 요약을 일치시키는 알고리즘.
추출된 요약은 의미상 문서에 더욱 가깝지만 gold summary는 아주 많이 가까움.

![Untitled](Week1-3%20%E1%84%80%E1%85%AA%20b74eb/Untitled%205.png)

4-1. 논문 키워드 1 **Compression in Summarization**

고질적 문제 : 문장을 더 짧게 하는데에 있어 핵심을 덜 지우는 방법
과거 해결 방법 : 통사론 기준(명사, 동사, 형용사, 부사 뭐 이런거), [discoursed-based](https://blackrice91.tistory.com/97)(앞선 문장에 의해 영향을 받고 다음 문장에 영향을 미침.)
ex. ‘6.25 전쟁은 남북을 분단 시킨 엄청난 사건이었다. 그리고 이 전쟁은 아직까지 종전되지 않고 있다. ‘ 에서 ‘이 전쟁’ = ‘6.25’ [앞 문장에 영향 받음])

제거 기준 : 긍정 명사구, 관계절과 부사절, 명사구의 형용사구 및 부사구, 명사구의 일부로서의 동명사, 월요일과 같은 특정한 구성의 전치사구, 괄호 및 기타 괄호 안의 내용

4-2. 논문 키워드 2 **Text Compression**

![Untitled](Week1-3%20%E1%84%80%E1%85%AA%20b74eb/Untitled%206.png)

![Untitled](Week1-3%20%E1%84%80%E1%85%AA%20b74eb/Untitled%207.png)

문장 선택 후, 개별 압축 옵션을 평가하고 선택한 문장에서 특정 구 또는 단어를 제거할지 여부를 결정하는 것
위 이미지에서 ‘PP’는 키워드 1에서 설명 했듯이  제거기준에 해당함.
일단 이 문장과 document context와 decoding context를 조합한 compression을 인코딩함.
그러고나서 ‘Feedforward network’를 사용하여 지울지 말지 결정함.

4-3. 논문 키워드 3 **Oracle Construction (여기서의 Oracle은 계속 고민해봤는데, 정처기 공부할 때 봤던 테스트 오라클의 ‘오라클’과 같은 의미인듯. 참/거짓 판별하는 방법)**

<aside>
📕 키워드 속의 키워드 : Beam Search
아까 위에서 Decoder에 대한 설명을 잠시 했는데....ㅋㅋㅋㅋ 후 파도파도 끝나지 않는 AI/ML....눈물만 난다.  모르는 용어는 제대로 짚고 넘어가야 담주가 편할라나~~(절대 그렇지 않겠지)~~
Decoder의 역할 : Encoder의 결과인 Context Vector(아까 언급한 z)를 받아 캡션 생성, 기계 번역 등의 작업을 연속적으로 수행함. 이 Decoding하는 방법으로 ‘Greedy Search’와 ‘Beam Search’가 있음.
Greedy Search : 컴퓨터가 기본적으로 배우는 방식에서  Fully Connected Layer를 통과한 결과에 Softmax를 취해서 가장 높은 확률을 가지는 단어 하나를 선택 → 단점 : 한 번 모델에서 틀린 값을 내놓게 되면 그대로 전달되어 학습됨. 클린 값 수정 불가
Beam Search : 1개의 출력 값이 아닌 Beam 개수(k) 만큼 출력해서 마지막에 가장 좋은  Sequence가 무엇인지 판단하도록 하는 방법 → 단점: Decoding 매우 많이 수행하여 연산량 증가, 문장이 길어지면 기존과 동일하게 정확도가 떨어짐.

</aside>

![Beam Search](Week1-3%20%E1%84%80%E1%85%AA%20b74eb/Untitled%208.png)

Beam Search

Beam Search를 사용하여 추출한 오라클 문장들을 식별함.
추가할 각 문장에 대해, 참조 요약과 관련하여  주어진 문장의 ROUGE 점수와 동일한 경험 비용을 계산하여 내림차순으로 정렬. (Beam의 갯수 정할 때 참고함.)
이렇게 검색된 다른 문장 조합을 반환하여 Extraction 전용 모델과 Extraction + Compression 모델에 추출된 문장 오라클을 사용함.
