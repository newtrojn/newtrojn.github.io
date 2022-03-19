---
layout: post
title: Sentiment Analysis, Text Classification
date: 2022-03-19 18:50:00
categories: [posts]
tags: [nlp, subtask, sentiment, txt_classification]
description: nlp subtask 2개 조사
comments: true
---
# Sentiment Analysis, Text Classification (feat. 원티드 프리온보딩)

### 1. 본인이 본 강의를 수강하는 목적에 대해서 자유롭게 적어보세요.

AI/ML과 관련하여 실무에서 요구하는 프로젝트 수준이 어느 정도인지, 얼만큼 공부해야 하는지 알고 싶어 지원하게 되었습니다.
프로젝트를 통해 취업까지 되면 일석이조 일 듯하네요 😁

### 2. Paperswithcode([https://paperswithcode.com/area/natural-language-processing](https://paperswithcode.com/area/natural-language-processing))에서 NLP sub task 중에 2개를 선택하여 본인 블로그에 정리해보세요. 
task 별로 아래 3가지 항목에 대해서 정리하세요.

- ****Sentiment Analysis****
    1. 문제 정의
    주어진 텍스트에 대하여 극성을 분류하는 것(긍정, 부정, 중립)
    2. 데이터 소개
    1) SST : 언어에서 감정의 구성 효과에 대한 완전한 분석을 허용하는, 레이블이 완벽하게 지정된 Parse tree로 존재하는 corpus, 영화 후기에서 추출된 단문으로 구성됨.
        
        SST-2 , SST binary: 긍정, 부정
        
        SST-5 : 매우 부정, 다소 부정, 중립, 다소 긍정, 매우 긍정
        
        2) GLUE : General Language Understading Evaluation은 9가지의 자연어 이해  Tasks의 집합체(single-sentence tasks CoLA and SST-2 포함)
        유사도 예측 : Tasks - MRPC, STS-B, QQP
        자연어 추론 :  MNLI, QNLI, RTE, WNLI
        
        3) IMDb : Internet Movie DB에서 긍정 또는 부정으로 라벨되어진 이중 감정 분석 시 사용되는 데이터셋
        
        4) MR : Movie Reviews : 긍정 부정의 감정의 극성 또는 평점으로 라벨링 되어진 영화 리뷰 문서의 모음과 객관적 상태에 대해(객관, 주관) 라벨 되어진 문장들
        
    3. SOTA 모델 소개
    1) SMART-RoBERTa Large
        - 기존 문제점 : 제한된 흘러 들어오는 데이터 자원과 사전 학습된 모델의 매우 큰 용량 →  공격적인 fine-tuning시 과적합 되는 문제점(사전 학습된 결과들을 잊고, 새로 들어오는 데이터들과의 과적합)
        - 이 모델의 제안점 : 
        모델의 용량을 효과적으로 관리하기 위한 부드러운 정규화, 
        모델의 과적합을 방지하기 위한 Bregman의 proximal point optimization   사용(TRPO; trust-region policy optimization).
        - 핵심 키워드 : Smoothness-inducing regularization, Bregman proximal point optimization
        
        2) NB-weighted-Bon + Cosine Similarity
        
        - 감성분석을 이용한 문서 분류에서, 분류의 결과 < 문서의 표현 결과(긍/부정)가 중요
        - 문서들이 임베딩된 모델들은 각각을 매핑하여 밀도가 높은. 실수 값 벡터로 변환
        - 이 모델의 제안점 : 
        내적(dot product) 사용보다 코사인 유사도(cosine similarity) 사용
        n-gram의 Naive Bayes 가중치를 feature combination에서 활용 → 97.42%의 정확도
        - 핵심 키워드 : cosine similarity, dot product, NB(Naive Bayes)
- ****Text Classification****
    1. 문제 정의
    문장 또는 문서를 적절한 카테고리에 할당하는 것. 
    이 카테고리는 선택된 데이터셋이나 토픽 내에서 지정됨.
    2. 데이터 소개
    1) AG News : “world”, “sports”, “business”, “sci/tech” 4가지 종류의 기사로 나뉘어져 있음.
        
        2) DBpedia : Wikipedia 프로젝트에서 만들어진 데이터. 다른 관련 데이터셋에 대한 링크를 포함하여 Wikipedia 자료의 관계 및 속성을 의미론적으로 쿼리 가능
        
        3) IMDb : Internet Movie DB에서 긍정 또는 부정으로 라벨되어진 이중 감정 분석 시 사용되는 데이터셋
        
    3. SOTA 모델 소개
    1) XLNet
        - 기존 문제점 : 마스크로 입력값을 손상시키며 학습하는 BERT는 마스킹된  토큰 간의 종속성(dependency)를 무시하고 사전 훈련 상의 미세 조정과 불일치의 문제
        - 이 모델의 제안점 : 
        AR, AE의 장단점 고려한 일반화된 자기회귀 사전훈련 방법 → BERT 한계 극복
        factorization order의 모든 순열에 대하여 예측된 likelihood를 최대화하여 bidirectional 학습을 가능하게 함.
        state-of-the-art Auto-regressive모델인 Transformer-XL의 아이디어를 사전 훈련에 통합
        - 핵심 키워드 : a generalized autoregressive pretraining method, AR, AE
        - 참고 링크 : [https://hyen4110.tistory.com/47](https://hyen4110.tistory.com/47)
