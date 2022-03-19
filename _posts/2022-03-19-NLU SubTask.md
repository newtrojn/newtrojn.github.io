---
layout: post
title: NLU SubTask
date: 2022-03-19 19:48:00
tags: [nlp, subtask, nlu, multi_label_classification,multiple_choice,commonsense_reasoning,semantic_similarity]
description: nlu subtask 3개 조사
comments: true
published: true
---
# multi-label classification & multiple choice QA, Commonsense reasoning & Semantic Similarity

- **multi-label classification, multiple choice QA**
    1. 문제 정의
    다양한 분야에서 법률, 법률 해석, 법적 다툼 등은 일반적으로 서면으로 표현되어 방대한 양의 법률 텍스트 생성 → 데이터의 규모가 점점 커짐
    현재의 최첨단 모델이 법률 영역의 다양한 업무(해석, 다툼, 주장 등)에 걸쳐 일반화 될 수 있는지의 여부가 중요함.
        
        아래 이미지를 보면 LexGLUE dataset 구축 후 성능 확인을 위한 task를 확인할 수 있음. 
        이를 통해 ‘어느 법’인지와 법에 대한 ‘검색’ 기능이 법률 영역의 다양한 업무를 보조 해주는 주요 기술임을 유추 가능
        
        ![Untitled](https://user-images.githubusercontent.com/94058241/159118152-295fda5d-024f-4b43-923f-183fdc122551.png)
        
    2. 데이터 소개
    1) [LeXGLUE](https://paperswithcode.com/dataset/lexglue)
    
    ```
    @article{chalkidis-etal-2021-lexglue,
            title={LexGLUE: A Benchmark Dataset for Legal Language Understanding in English},  #논문 제목
            author={Chalkidis, Ilias and Jana, Abhik and Hartung, Dirk and
            Bommarito, Michael and Androutsopoulos, Ion and Katz, Daniel Martin and
            Aletras, Nikolaos}, #저자
            year={2021}, #발행년도
            eprint={2110.00976}, #발행번호..? 고유번호
            archivePrefix={arXiv}, #아카이브 이름
            primaryClass={cs.CL}, #분야, cs : computer science /  CL: Computer and Language
            note = {arXiv: 2110.00976}, #archivePrefix+eprint
    }
    ```
    
    1. SOTA 모델 소개
        1. BERT
        2. Legal-BERT
        3. CaseLaw-BERT
        4. BigBird
        5. Longformer
        6. RoBERTa
        
        ![Untitled 1](https://user-images.githubusercontent.com/94058241/159118105-13f6fed4-bd00-4473-8360-beacf162ddb2.png)
        
    2. 논문 요약 키워드
        
         **pre-trained Transformer-based( : Transformer로 미리 학습)
        
        mask(문장의 특정 단어를 가림. BERT가 예측할 수 있게끔. 입력 텍스트의 단어 집합의 15%의 단어를 랜덤으로 마스킹(Masking))**
        
        ![Untitled 2](https://user-images.githubusercontent.com/94058241/159118126-9e499c5f-96a7-4982-8269-9269a6707189.png)
        
- **Commonsense reasoning, Semantic Similarity**
    1. 문제 정의
    1) Commonsense reasoning(상식적 추론)
    ex. 시 의원들은 시위대의 허가증 발급을 거부했다. 왜냐하면 그들은 폭력을 하기 때문이다. 누가 폭력을 무서워하는가? A.  시위대  **B. 시의원**
    2) Semantic Similarity(의미적 유사성)
    
    2. 데이터 소개
        1. WSC, WSCR, WNLI(WSC에서 비롯됨. 형식 비슷함.)
            
            ```
            {
              'label': 0,
              'options': ['The city councilmen', 'The demonstrators'],
              'pronoun': 'they',
              'pronoun_loc': 63,
              'quote': 'they feared violence',
              'quote_loc': 63,
              'source': '(Winograd 1972)',
              'text': 'The city councilmen refused the demonstrators a permit because they feared violence.'
            }
            ```
            
        2. [PDP60](https://cs.nyu.edu/~davise/papers/WinogradSchemas/PDPChallenge2016.xml)
        
        ```xml
        1. Then Dad figured out how much the man owed the store; to that he added the man's board-bill at the cook-shanty. He subtracted that amount from the man's wages, and made out his check
        Snippet: **He** subtracted
        	A. Dad
        	B. the man
        **Correct Answer**: A
        **Results on human subjects**: 19 out of 19 subjects got this answer.
        **Source**: Laura Ingalls Wilder, By the Shores of Silver Lake
        ```
        
    3. SOTA 모델 소개
        1. HNN(Hybird neural network) : 키워드는 바탕색 있음.
            1. 구조
            input : sentence S(a pronoun to be resolved), candidate antecedent C
            input 내용이 MLM(Masked Languaged Model)과 SSM(Semantic Similarity Model) input layer에 들어감.
            BERT Encoder를 지나서 각각의 MLM과 SSM의 output layer에 산출.
            Final output score : MLM의 값과 SSM의 값의 평균 → S의 대명사가 C일 확률
                
                ![Untitled 3](https://user-images.githubusercontent.com/94058241/159118135-57a984d0-0f18-430d-9ef4-c65aa2957ea3.png)
                
            2. Ablation Study
            
            <aside>
            💡 Ablation 의 사전적 뜻 : 삭마(削磨: 풍화·침식 작용에 의해 얼음·눈·암석이 깎이는 현상)    *~~※ 삭마라는 단어는 살면서 첨 보는 것 같다... 자연현상이 AI/ML에서 쓰이진 않겠짘ㅋㅋㅋㅋㅋ~~*
            Ablation Study : In artificial intelligence(AI), particularly machine learning (ML), **ablation** is the removal of a component of an AI system. An **ablation study** studies the performance of an AI system by removing certain components, to understand the contribution of the component to the overall system. ([wikipedia](https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)))
            **즉, AI 시스템의 일부를 제거함으로써 해당 부분이 전체적인 시스템의 성능에 기여하는 바를 연구하는 것. 이를 통해 **시스템의 인과관계 파악 가능**
            
            </aside>
            
            아래 표를 보면 정확도가 확연히 떨어졌음을 확인할 수 있음.
            
            이미지를 보면  SSM과 MLM은 서로 보완 관계임.
            첫 번째 한 쌍의 예시에서 MLM은 제대로 예측함. 모델 사전 교육에 사용되는 문장 말뭉치에 ‘the tree repaired’가 더 많이 나타나기 때문.
            두 번째 한 쌍의 예시에서는 반대의 결과로, SSM이 “시의회”와 “시위대” 모두 폭력을 옹호할 수 있고, 둘 다 다른 쪽보다 상당히 자주 발생하지 않기 때문에 맥락에 따라 차이를 구별하는데 효과적임을 알 수 있음.
            
            |  | WNLI | WSCR | WSC | PDP60 |
            | --- | --- | --- | --- | --- |
            | HNN | 77.1 | 85.6 | 75.1 | 90.0 |
            | -SSM | 74.5 | 82.4 | 72.6 | 86.7 |
            | -MLM | 75.1 | 83.7 | 72.3 | 88.3 |
            
            ![Untitled 4](https://user-images.githubusercontent.com/94058241/159118143-93199b0d-ecb6-40d2-9a9f-60dd3283ce40.png)
            
            1. 논문 요약 키워드 :HNN **키워드 : MLM(자주 언급되는 단어를 예측할 가능성이 높음)  , SSM(맥락에 따른 차이를 구별하는데 효과적, cosine similarity 활용)**
        2. BERTWiki-WSCR(사전 훈련시 사용되는 데이터셋에 대한 연구)
            1. 키워드 1, **WSC**
            WSCR 훈련 세트 및 실험에 도입되는 Winograd 유사 데이터 세트에 대해 사전 훈련된 BERT LM(Devlin 등,2018년) 활용
            **훈련 문장 s가 주어졌을 때, 풀어야 할 대명사는 문장으로부터 가려지고, LM은 복면 대명사 대신 정확한 후보를 예측하기 위해 사용된다.**
            2. 키워드 2, **MaskedWiki Dataset**
            **미세 조정을 위한 더 많은 데이터를 얻기 위해 WSC와 유사한 대규모 문장 컬렉션을 자동으로 생성
            동일한 명사가 두 번 이상 포함된 문장의 텍스트 말뭉치 → 2번째 corpus는 masking**
            대체 명사와 다른 문장의 각 명사에 대해 토큰이 주어짐
            WSC의 사례와 구조적으로 유사한 예를 얻지만, 모든 요건을 충족하는지 확인할 수는 없음.
            3. 키워드 3,  **WNLI**
            **모델은 WNLI 데이터 세트의 테스트 세트에서 추가로 테스트됨.
            WSC273 데이터 세트와 동일한 평가 접근 방식을 사용하기 위해 WNLI의 예를 전제-가설 형식에서 마스킹된 단어 형식으로 변환
            각 가설은 대명사가 후보에게 대체된 전제의 하위 문자열에 불과하기 때문에, 대체된 대명사와 후보 하나를 찾는 것은 전제의 하위 문자열로 그 가설을 찾는 것으로 할 수 있음.
            WSC273과 겹치지 않기 때문에 WNLI 데이터 세트의 테스트 세트만 사용**
