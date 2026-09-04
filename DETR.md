## (1) 배경지식

---

#### 01 Attention

- input sequence가 길어지면 output sequence의 정확도가 떨어지는 것을 보정해주기 위한 등장한 기법
- **데이터 전체를 살펴보고 집중해서 살펴볼 위치를 정하게 된다.**
- **decoder** 에서 출력 단어를 예측하는 매 시점마다, **encoder 에서의 전체 입력 문장을 다시 참고 !**
  단, 전체 input sequence를 전부 다 동일한 비율로 참고하는 것이 아니라, 해당 시점에서 예측해야할 output과 연관이 있는 input 부분을 좀 더 집중  (바율 조정)
  → 학습시키고자 하는 class에 해당하는 부분만 집중하는 효과를 나타낼 수 있음
- **Query, Key, Value**로 구성되며, 일반적으로 Key와 Value를 같은 값을 가지게 함
  - Query : 찾고자 하는 대상으로, t시점의 decoder 셀에서의 hidden state
  - Key : 데이터를 찾고자 할 때 참조하는 값으로, 모든 시점의 encoder 셀의 hidden states
  - Value : Key에 대한 값으로, 모든 시점의 encoder 셀의 hidden states

    Query에 대한 Key를 찾아 Value를 계산 하는 과정을 거치는 것이 attention value를 구하는 것이며, **key와 query의 유사도에 value를 곱한 것을 더하는 방법**을 이용
    ![](https://blog.kakaocdn.net/dna/z74Wd/btrZbonkeQw/AAAAAAAAAAAAAAAAAAAAAGT_V_ot9Fmh11UCNKPYFoI2s94M_JdyfE8CsE0agrpz/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1790780399&allow_ip=&allow_referer=&signature=SD7bdvxMFKU9hFQjkzZ3hxZOmnM%3D)

#### 02 Self-Attention

- **encoder에서 이루어지는 attention 연산
  ![](https://blog.kakaocdn.net/dna/bvbFQT/btrZato8Wmh/AAAAAAAAAAAAAAAAAAAAAK4Gci8j1i7zc6bqATLfT_MKf2sh1G7U7tvt5bfmbaUO/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1790780399&allow_ip=&allow_referer=&signature=JDWrMM4HIQ5eG8q6YzfhE5RuWMg%3D)**

#### 03 Transformer

입력값에 상대적인 위치정보를 더하여 이용, 행렬 곱으로 한 번에 모든 연산을 수행

one-hot encoding 대신 label smoothing 이용 (0과 1 대신, 0과 1이 아닌 그에 가까운 값으로 만드는 것)

---

## (2) DETR

- ECCV 2020 발표. **Transformer 개념을** 처음으로 object **detection에 도입 !**
- **CNN Backbone + Transformer + FFN (Feed Forward Network)** 으로 구성됨.

### 2-1) CNN Backbone

> **input image($H_0×W_0$) → feature map ($d×HW$)**

input image를 CNN Backbone을 통과시켜 **feature map**을 뽑아냅니다.
이 feature map이 **transformer에 들어갈 수 있도록 처리를 해주어야 합니다.**

ⓐ input image(원본 이미지) 크기는 $H_0$ x $W_0$
ⓑ CNN을 통과하여 출력된 feature map은 $C×H×W$ ResNet50을 사용하였기 때문에 ($C$= $2048$, $H$ = $H_0/32$, $W$ = ${W_0}/32$ )
ⓒ 1x1 convolution을 적용하여 $d×H×W$ 형태로 바꿈 ( $C>d$ )
ⓓ transformer에 들어가기 위해서는 2차원이어야 하므로, $d×H×W$의 **3차원에서** $d×HW$의 **2차원으로 구조를 바꿈.**

---

### 2-2) Transformer

> **feature map ($d×HW$)** → *

![](https://static.wikidocs.net/images/page/145910/transformer.png)

- **Encoder (파란색 박스)**
  - ($d×HW$)의 **feature matrix**에 **Positional encoding 정보를 더한** matrix를 multi-head self-attention에 통과시킵니다.
    ( *Positional encoding 정보? ->* )
- ***Decoder***
  - (분홍색 박스) **N개의 bounding box**에 대해 **N개의 object query를 생성**합니다. 초기 object query는 0으로 설정되어있습니다. | N개의 object query
  - (보라색 박스) Decoder는 앞서 설명한 N개의 object query를 입력받아 **multi-head self-attention을 거쳐** 가공된 **N개의 unit을 출력**합니다. | N개의 unirt
  - (노란색 박스) 이 **N개의 unit**들이 Query로, 그리고 **encoder의 출력** unit들이 Key와 Value로 작동하여 **encoder-decoder** **multi-head attention**을 수행합니다. | N개의 unit
  - (초록색 박스) 최종적으로 N개의 unit들은 **각각 FFN**을 거쳐 **object class와 box 정보를 출력**합니다. | object class, N개의 box

> **Attention is All You Need와 구조상 살짝 다른 면이 있음**
>
> 1. **Positional encoding하는 위치가 다름**
>    CNN Backbone으로 뽑아낸 feature matrix d×HW에는 위치 정보가 소실되어있습니다.
>    기존의 Transformer도 이와 같은 문제점을 해결하기 위해 Positional encoding을 더해준다.
>    DETR도 마찬가지로 Positional encoding을 더해주는데 위치가 살짝 다르다.![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FpC6Jj%2Fbtq4J8mtzwj%2FYUrKHhrg9Ipvgd8LwwMv7K%2Fimg.png)
> 2. **Autoregression이 아닌 Parallel 방식으로 output을 출력함**
>    기존 Transformer는 단어 한 개씩 순차적으로 출력값을 내놓습니다.
>    Autoregression은 현재 output 값을 출력하기 위해 이전 단계까지 출력한 output 값을 참고하는 방식입니다.
>    반면 DETR에서 사용한 Transformer는 Parallel 방식으로, 즉 **모든 output 값을 통채로 출력**하는 방식입니다.

---

### 2-3) FFN (Feed Forward Network)

- Transformer의 결과로 나온 **N개의 unit**은 FFN을 통과하여 **class와 bounding box의 크기와 위치**를 동시에 예측합니다.
  이때 **bipartite matching**을 통해 각 bounding box가 겹치지 않도록 합니다. (이에 대한 자세한 설명은 loss function & Training에서)
  - ***Bipartite matching (이분 매칭)**
    - **DETR은 충분히 큰 수의 Bounding box를 N개 설정하고 이에 대해서 class와 bounding box의 크기 및 위치를 예측합니다.**
      Input image 상에 2개의 object만 존재한다면 2개의 bounding box에 대해서는 class, bounding box의 크기 및 위치를 예측하고, 나머지 2개에 대해서는 no object를 출력하게 됩니다.
    - 다만 DETR은 Autoregression이 아닌 Parallel 방식으로 output을 출력하기 때문에 N개의 Bouding box가 동시에 출력합니다.
      **즉 N개의 bounding box가 어떤 ground-truth object를 검출하고 있는지 알 수 없는 문제가 발생합니다.**
    - 만약 그림에서 분홍색 boudning box가 1번 갈매기에 대한 bouding box였다면 loss 값이 작겠지만, 만약 2번 갈메기에 대한 bouding box 였다면 loss 값이 크게 됩니다.
      띠라서 **bounding box가** ground truth의 **어떤 object를 검출하고 있는지 1대1로 매칭**을 해주는 과정이 필요하며 이를 **bipartite matching**이라고 합니다.

---

### 2-4) Loss Function & Training

#### 2-4-0) 전체 흐름 한눈에 보기

DETR의 손실은 한 번에 계산되지 않고 **"짝을 먼저 정하고, 그 다음 점수를 매기는"** 2단계로 나뉩니다.

```
   GT M개 (갈매기1, 갈매기2)
        │
        │   N개 예측 (pred1 ~ pred4)
        ▼
 ┌──────────────────────────────────────────────────────────┐
 │ STEP 1.  비용 행렬(cost matrix) 계산                       │
 │          "GT × 예측" 모든 조합의 안 맞는 정도를 표로 만든다    │
 └──────────────────────────────────────────────────────────┘
        ▼
 ┌──────────────────────────────────────────────────────────┐
 │ STEP 2.  Bipartite Matching (헝가리안 알고리즘)             │
 │          총 비용이 최소가 되는 1:1 짝을 확정한다              │
 │          → 미분 안 함 (no_grad). 짝만 정하는 단계            │
 └──────────────────────────────────────────────────────────┘
        ▼
 ┌──────────────────────────────────────────────────────────┐
 │ STEP 3.  Hungarian Loss                                  │
 │          확정된 짝에만 (분류손실 + 박스손실)을 계산해 역전파    │
 │          → 미분 함. 실제로 학습되는 단계                     │
 └──────────────────────────────────────────────────────────┘
```

| 단계             | 이름               | 하는 일                              | 미분(역전파) |
| ---------------- | ------------------ | ------------------------------------ | :----------: |
| **STEP 1** | 비용 행렬          | 모든 (GT, 예측) 쌍의 비용 계산       |      ❌      |
| **STEP 2** | Bipartite Matching | 최적 1:1 대응$\hat\sigma$ 찾기     |      ❌      |
| **STEP 3** | Hungarian Loss     | 확정된 짝의 손실 계산 → 가중치 갱신 |      ✅      |

> **초심자용 비유**
> 시험 답안 4장(예측)과 정답 2장(GT)이 있는데,
> **어느 답안이 어느 문제에 대한 것인지 이름이 안 적혀 있는 상황**입니다.
> 채점(손실 계산)을 하기 전에 먼저 **"이 답안은 1번 문제 것"** 이라고 짝을 지어줘야 합니다.
> 그 짝짓기가 STEP 2, 채점이 STEP 3입니다.

---

#### 2-4-1) 기호 정리

| 기호                                                                                                      | 의미                                                                     |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| $N$                                                                                                     | 예측(object query) 개수.**고정값**, 논문은 $N = 100$             |
| $M$                                      | 이미지 안의 실제 GT 객체 개수. 이미지마다 다름 ($M \le N$) |                                                                          |
| $y = \{ y_i \}_{i=1}^{N}$                                                                               | GT 집합. 실제 객체$M$ 개 + $\varnothing$ 패딩 $(N-M)$ 개           |
| $y_i = (c_i,\ b_i)$                                                                                     | $i$ 번째 GT = (클래스 라벨, 박스)                                      |
| $b_i \in [0,1]^4$                                                                                       | 이미지 크기로 정규화한$(c_x, c_y, w, h)$ = 중심좌표 + 폭/높이          |
| $\varnothing$                                                                                           | "객체 없음(배경)" 이라는**특별한 클래스**                          |
| $\hat{y} = \{ \hat{y}_i \}_{i=1}^{N}$                                                                   | 모델의 예측 집합.$\hat{}$(hat)은 "예측값"이라는 표시                   |
| $\hat{p}_{\sigma(i)}(c_i)$                                                                              | $\sigma(i)$ 번째 예측이 정답 클래스 $c_i$ 에 준 **확률** (0~1) |
| $\hat{b}_{\sigma(i)}$                                                                                   | $\sigma(i)$ 번째 예측 박스                                             |
| $\sigma$                                                                                                | 하나의 짝짓기 방법 (GT 번호 → 예측 번호로 보내는 함수)                  |
| $\mathfrak{S}_N$                                                                                        | 가능한 모든 짝짓기 방법의 집합 (총$N!$ 가지)                           |
| $\hat{\sigma}$                                                                                          | 그중**비용이 최소인 최적 짝짓기** ← 우리가 찾는 것                |
| $\mathbb{1}_{\{c_i \neq \varnothing\}}$                                                                 | 조건이 참이면 1, 거짓이면 0. 즉**실제 객체일 때만 1**              |

> $\sigma(i) = 3$ 이라는 표기는 **"$i$ 번째 GT는 3번 예측과 짝"** 이라는 뜻입니다.

---

#### 2-4-2) STEP 1 : 비용 행렬 (Matching Cost)

먼저 **"GT $i$ 와 예측 $j$ 가 얼마나 안 어울리는가"** 를 숫자 하나로 만듭니다. 이 값이 **매칭 비용**입니다.

$$
\mathcal{L}_{\text{match}}\big(y_i,\ \hat{y}_{\sigma(i)}\big)
\;=\;
\underbrace{-\,\mathbb{1}_{\{c_i \neq \varnothing\}}\ \hat{p}_{\sigma(i)}(c_i)}_{\text{① 분류 항}}
\;+\;
\underbrace{\mathbb{1}_{\{c_i \neq \varnothing\}}\ \mathcal{L}_{\text{box}}\big(b_i,\ \hat{b}_{\sigma(i)}\big)}_{\text{② 위치 항}}
$$

**말로 풀면** : `비용 = -(정답 클래스일 확률) + (박스가 어긋난 정도)`

| 항         | 부호 | 의미                                                                                  |
| ---------- | :---: | ------------------------------------------------------------------------------------- |
| ① 분류 항 | $-$ | 정답 클래스에 높은 확률을 줄수록**비용이 내려간다**(좋은 짝) → 그래서 마이너스 |
| ② 위치 항 | $+$ | 박스가 GT와 어긋날수록**비용이 올라간다**(나쁜 짝)                              |

**$\mathbb{1}_{\{c_i \neq \varnothing\}}$ 의 역할**
GT가 패딩 $\varnothing$ 이면 두 항이 모두 0이 되어 **비용이 상수 0**입니다.
→ 매칭 결과를 결정하는 것은 **실제 객체 $M$ 개뿐**이고, 남는 예측들은 자동으로 $\varnothing$ 에 배정됩니다.

> ⚠️ **가장 헷갈리는 포인트 : 여기서는 $-\log \hat p$ 가 아니라 $-\hat p$ 를 쓴다**
>
> - $-\log \hat p$ 는 확률이 0에 가까우면 값이 **무한대로 커집니다.** 반면 박스 항은 대략 $0 \sim 2$ 정도입니다.
>   → 스케일이 안 맞아서 **두 항을 더해 비교하는 것이 무의미**해집니다.
> - $\hat p \in [0,1]$ 은 박스 항과 크기가 비슷해서 더해서 비교하기 적절합니다. (논문 표현 : *commensurable*)
> - 반대로 **STEP 3의 실제 손실에서는 그래디언트가 잘 흐르도록 $-\log \hat p$ 를 씁니다.**
> - 정리 : **"짝 고를 때 쓰는 비용" ≠ "학습할 때 쓰는 손실"**

---

#### 2-4-3) STEP 2 : Bipartite Matching (이분 매칭)

##### ① "이분 그래프" 라는 말의 뜻

**이분 그래프(Bipartite Graph)** = 정점을 **서로 겹치지 않는 두 집합**으로 나눌 수 있고, **모든 간선이 두 집합 사이만** 연결하는 그래프. (같은 집합 안끼리는 연결 없음)

```
   GT 쪽                     예측 쪽
 ┌──────────┐             ┌──────────┐
 │ 갈매기1  ─┼──── 비용  ───┼→ pred1   │    간선의 가중치 = L_match (안 맞는 정도)
 │ 갈매기2  ─┼──── 비용  ───┼→ pred2   │
 │   ∅     ─┼─────────────┼→ pred3   │    제약 : GT 하나 ↔ 예측 하나
 │   ∅     ─┼─────────────┼→ pred4   │           (one-to-one, 1:1)
 └──────────┘             └──────────┘
```

이렇게 **양쪽을 총 비용이 최소가 되도록 1:1로 짝짓는 문제**를 조합최적화에서 **Assignment Problem(할당 문제)** 이라고 부릅니다.

##### ② 최적 짝짓기의 정의

$$
\hat{\sigma} \;=\; \underset{\sigma \in \mathfrak{S}_N}{\arg\min} \; \sum_{i=1}^{N} \mathcal{L}_{\text{match}}\big(y_i,\ \hat{y}_{\sigma(i)}\big)
$$

**말로 풀면** : "가능한 모든 짝짓기 방법 $\sigma$ 중에서, **비용의 총합**이 가장 작아지는 방법 $\hat\sigma$ 를 골라라."

- $\arg\min$ : 최솟값 자체가 아니라, **최솟값을 만드는 입력**($\sigma$)을 돌려줍니다.
- 주의 : 개별 쌍의 비용이 아니라 **총합(sum)** 을 최소화합니다. → 다음 예시가 핵심입니다.

##### ③ 예시로 확인 (GT 2개 = 갈매기 2마리, 예측 4개)

비용 행렬 $\mathcal{L}_{\text{match}}$ (**작을수록 좋은 짝**):

|                   | pred1           | pred2 | pred3 | pred4 |
| ----------------- | --------------- | ----- | ----- | ----- |
| **갈매기1** | **-0.90** | -0.85 | +0.30 | +0.50 |
| **갈매기2** | -0.80           | +0.10 | +0.40 | +0.60 |

칸 하나가 어떻게 계산됐는지 예 : (갈매기1, pred2) 는 클래스 확률 $\hat p(\text{bird}) = 0.95$, 박스 손실 $0.10$
→ $-0.95 + 0.10 = -0.85$

이제 짝짓기 두 가지를 비교해 봅니다.

| 방식                             | 배정                                                                                                                        | 총 비용   |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------- |
| **탐욕적(greedy)**         | 갈매기1 → pred1 ($-0.90$, 자기 최선을 선점)<br /> 갈매기2 → pred2 ($+0.10$, 남은 것 중 최선)                          | $-0.80$ |
| **헝가리안(전역 최적)** ✅ | 갈매기1 → pred2 ($-0.85$)<br /> 갈매기2 → pred1 ($-0.80$)                                       | **$-1.65$** |           |

→ 갈매기1이 **자기 1등(pred1)을 조금 양보**하면 전체 합이 훨씬 좋아집니다. **개별 최선(greedy)이 전체 최선이 아니기 때문에** 헝가리안 알고리즘이 필요합니다. 남은 pred3, pred4 는 $\varnothing$ 에 배정되어 **"배경(no object)으로 예측하라"** 고 학습됩니다.

##### ④ 어떻게 계산하는가 : 헝가리안 알고리즘

- 모든 순열을 다 시도하면 $N!$ 가지 → $N = 100$ 이면 계산 불가능
- **헝가리안 알고리즘(Hungarian / Kuhn–Munkres)** 은 $O(N^3)$ 에 **전역 최적해**를 보장
- 실제 구현 : $M \times N$ 비용 행렬을 만들고 `scipy.optimize.linear_sum_assignment(cost_matrix)` 호출 (DETR 공식 코드도 동일)
- 이 전체 과정은 `torch.no_grad()` 안에서 실행 → **매칭은 미분되지 않는 이산(discrete) 연산**

##### ⑤ 1:1 매칭이 주는 효과 — NMS가 사라지는 이유

- 하나의 GT에는 **오직 하나의 예측**만 정답(positive)이 됩니다.
- 같은 객체를 가리키는 나머지 예측들은 전부 $\varnothing$ 으로 학습되므로, **중복 박스가 손실 단계에서 이미 억제**됩니다.
  → **후처리 NMS 불필요, Anchor 불필요** = DETR이 말하는 "end-to-end"의 핵심
- **단점** : 학습 초기에는 매칭 결과가 에폭마다 요동쳐(unstable assignment) **수렴이 매우 느립니다** (논문 기준 300~500 epoch).
  → 이 수렴 속도 문제를 다룬 것이 이후의 후속 연구들입니다.

---

#### 2-4-4) STEP 3 : Hungarian Loss (최종 손실)

$\hat\sigma$ 가 확정되었으니, 이제 **그 짝에 대해서만** 실제 손실을 계산해 역전파합니다.

$$
\mathcal{L}_{\text{Hungarian}}(y, \hat{y})
\;=\;
\sum_{i=1}^{N}
\Big[
\underbrace{-\log \hat{p}_{\hat{\sigma}(i)}(c_i)}_{\text{분류 손실}}
\;+\;
\underbrace{\mathbb{1}_{\{c_i \neq \varnothing\}}\ \mathcal{L}_{\text{box}}\big(b_i,\ \hat{b}_{\hat{\sigma}(i)}\big)}_{\text{박스 손실}}
\Big]
$$

**말로 풀면** : `손실 = Σ [ 분류 손실(모든 짝) + 박스 손실(실제 객체 짝만) ]`

| 항        | 적용 대상                                      | 이유                                                 |
| --------- | ---------------------------------------------- | ---------------------------------------------------- |
| 분류 손실 | **$N$ 개 전부** ($\varnothing$ 포함) | 남는 예측에게 "너는 배경이다"라고 가르쳐야 하기 때문 |
| 박스 손실 | **실제 객체 $M$ 개만**                 | 배경에는 맞춰야 할 좌표가 없기 때문                  |

> **클래스 불균형 문제와 그 해결**
> $N = 100$, $M \approx 2\sim7$ 이므로 $\varnothing$ 짝이 압도적으로 많습니다.
> 그대로 두면 **"전부 배경"이라고 답하는 것이 손실상 유리**해집니다.
> → DETR은 $\varnothing$ 클래스의 log-prob 항에 **가중치 $\tfrac{1}{10}$** 을 곱해 배경의 영향력을 줄입니다.

---

#### 2-4-5) 박스 손실 $\mathcal{L}$

$$
\mathcal{L}_{\text{box}}(b_i,\ \hat{b}_{\hat\sigma(i)})
\;=\;
\lambda_{L1}\,\big\lVert b_i - \hat{b}_{\hat\sigma(i)} \big\rVert_1
\;+\;
\lambda_{\text{giou}}\,\mathcal{L}_{\text{giou}}\big(b_i, \hat{b}_{\hat\sigma(i)}\big)
$$

논문 하이퍼파라미터 : $\lambda_{L1} = 5$, $\lambda_{\text{giou}} = 2$

##### ① 왜 두 개를 섞어 쓰는가

| 항      | 성질                                           | 단독 사용 시 문제                                                                   |
| ------- | ---------------------------------------------- | ----------------------------------------------------------------------------------- |
| $L_1$ | 좌표 차이를 직접 벌점 → 최적화가 안정적       | **스케일 의존적**. 같은 상대 오차라도 **큰 박스가 훨씬 큰 손실**을 받음 |
| GIoU    | **스케일 불변**, 겹침 품질을 직접 최적화 | 두 박스가 아예 안 겹치면 신호가 약하고, 단독으로는 수렴이 불안정                    |

→ 서로의 약점을 보완하므로 **선형 결합**해서 씁니다.
(예: 10px 오차는 작은 박스에는 치명적이지만 큰 박스에는 사소합니다. $L_1$ 만 쓰면 큰 박스만 열심히 맞추게 됩니다.)

##### ② GIoU (Generalized IoU)

GT 박스 $A$, 예측 박스 $B$, 그리고 **둘을 모두 감싸는 최소 사각형** $C$ 에 대해

$$
\text{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|},
\qquad
\text{GIoU}(A,B) = \text{IoU}(A,B) \;-\; \frac{\big|C \setminus (A \cup B)\big|}{|C|}
$$

$$
\mathcal{L}_{\text{giou}}(A,B) = 1 - \text{GIoU}(A,B)
$$

```
     ┌────────────────── C (둘을 감싸는 최소 사각형) ─────┐
     │  ┌──────┐                                       │
     │  │  A   │      ← 빈 공간(C \ (A∪B))이 넓을수록    │
     │  └──────┘         GIoU가 작아진다                │
     │              ┌──────┐                           │
     │              │  B   │                           │
     │              └──────┘                           │
     └─────────────────────────────────────────────────┘
```

- 값 범위 : $\text{GIoU} \in (-1,\ 1]$, 따라서 $\mathcal{L}_{\text{giou}} \in [0,\ 2)$
- 두 번째 항은 **"$C$ 안에서 두 박스가 채우지 못한 빈 공간의 비율"** 에 대한 벌점입니다.
- **IoU만 쓰면 안 되는 이유** : 두 박스가 겹치지 않으면 IoU는 **항상 0** 이라 "조금 빗나감"과 "완전히 딴 곳"을 구분할 수 없고 그래디언트도 0입니다.
  GIoU는 멀어질수록 빈 공간 비율이 커져 값이 $-1$ 쪽으로 내려가므로, **안 겹치는 상황에서도 "가까워지라"는 학습 신호가 살아 있습니다.**

---

#### 2-4-6) Auxiliary Decoding Loss (보조 손실)

디코더 **마지막 레이어에만** 손실을 걸면 깊은 디코더의 학습이 느려집니다. 그래서 DETR은

- 6개 디코더 레이어 **각각의 출력**에 대해 (매칭 + Hungarian Loss)를 계산해 **모두 더합니다.**
- 각 레이어의 예측 헤드(FFN)는 **파라미터를 공유**하고, 입력 전에 공용 LayerNorm을 통과시켜 스케일을 맞춥니다.

효과
① 중간 레이어가 직접 감독을 받아 **수렴 가속**
② **각 레이어가 이미 올바른 객체 개수를 출력**하도록 유도

$$
\mathcal{L}_{\text{total}} = \sum_{d=1}^{6} \mathcal{L}_{\text{Hungarian}}^{(d)}
$$

---

#### 2-4-7) Training 설정 (논문 기준)

| 항목              | 값                                                                          |
| ----------------- | --------------------------------------------------------------------------- |
| Optimizer         | AdamW                                                                       |
| Learning rate     | Transformer$10^{-4}$ / **Backbone $10^{-5}$** (백본은 더 작게)    |
| Weight decay      | $10^{-4}$                                                                 |
| Gradient clipping | max norm$0.1$                                                             |
| Schedule          | 300 epoch (200 epoch에서 lr$\times \tfrac{1}{10}$), 장기 학습은 500 epoch |
| Dropout           | Transformer 내부$0.1$                                                     |
| 쿼리 개수$N$    | 100                                                                         |
| Augmentation      | Scale augmentation (짧은 변 480~800, 긴 변$\le 1333$), random crop        |
| 학습 환경         | 16× V100, 300 epoch 학습에 약 3일                                          |

> 백본의 learning rate를 10배 작게 두는 이유 : ImageNet으로 **이미 잘 학습된 CNN**이므로 크게 흔들면 오히려 성능이 나빠집니다. 반면 Transformer는 **처음부터 학습(from scratch)** 하므로 더 큰 lr이 필요합니다.

---

#### 2-4-8) 자주 헷갈리는 3가지 (정리)

| 질문                                                   | 답                                                                                                                                                   |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **매칭 비용과 손실 함수는 같은 것인가?**         | **다릅니다.** 매칭 비용은 확률 $-\hat p$ 를 쓰고 미분하지 않습니다. 손실은 $-\log \hat p$ 를 쓰고 역전파합니다.                            |
| **헝가리안 알고리즘은 학습되는(미분되는) 건가?** | 아닙니다.`no_grad` 안의 이산 최적화이며, **"어떤 예측이 어떤 GT의 정답인지"를 정해주는 라벨 배정기** 역할만 합니다.                          |
| **왜 NMS가 필요 없나?**                          | 1:1 매칭 때문에 한 객체당 정답 예측이 딱 하나이고, 나머지 중복 예측은 $\varnothing$ 으로 학습되어 **손실 단계에서 이미 중복이 제거**됩니다. |

---

> DETR의 손실은 **"① 비용 행렬을 만들고 → ② 헝가리안 알고리즘으로 GT↔예측을 1:1 최적 매칭(미분 X) → ③ 확정된 짝에만 분류 손실 + $L_1$ + GIoU를 부과(미분 O)"** 하는 2단계 구조입니다.

> 이 **1:1 제약**이 중복 예측을 손실 차원에서 억제해 **Anchor와 NMS를 제거**했고, 그 대가로 **수렴이 매우 느려지는 한계**(300~500 epoch)를 갖습니다.
