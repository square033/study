> 이 문서는 [DETR.md](DETR.md) 를 이미 읽었다는 전제로, **DETR과 비교해 무엇이 바뀌는지** 위주로 정리합니다.
> Bipartite Matching, Hungarian Loss의 기본 구조는 [DETR.md (2-4)](DETR.md) 와 동일하므로 여기서는 **달라진 부분만** 자세히 다룹니다.

참고 : [lcyking.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Deformable-DETR](https://lcyking.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Deformable-DETR)

## (1) 무엇이 문제였나 — DETR의 한계

### 1-1) 문제 1 : 느린 수렴 (Slow Convergence)

- DETR은 학습에 **300~500 epoch** 이 필요합니다. (일반적인 CNN 검출기는 12~36 epoch)
- 원인 : object query가 이미지의 **어디를 봐야 하는지 처음엔 전혀 모릅니다.** Self/Cross-Attention이 **모든 위치를 균등하게(dense)** 보다가, 학습이 한참 진행돼야 비로소 "의미 있는 소수의 위치"에 집중하는 attention map으로 서서히 좁혀집니다. 이 좁혀지는 과정 자체가 매우 느립니다.

### 1-2) 문제 2 : 작은 물체(Small Object) 성능 저하

- DETR은 CNN backbone의 **마지막 feature map 1장(C5, stride 32)만** 사용합니다. → **작은 물체는 이 저해상도 feature map에서 정보가 거의 뭉개짐.**
- 작은 물체를 잘 잡으려면 **고해상도 / 멀티스케일(C3, C4, C5, …) feature map**이 필요합니다.

### 1-3) 두 문제의 공통 원인 : Dense Attention의 계산 복잡도

DETR이 고해상도·멀티스케일을 못 쓴 이유는 성능이 아니라 **계산량** 때문입니다.

$$
\text{Self/Cross-Attention 복잡도} = O(N_q \cdot N_k \cdot C)
$$

| 기호    | 의미                                     |
| ------- | ---------------------------------------- |
| $N_q$ | Query 개수                               |
| $N_k$ | Key 개수 (= feature map의 픽셀 수$HW$) |
| $C$   | 채널(feature dimension)                  |

**말로 풀면** : encoder self-attention은 query와 key가 둘 다 feature map 전체($HW$)이므로 비용이 $O((HW)^2 C)$ — **이미지 한 변이 2배 커지면 계산량은 16배**가 됩니다.

> 그래서 DETR은 어쩔 수 없이 **저해상도 feature map 1장**만 쓴 것이고, 이것이 문제 1(느린 수렴)과 문제 2(작은 물체) 둘 다의 근본 원인입니다.

---

## (2) 핵심 아이디어 : Deformable Attention Module

### 2-1) 한 줄 요약

> **"모든 픽셀을 다 보지 말고, query 근처의 소수 점 $K$개만 골라서 보자."![](https://blog.kakaocdn.net/dna/0x66y/btsF87LwQqy/AAAAAAAAAAAAAAAAAAAAAPoBUEUB-Q771hsGebDuPAtMmEGtJufXn5QnKx67Nc5O/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1790780399&allow_ip=&allow_referer=&signature=YnuD9ohpRU%2B3ZolMwZ8%2FobOEg6s%3D)
> ![](https://blog.kakaocdn.net/dna/BCsnD/btsGaDWXkXX/AAAAAAAAAAAAAAAAAAAAAOqo3zauw_fdzI1tFtXgjaiAMF9oCtXVJiMovJQE4ukB/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1790780399&allow_ip=&allow_referer=&signature=eeZiG0d0Qv0pV7thg2%2Fv8Xa%2FCxM%3D)**

Dense self-attention은 각 query가 **feature map 전체($HW$개)** 를 key로 참조합니다.
Deformable Attention은 각 query가 **자기 주변의 $K$개(논문 $K=4$) 샘플링 포인트만** 참조합니다. ($K \ll HW$)

```
[ 기존 Self-Attention (Dense) ]              [ Deformable Attention (Sparse) ]

  Query q 하나가                                Query q 하나가
  feature map 전체를 참조                        reference point p_q 주변
                                                 K개 점만 참조

  ┌─────────────────────┐                     ┌─────────────────────┐
  │ ● ● ● ● ● ● ● ● ● ● │                     │ · · · · · · · · · · │
  │ ● ● ● ● ● ● ● ● ● ● │  모든 ●를 Key로 사용  │ · · · ●   · ● · · · │  p_q 주변 4개(●)만
  │ ● ● ● ● q ● ● ● ● ● │  → HW번 연산         │ · · · ● q ● · · · · │  Key로 사용 → K번 연산
  │ ● ● ● ● ● ● ● ● ● ● │                     │ · · · · · ● · · · · │
  └─────────────────────┘                     └─────────────────────┘
```

- **아이디어의 출처** : CNN의 **Deformable Convolution**(고정된 3×3 격자가 아니라, 학습된 offset만큼 이동한 위치에서 샘플링)을 attention에 적용한 것입니다.
- $K$개 점의 **위치(offset)** 와 **중요도(가중치)** 둘 다 **query 자신으로부터 학습**되어 예측됩니다. 즉 "어디를 볼지"와 "얼마나 집중할지"를 network가 직접 정합니다.

### 2-2) 수식

$$
\text{DeformAttn}(z_q,\ p_q,\ x) \;=\; \sum_{m=1}^{M} W_m \left[ \sum_{k=1}^{K} A_{mqk} \cdot W_m' \, x\big(p_q + \Delta p_{mqk}\big) \right]
$$

| 기호                | 의미                                                                                               |
| ------------------- | -------------------------------------------------------------------------------------------------- |
| $z_q$             | query$q$ 의 feature 벡터                                                                         |
| $p_q$             | query$q$ 의 **reference point** (2D 좌표, $[0,1]^2$ 로 정규화)                           |
| $x$               | 참조 대상 feature map (encoder에선 자기 자신, decoder에선 encoder 출력)                            |
| $M$               | attention head 개수 (논문$M = 8$, 표준 multi-head attention과 동일)                              |
| $K$               | head당 샘플링 포인트 개수 (논문$K = 4$, **고정된 소수**)                                   |
| $\Delta p_{mqk}$  | $m$ 번째 head, $k$ 번째 샘플의 **위치 오프셋**. $z_q$ 로부터 **선형층으로 예측** |
| $A_{mqk}$         | 그 샘플의**attention 가중치**. $z_q$ 로부터 예측 후 $\sum_k A_{mqk}=1$ 로 softmax 정규화 |
| $x(p_q+\Delta p)$ | 그 좌표의 feature 값. 좌표가 정수가 아니므로**bilinear interpolation**으로 값을 얻음         |

**말로 풀면** : "query $q$ 는 자기 위치 $p_q$ 근처의 $K$개 지점을 (network가 정한 오프셋만큼 이동해서) 뽑고, 그 $K$개 값을 (network가 정한 가중치로) 가중합한다." — 표준 attention의 "유사도로 가중치를 계산"하는 과정 자체가 **"query에서 직접 오프셋과 가중치를 예측"** 하는 것으로 바뀝니다.

### 2-3) 복잡도 비교

| 방식                      | 복잡도                                                                               | $N=100$(query), $HW=850$(feature map 픽셀 수) 기준 상대 비용 |
| ------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| 표준 Self/Cross-Attention | $O(N_q \cdot HW \cdot C)$                                                          | $HW$ 에 비례 (feature map이 커지면 그만큼 늘어남)              |
| Deformable Attention      | $O(N_q \cdot K \cdot C)$     | **$K=4$ 로 고정** (feature map 크기와 무관) |                                                                  |

→ **feature map이 아무리 커져도 (해상도가 높아지든 멀티스케일이든) 비용이 늘지 않습니다.** 이 덕분에 **고해상도 + 멀티스케일 feature map**을 처음으로 쓸 수 있게 됩니다.

---

## (3) Multi-scale Deformable Attention

### 3-1) 왜 멀티스케일이 필요한가

- 작은 물체는 고해상도(예: stride 8) feature map에서 잘 보이고, 큰 물체는 저해상도(stride 32) feature map에서 충분합니다.
- Deformable Attention이 저렴해졌으므로, 이제 CNN backbone의 **여러 단계 feature map을 동시에** 씁니다.
- 논문 설정 : backbone에서 뽑은 **C3, C4, C5** + C5에 stride-2 conv를 하나 더 적용한 **추가 저해상도 레벨**, 총 **$L=4$ 개 레벨**.

### 3-2) 멀티스케일로 확장한 수식

$$
\text{MSDeformAttn}(z_q,\ \hat p_q,\ \{x^l\}_{l=1}^{L}) \;=\; \sum_{m=1}^{M} W_m \left[ \sum_{l=1}^{L} \sum_{k=1}^{K} A_{mlqk} \cdot W_m' \, x^l\big(\phi_l(\hat p_q) + \Delta p_{mlqk}\big) \right]
$$

- $\hat p_q$ : 레벨에 무관한 **정규화 좌표** $[0,1]^2$
- $\phi_l(\cdot)$ : 그 좌표를 $l$ 번째 레벨의 실제 픽셀 좌표로 변환하는 함수
- $A_{mlqk}$ : 이제 $L \times K$ 개(4레벨 × 4포인트 = 16개) 샘플에 대해 **합이 1**이 되도록 정규화

**말로 풀면** : "각 query는 4개 레벨 각각에서 4개씩, 총 16개 지점을 보고, 그 16개를 가중합한다." — 표준 attention이 $HW$개를 보던 것을, **레벨 수 × 4 개(작은 고정값)** 로 줄인 것입니다.

### 3-3) Encoder에 적용 — Self-Attention 자체를 교체

| 항목              | DETR                                         | Deformable DETR                                                        |
| ----------------- | -------------------------------------------- | ---------------------------------------------------------------------- |
| Encoder 입력      | feature map**1장** ($d \times HW$)   | feature map **4장** (멀티스케일)                                |
| Encoder attention | 표준 Self-Attention (모든 픽셀 ↔ 모든 픽셀) | **Multi-scale Deformable Self-Attention**                        |
| Query = Key       | 둘 다 feature map 픽셀 (dense)               | 둘 다 feature map 픽셀이지만,**각 픽셀당 $K$개만 sparse 참조** |
| reference point   | (해당 없음)                                  | **자기 자신의 픽셀 좌표**를 그대로 사용                          |

### 3-4) Decoder에 적용 — Cross-Attention만 교체 (Self-Attention은 그대로)

| 항목                                          | DETR                                 | Deformable DETR                                                                           |
| --------------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------- |
| Decoder Self-Attention (query 간)             | 표준 (query 100개끼리)               | **동일하게 표준 유지** — $N$이 작아 비용 문제 없음                               |
| Decoder Cross-Attention (query→encoder 출력) | 표준 (query 1개가$HW$ 전체를 참조) | **Deformable Cross-Attention** (query 1개가 멀티스케일에서 $L\times K$ 개만 참조) |
| reference point                               | (해당 없음)                          | object query embedding으로부터**선형층 + sigmoid**로 예측                           |

> **주의** : Deformable DETR이 바꾸는 것은 **"Cross-Attention이 무엇을 참조하는가"** 뿐입니다. object query끼리의 self-attention(경쟁/중복 억제 역할)은 DETR과 동일하게 표준 attention을 씁니다. query 개수가 적어(수백 개) 여기는 비용 문제가 없기 때문입니다.

---

## (4) Bounding Box 표현의 변화 : Reference Point + 상대 오프셋

DETR은 FFN이 **박스의 절대 좌표 $(c_x, c_y, w, h)$ 를 처음부터 끝까지** 직접 예측했습니다.
Deformable DETR은 decoder가 이미 **reference point $p_q$** 를 갖고 있으므로, 이를 기준으로 한 **상대 보정값**만 예측합니다.

$$
\hat b = \sigma\big( \Delta b + \sigma^{-1}(p_q) \big) \qquad (\sigma = \text{sigmoid})
$$

**말로 풀면** : "박스 위치 = 참조점 + 거기서부터 얼마나 옮길지(오프셋)". 참조점이 이미 대략 맞는 위치를 잡아주므로, network는 **미세 조정값만** 학습하면 됩니다 — 이것이 수렴을 더 빠르게 만드는 또 하나의 요인입니다.

---

## (5) Iterative Bounding Box Refinement

- 디코더는 6개 레이어를 쌓는데, DETR은 6개 레이어 **각각이 독립적으로** 최종 박스를 예측합니다 (aux loss만 공유).
- Deformable DETR은 **레이어를 거칠수록 박스를 점점 정밀하게 다듬습니다.**

$$
\hat b^{(d)} = \sigma\Big( \Delta b^{(d)} + \sigma^{-1}\big(\hat b^{(d-1)}\big) \Big)
$$

```
 layer 1 예측 박스 ──(detach)──► layer 2의 reference point로 사용 ──► layer 2가 그 주변을 더 정밀하게 보정
      │                                                                      │
      └── layer 1도 여전히 자기 레이어의 Hungarian Loss로 감독됨 ─────────────────┘
```

- $\hat b^{(d-1)}$ 는 **detach** 해서(그래디언트 차단) 다음 레이어의 참조점으로만 사용합니다. → 각 레이어가 독립적으로 안정적으로 학습되게 함.
- 레이어마다 **독립된 파라미터**로 박스를 예측합니다 (DETR의 공유 FFN과 달리).

---

## (6) Two-Stage Deformable DETR

DETR의 object query는 **완전히 학습된 고정 파라미터**(이미지와 무관하게 항상 같은 100개)로 시작합니다. Two-Stage는 여기에 **1단계 proposal 생성**을 추가합니다.

```
1단계 (Region Proposal)                     2단계 (Refine, 기존 decoder)
─────────────────────────                  ───────────────────────────
Encoder의 각 픽셀 위치에서                    top-k proposal의
"물체가 있을 법한 박스"를 예측                  (위치, feature)를
      │                                     object query의
      ▼                                     (reference point, 초기값)으로 사용
top-k(점수 상위 300개) 선택
```

- Faster R-CNN의 **RPN(Region Proposal Network)** 과 비슷한 역할을 encoder가 대신합니다.
- 이 proposal 생성 단계에도 **동일한 방식의 Bipartite Matching + Loss**가 별도로 걸립니다 (클래스는 물체/배경 이진).
- 효과 : 랜덤한 고정 쿼리보다 **이미지 내용에 맞춘 좋은 시작점**을 주므로 성능이 추가로 향상됩니다.

---

## (7) Loss Function에서 달라지는 것

### 7-1) 그대로인 것 : Bipartite Matching 그 자체

> [DETR.md (2-4)](DETR.md) 의 **비용 행렬 → 헝가리안 알고리즘 → 1:1 매칭** 구조는 **완전히 동일합니다.**
> 바뀌는 것은 "매칭 이후 손실을 어떻게 계산하는가"의 세부 항목들입니다.

### 7-2) 바뀐 것 1 : 분류 손실 — Cross Entropy → **Focal Loss**

$$
\mathcal{L}_{\text{focal}}(p_t) = -\,\alpha_t\,(1 - p_t)^{\gamma}\,\log(p_t), \qquad \alpha = 0.25,\ \ \gamma = 2
$$

| 항목               | DETR                                                    | Deformable DETR                                               |
| ------------------ | ------------------------------------------------------- | ------------------------------------------------------------- |
| 분류 손실          | Cross Entropy +$\varnothing$ 가중치 $\tfrac{1}{10}$ | **Focal Loss**                                          |
| 분류기 구조        | softmax (클래스 간 배타적)                              | **클래스별 sigmoid**                                    |
| 매칭 비용의 분류항 | $-\hat p_{\sigma(i)}(c_i)$                            | **Focal 형태**로 계산 (매칭 비용도 손실과 같은 식을 씀) |

**말로 풀면** : $(1-p_t)^\gamma$ 항이 **이미 잘 맞춘 쉬운 샘플의 기여를 자동으로 줄이고, 어려운(헷갈리는) 샘플에 집중**하게 만듭니다. DETR처럼 "$\varnothing$ 에 $\tfrac{1}{10}$ 가중치"라는 수작업 보정이 필요 없어집니다.

**왜 바뀌었나** : 아래 7-3에서 쿼리 수가 $100 \to 300$ 으로 늘면서 $\varnothing$(배경) 샘플이 더 많아져 클래스 불균형이 심해지는데, 고정 가중치 $\tfrac{1}{10}$ 보다 **샘플별로 자동 조절되는 Focal Loss**가 이 불균형에 더 적합합니다.

### 7-3) 바뀐 것 2 : Query 개수 $N$

| DETR        | Deformable DETR |
| ----------- | --------------- |
| $N = 100$ | $N = 300$     |

Deformable Attention이 저렴해져 query를 늘려도 비용 부담이 적고, query가 많아지면 **recall(놓치는 물체가 줄어드는 것)** 이 좋아집니다.

### 7-4) 바뀐 것 3 : 박스 손실의 파라미터화

$\mathcal{L}_{\text{box}} = \lambda_{L1}\lVert b_i - \hat b \rVert_1 + \lambda_{\text{giou}}\mathcal{L}_{\text{giou}}$ 라는 **손실 식 자체는 동일**($\lambda_{L1}=5,\ \lambda_{\text{giou}}=2$)합니다.
다만 $\hat b$ 가 이제 **(4)에서 설명한 reference point + offset**으로 계산되고, **(5)의 iterative refinement**로 레이어마다 갱신된다는 점이 다릅니다. GIoU 자체의 정의는 그대로입니다.

---

## (8) DETR vs Deformable DETR 한눈에 비교

| 구분                     | DETR                                                                                              | Deformable DETR                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Attention 방식           | Dense (모든 위치 참조)                                                                            | **Deformable** (query 주변 $K$개만 참조)             |
| Feature map              | 1개 (저해상도, stride 32)                                                                         | **멀티스케일 4개** (고해상도 포함)                     |
| Attention 복잡도         | $O(N_q \cdot HW \cdot C)$                                                                       | $O(N_q \cdot K \cdot C)$ — **$HW$ 와 무관**       |
| Encoder Self-Attention   | 표준                                                                                              | Multi-scale Deformable Self-Attention                        |
| Decoder Cross-Attention  | 표준                                                                                              | Deformable Cross-Attention                                   |
| Decoder Self-Attention   | 표준                                                                                              | **표준 그대로 유지**                                   |
| 박스 예측 방식           | 절대 좌표 직접 예측                                                                               | reference point + 상대 오프셋,**레이어마다 반복 정제** |
| Query 개수$N$          | 100                                                                                               | 300                                                          |
| 분류 손실                | Cross Entropy ($\varnothing$ 가중치 1/10) | **Focal Loss** ($\alpha{=}0.25,\gamma{=}2$) |                                                              |
| Query 초기화             | 학습된 고정 파라미터                                                                              | (Two-Stage 옵션)**encoder 기반 region proposal**       |
| Bipartite Matching       | 있음                                                                                              | **동일하게 있음 (변화 없음)**                          |
| 학습 epoch               | 300~500                                                                                           | **약 50** (10배 단축)                                  |
| 작은 물체 성능(AP$_S$) | 상대적으로 낮음                                                                                   | **크게 향상** (멀티스케일 고해상도 feature 덕분)       |

---

## (9) 한 줄 정리

> Deformable DETR은 DETR의 **매칭·손실 구조(Bipartite Matching, Hungarian Loss)를 그대로 둔 채**, DETR이 저해상도 feature map 1장에 갇혀 있던 근본 원인 — **dense attention의 $O((HW)^2)$ 비용** — 을 **"query 주변 $K$개 점만 보는 Deformable Attention"** 으로 해결합니다.
> 그 결과 **멀티스케일·고해상도 feature map**을 처음으로 쓸 수 있게 되어 **작은 물체 성능이 크게 오르고, 수렴 속도가 약 10배 빨라졌으며**(500 → 50 epoch), 그 위에 **Iterative Box Refinement**와 **Two-Stage** 를 얹어 추가로 성능을 끌어올립니다.
