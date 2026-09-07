곰 그림의 특성을 지닌 필터를 찾아내는 것

### 1) RNN (시계열 데이터)

![](https://blog.kakaocdn.net/dna/oc2jL/btsiO44Up2R/AAAAAAAAAAAAAAAAAAAAAKaslohKl95sA006l7WQzBJGQidgNcjLht6dMY-FqvCd/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1790780399&allow_ip=&allow_referer=&signature=2MUMX4hFy95Au1oK0pIlOQr9InY%3D)

$$
h_{t} = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_{t})
$$

$$
y_{t}=W_{hy}h_{t}
$$

- $x_t$ : t시점의 입력값 (예: 문장이면 t번째 단어의 임베딩 벡터, 센서 데이터면 t시점의 측정값)
- $h_{t-1}$ : 이전 시점까지 누적된 은닉상태(기억)
- $h_t$ : 현재 시점의 은닉상태 = 이번 스텝의 출력이자, 다음 시점으로 넘어가는 기억

RNN은 $h_t$ 하나로 기억과 출력을 동시에 처리하다 보니 오래된 정보가 쉽게 사라지는 문제(vanishing gradient)가 있고, 이를 보정한 것이 **LSTM**이다.

---

### 2) LSTM (장기데이터 망각 보정)

> **cell state $C_{t-1}$에 값들을 더하거나 없애면서 $C_t$를 만들고,
> 이 cell state $C_t$를 tanh layer에 태워서 -1과 1 사이의 값을 받은 뒤에, 방금 전에 계산한 sigmoid gate의 output과 곱함으로써 우리가 output으로 보내고자 하는 부분만 내보낼 수 있게 된다.**

---

#### [2-1] 기본 공식

##### 1. $C_{t}$를 계산하기 위해서 $C_{t-1}$에 더하거나 없애는 값들은 아래 2 gate 연산들에 의한다.

- **Forget gate**

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

* **Input gate**

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

$$
\tilde{C} = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

* **★ Cell state update**

$$
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)*C_{t-1} + \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) *\tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t * \tanh(C_t)
$$

---

##### 2. 도출된 $C_{t}$에 $tanh \,\, layer$를 태워서 -1과 1 사이의 값을 받은 뒤에, sigmoid gate의 output ($O_t$)과 곱함으로써 우리가 보내고자 하는 부분만 output($h_t$)으로 내보낼 수 있게 된다.

**Output gate**

$$
h_t = o_t * \tanh(C_t)
$$

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

- $C_t$ : 장기 기억(long-term memory). forget gate로 옛 정보를 얼마나 지울지, input gate로 새 정보를 얼마나 더할지를 결정해서 누적된 값.
- $\tanh(C_t)$ : cell state 값을 -1~1 사이로 눌러서, "지금 시점에 내보낼 수 있는 후보 값"으로 정규화.
- **$o_t$ **: output gate. sigmoid로 계산되어 0~1 사이 값을 가지며, **"$\tanh(C_t)$ 중 어느 부분을 실제로 내보낼지"**를 결정하는 필터 역할.
- $h_t = o_t * \tanh(C_t)$ : 장기 기억 $C_t$ 중 output gate가 허락한 부분만 걸러서 내보낸 것 = 이번 시점의 은닉상태(출력)이자, 다음 시점으로 넘어가는 단기 기억.

즉 $C_t$는 계속 안에 쌓이는 장기 기억이고, $h_t$는 그중 "지금 당장 밖으로 보여줄 부분"만 추려낸 값이다.

---

#### [2-2] $h$, $x$, $W$는 어떻게 초기화되고 학습 중에 변하는가

| 값                                                          | 초기값                                                             | 학습 중 변화                                                                                                                |
| ----------------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| $x_t$                                                     | 데이터셋에서 그대로 주어짐                                         | 학습 중에도 바뀌지 않음 (단, 임베딩을 함께 학습시키는 경우엔 "단어 → 벡터" 임베딩 테이블 자체는 학습 대상)                 |
| $h_0$, $C_0$                                            | 보통 0벡터로 초기화<br />(드물게 학습 가능한 파라미터로 두기도 함) | 파라미터가 아니라, forward pass마다 위 수식으로 매번 새로 계산되는 중간 결과물                                              |
| $W$ (즉 $W_{hh}, W_{xh}, W_f, W_i, W_C, W_o$ 등), $b$ | 랜덤 초기화 (Xavier/Glorot, He 초기화 등)                          | 역전파(BPTT, Backpropagation Through Time)로 gradient를 계산하고 경사하강법(SGD, Adam 등)으로 업데이트되는 실제 "학습 대상" |

정리하면, 학습을 통해 실제로 값이 바뀌는 건 $W$와 $b$뿐이고, $h_t$·$C_t$는 그 학습된 $W$를 이용해 매 입력마다 새로 계산되어 흘러나오는 신호다.

##### [2-2-1] 시퀀스 간 hidden state를 리셋할지 이어받을지 판단 기준

모델이 자동으로 판단하는 것이 아니라, 데이터 구조를 보고 **사람이 미리 정해두는 것**이다.

- **리셋 (h_0 = 0으로 새로 시작)** : 데이터를 서로 독립적인 샘플들의 묶음으로 다룰 때.
  예) 감성분석에서 문장 A와 문장 B는 서로 무관 → 문장마다 h_0=0에서 새로 시작. 미니배치 학습에서 배치 안 시퀀스들은 서로 다른 샘플이므로 배치가 바뀔 때마다 리셋하는 것이 기본값.
- **유지 (이전 h_t를 다음 h_0로 이어받음)** : 데이터가 하나의 긴 연속된 흐름인데 메모리/연산 제약으로 편의상 잘라서 처리할 때. 이를 **stateful RNN** 또는 **TBPTT(Truncated Backpropagation Through Time)** 라 부른다.
- 예) 아주 긴 문서나 센서 스트림을 100 스텝씩 잘라 넣는 경우, 청크 경계에서는 이전 h_t를 이어받고 문서가 바뀔 때만 리셋.

구현 레벨에서는 개발자가 "이 배치가 이전 배치의 연속인지"를 알고 있으므로, 연속이면 h를 (그래디언트 연결은 끊고 값만) detach해서 다음 스텝에 넘기고, 아니면 zeros로 새로 만든다.
Keras의 `stateful=True` LSTM처럼 프레임워크가 옵션을 제공하기도 하지만, 리셋 시점(`model.reset_states()` 호출 시점 등)은 결국 사람이 데이터의 논리적 경계(문서 끝, 화자 교체, 샘플 경계 등)를 보고 코드로 지정한다.

$$
h_2 = \tanh(W_{hh} \cdot h_1 + W_{xh} \cdot x_2)
$$

---

***다시 RNN으로 가서 ...***

---

또한 RNN은 한 input의 토큰으로 한 output의 토큰을 예상하는 것이므로
길이가 다른 시퀀스에서는 예상이 쉽지 않고, 전체적인 문장의 흐름이 아니라 각 단어마다 이전의 Hidden State만 보고 계산하는 Locality의 문제가 있다.

예를 들어 한국어와 영어를 번역하는 RNN모델이라고 가정을 하면,
**한국어와 영어는 문법적으로 위치가 다르고 길이도 다르기 때문에 현실적으로 반영하기에는 어려움이 있다.**

이를 개선하기 위해 제안된 구조가 Encoder-Decoder 구조인 **Seq2 Seq이다.**

![img](https://blog.kakaocdn.net/dna/DYd7B/btsiPivakkP/AAAAAAAAAAAAAAAAAAAAAHo-t1fVTDa-nnPAEGo2MBf_uVTEXWJA2RrQ-Yr-1vAM/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1790780399&allow_ip=&allow_referer=&signature=WAwxP9b1x1OJY%2B9B0%2FIF7F8OdmI%3D)

Encoder에서 입력된 단어의 흐름을 순차적으로 입력받아 이 정보들을 하나의 벡터로 압축한 것을 **Context Vector**라고 한다.

이 압축된 Context Vector를 Decoder로 전송하여 번역된 단어를 한 개씩 순차적으로 출력하는 구조가 **Seq2 Seq이다.**

하지면 시퀀스 길이마다 입력되어야 정보가 다르기 때문에 Context Vector의 크기는 가변적이어야 했고,
반면에 **Context Vector의 크기는 고정**된 크기이므로, Encoder에서 시퀀스의 정보를 압축하는 과정에서도 **병목현상**이 발생하며 성능 하락에 원인이 된다는 제한점이 존재했다.

---

이 틀을 깨버린 것이 Transformer이다.

**기존 이전 Hidden State만 고려하는 Locality의 구조가 아니며** 병목현상이 발생하지 않는 Attention Mechanism을 사용하여 Global 하게 **전체 Sequence의 구조를 파악하여 Memory와 새로운 State-of-the-art의 성능**을 내었다.
기존에 도달하지 못했던 모든 구조를 Attention으로 바꾼 것에 많은 혁신적인 성과를 이루어 낸 것이다.

---

### (3) Transformer

![](https://blog.kakaocdn.net/dna/3na1D/btsiO53Oxk7/AAAAAAAAAAAAAAAAAAAAAMhYAOZDfjLkhqfiL5xpgvjqvQgZ5Z0VTEi9h-2_J1gX/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1790780399&allow_ip=&allow_referer=&signature=pGtd1s%2BFdmOS4iBTS9Oft2L7fnw%3D)

[lcyking.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Attention-is-All-you-need%EC%9D%98-%EC%9D%B4%ED%95%B4](https://lcyking.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Attention-is-All-you-need%EC%9D%98-%EC%9D%B4%ED%95%B4)
