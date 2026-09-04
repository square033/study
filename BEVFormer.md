## 0. 사전조사

**시공간(Spatiotemporal)** : 공간(Space)과 시간(Time)을 결합한 개념 / 특정 위치와 시간에 따라 변화하는 현상이나 데이터를 다루는 용어

- ★ Spatial Cross-Attention
- ★ Temporal Self-Attention
- Vision Transformer **(ViT)에서 패치(Patch) 의 의미** :
  입력 이미지를 트랜스포머(Transformer) 모델이 처리할 수 있는 단어(Token) 단위로 쪼갠 기본 조각

---

* nuScenes 3D 객체 검출 데이터셋에서 **LiDAR 단독(LiDAR-only)** 기반 모델들의 점수는 시대 및 구조에 따라 차이가 있습니다.

- **1**. **LiDAR 기반 모델의 NDS 점수대**
  - **초기/표준 Baseline 모델** (60% ~ 68% NDS)
    - 대표적인 LiDAR 베이스라인인 **PointPillars, SECOND, CenterPoint, TransFusion-L** 등이 이 범위에 해당하며, 약 **60% 초반에서 68% NDS** 수준을 형성합니다.
  - **고성능 LiDAR SOTA 모델** (70% ~ 75%+ NDS)
    - 최신 Transformer 기반 고성능 LiDAR 모델(**VoxelNeXt, FocalFormer3D, NEMOT** 등)은 **70%~75% NDS** 이상의 높은 점수를 기록합니다.
- **2. BEVFormer의 의미**
  - BEVFormer 논문이 발표되던 시기(2022년 ECCV) 기준으로 카메라 전용(Camera-only) 모델들은 **40%대 NDS**에 머무는 경우가 많았습니다.
  - 따라서 **카메라 전용 모델인 BEVFormer가 56.9% NDS**를 달성했다는 것은,
    센서 비용이 비싼 LiDAR를 쓰지 않고 단안/다중 카메라 입력만으로도 초기/표준 LiDAR 기반 모델(60% 내외)의 성능 영역에 육박했음을 뜻합니다.

---

- **ill-posed 문제** : 수학자 자크 아다마르(Jacques Hadamard)가 정의한 Well-posed problem(적절한 문제)의 3가지 조건을 충족하지 못할 때 ill-posed 문제라고 부릅니다. 컴퓨터 비전에서 "해(solution)가 존재하지 않거나, 하나로 정해지지 않거나, 약간의 조건 변화에도 결과가 극도로 크게 달라지는 문제"를 말합니다.

* **Well-posed 문제의 3가지 조건** : 이 중 **하나라도 만족하지 못하면 ill-posed 문제**가 됩니다.

  * **존재성 (Existence):** 해가 적어도 하나 존재해야 한다.
  * **유일성 (Uniqueness):** 해가 단 하나만 존재해야 한다.
  * **안정성 (Stability):** 입력 데이터가 조금 바뀌어도 해가 크게 변하지 않아야 한다.

- 하지만 깊이 정보에 기반하는 **기존 프레임워크**(기존 monocular 카메라 개선을 위해 multi camera를 사용하고 2d->3d 투영을 진행함)**는**
  **3d 정보가 2d 이미지로 변환되면서 유일성이 결여**되게되고, **depth 값이나 distribution에 너무 민감하게 정확도가 반응하며, 에러가 누적된다**는 문제가 있습니다.
  - **유일성 결여 (2D → 3D 변환)** : 3D 공간상의 물체가 2D 카메라 이미지로 투영(Projection)될 때, **깊이(Depth) 정보가 손실**됩니다.
    예를 들어, 화면에 작게 찍힌 물체가 '원래 작은 물체가 가까이 있는 것'인지, '거대한 물체가 아주 멀리 있는 것'인지 2D 사진 한 장만으로는 단하나의 3D 형태/위치로 정할 수 없습니다. 즉, **하나의 2D 관측값에 대응할 수 있는 3D 상태가 무수히 많이 존재(해의 유일성 부재**)하기 때문에 ill-posed 문제가 됩니다.

---

* ill-posed 문제를 풀기 위해 인공지능이나 수학 모델에서는 **추가적인 제약 조건이나 사전 정보(Prior)를 부여**합니다.

  * **BEVFormer의 해결 방식 :**
    단일 2D 이미지의 한계를 극복하기 위해 1. **다중 카메라(Spatial Cross-Attention)정보**와 **2. 시간 흐름에 따른 이전 프레임 정보 (Temporal Self-Attention)** 를
    제약 조건으로 활용하여 3D 위치 및 깊이를 안정적으로 추정합니다.
    *( Temporal Self-Attention : BEVFormer가 데이터를 단순 스태킹하지 않고 RNN처럼 과거 프레임의 정보(History BEV)를 하나만 가볍게 이어서 전달받는 방식 )*
  * **★ 시간 정보가 중요한 역할을 하는 이유 (motion, occluded objects)**
    - 가려진 물체(occluded objects) 추적 : 앞차가 순간적으로 건물의 그늘이나 다른 대형 차에 가려지더라도, 직전 프레임들의 위치 데이터를 기억하고 있다면 "저 뒤에 차가 가려져 있다"고 계속 인식할 수 있습니다.
    - 속도/직진성(motion) 파악 : 사진 한 장만 보면 정지한 차인지 달리는 차인지 알 수 없지만, 연속된 장면(시간 정보)을 보면 물체가 어느 방향으로 얼마나 빠르게 움직이는지 알 수 있습니다.
* 시간 정보를 쓰려고 했던 일부 시도에서는 과거 프레임 이미지나 feature들을 **단순히 차곡차곡 쌓아서(stacking) 한 번에 연산**했습니다.
  → 이렇게 하면 처리해야 할 데이터 양이 기하급수적으로 늘어나 계산량이 극도로 많아지고 GPU 메모리를 엄청나게 잡아먹는 비효율성이 발생했습니다.
* BEVFormer의 차별점 : 모든 과거 프레임을 쌓지 않고, RNN 구조처럼 **압축된 단일 History BEV 상태만 이어받아** 효율성과 성능을 동시에 확보했습니다.
* **BEVFormer 구조 vs 표준 Transformer 구조 비교**

  **공통점**

  > **Add & Norm + Feed Forward** : 각 서브모듈(Attention)) 뒤에 residual connection과 LayerNorm을 붙이고, 그 뒤에 FFN을 두는 기본 골격은 동일합니다.
  >
  > **레이어 반복(stacking)** : 표준 Transformer 인코더가 $N$개 레이어를 쌓듯, BEVFormer도 동일 구조의 인코더 레이어를 6개($\times 6$) 쌓습니다.
  >
  > **어텐션 기반 정보 집약** : 쿼리(Query)가 키/밸류를 참조해 정보를 모으는 어텐션 패러다임 자체는 동일합니다.
  >
  > **쿼리 기반 설계** : 디코더 쿼리처럼, BEVFormer도 미리 정의된 학습 가능한 쿼리(BEV Queries $Q$)를 사용해 출력을 생성합니다.
  >

  **차이점**

  > **어텐션 개수/종류** : 표준 Transformer는 **Self-Attention 1개**인 반면,
  > BEVFormer는 **Temporal Self-Attention**(시간축)과 **Spatial Cross-Attention**(공간/모달리티축) **2개로 분리**되어 있습니다.
  >
  > **어텐션 범위** : 표준 Transformer는 **Dense(모든 토큰 쌍 계산)**인 반면,
  > BEVFormer는 **Deformable Attention**을 사용해 각 BEV 쿼리가 이미지 전체가 아니라 기하학적으로 투영된 소수의 reference point(Hit Views)만 참조하여 연산량을 줄입니다.
  >
  > **쿼리의 의미** : 표준 Transformer의 쿼리는 입력 시퀀스 토큰(혹은 학습된 디코더 쿼리)인 반면,
  > BEVFormer의 쿼리는 **그리드 형태의 BEV 공간 좌표**를 나타내는 구조화된 쿼리입니다($x, y$ 위치가 곧 물리적 공간 위치).
  >
  > **시간 처리** : 표준 Transformer는 기본적으로 병렬/순서 독립적인 반면,
  > BEVFormer는 **History BEV $B_{t-1}$ 을 명시적으로 입력**받아 RNN처럼 이전 프레임 정보를 순차적으로 누적합니다.
  >
  > **Cross-Attention의 대상** : 표준 Transformer의 Cross-Attention은 같은 모달리티(예: 디코더가 인코더 출력을 참조)를 대상으로 하지만,
  > BEVFormer의 Spatial Cross-Attention은 **이종 모달리티 간 연결** (2D 원근 영상 특징 → 3D BEV 공간)로, 카메라 파라미터로 $(x, y) \rightarrow (x', y', z')$ 를 투영해 참조점을 계산합니다.
  >
  > **위치 정보 부여 방식** : 표준 Transformer는 Sinusoidal/학습형 Positional Encoding을 쓰지만,
  > BEVFormer는 카메라 캘리브레이션 기반 **기하학적 투영**으로 위치를 대응시킵니다.
  >

  **한 줄 요약** : BEVFormer는 표준 Transformer 인코더의 "Self-Attention → FFN" 골격은 그대로 유지하되,
  Self-Attention을 시간축(Temporal)과 공간/모달리티축(Spatial Cross)으로 분리하고, 둘 다 Deformable Attention으로 희소화하여 멀티카메라 영상을 3D BEV 공간으로 효율적으로 변환·누적하는 구조입니다.

---

## 1. 주요 개념

전체 파이프라인을 먼저 잡고 가면 이해가 쉽습니다.

> **BEV Queries**(격자 쿼리 준비) → **Spatial Cross-Attention**(공간: 여러 카메라에서 특징 수집) → **Temporal Self-Attention**(시간: 직전 BEV 이어받기) → **BEV Features**(완성) → **Task Head**(검출 / 분할)

### 1.0 기호 정리

본문에서 반복적으로 쓰이는 기호는 다음과 같습니다.

| 기호                                                                                                                                                                                                                        | 의미                                                               |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| $Q \in \mathbb{R}^{H \times W \times C}$                                                                                                                                                                                  | BEV Queries 전체 (격자 형태의 학습 가능한 파라미터)                |
| $Q_p$                                    | BEV 평면 위치 $p = (x, y)$ 를 담당하는 하나의 BEV Query                               | BEV 평면 위치 $p = (x, y)$ 를 담당하는 하나의 BEV Query                          |                                                                    |
| $B_t$                                    | 시점 $t$ 에서 완성된 BEV Feature                                                      | 시점 $t$ 에서 완성된 BEV Feature                                                 |                                                                    |
| $B '_{t-1}$                                                                                                                                                                                                               | Ego motion에 따라 현재 좌표계로 정렬(Alignment)된 직전 BEV Feature |
| $F_t = \{ F_t^{\,i} \}_{i=1}^{N_{view}}$ | 시점 $ t$ 의 멀티카메라 feature map ($i$ 번째 카메라의 feature map이 $F_t^{\,i}$)                                                                                       |                                                                    |
| $\mathcal{V}_{hit}$                                                                                                                                                                                                       | hit view의 집합                                                    |
| $N_{ref}$                                | 하나의 BEV Query가 갖는 3D reference point의 개수 (논문에서는$N_{ref} = 4$)            | 하나의 BEV Query가 갖는 3D reference point의 개수 (논문에서는$N_{ref} = 4$)       |                                                                    |
| $\mathcal{P}(p, i, j)$                   | projection function.$j$ 번째 reference point를 $i$ 번째 카메라 영상 좌표로 변환      | projection function.$j$ 번째 reference point를 $i$ 번째 카메라 영상 좌표로 변환 |                                                                    |
| $s$                                                                                                                                                                                                                       | BEV Grid 한 칸이 나타내는 실제 거리 (meter)                        |
| $H,\ W$                                                                                                                                                                                                                   | BEV Grid의 세로 / 가로 크기                                        |

---

### 1.1 BEV Queries

- **정의** : 격자(grid) 형태의 **학습 가능한(learnable) 파라미터** $Q \in \mathbb{R}^{H \times W \times C}$ 입니다.
- **역할** : 각 쿼리 $Q_p$ 는 BEV 평면의 위치 $p = (x, y)$ 에 해당하는 **격자 한 칸(Grid Cell)** 을 담당하며, 한 칸은 실제 세계에서 $s$ **미터** 크기에 대응합니다.
- **기준점** : BEV의 **중앙은 기본적으로 Ego Vehicle(자기 차량)** 의 위치입니다.
- **입력 전처리** : BEVFormer에 입력하기 전에 **Position Embedding**을 추가합니다.

즉, BEV Query는 "이 위치에 무엇이 있는지 알려 줘"라고 묻는 **공간 좌표 그 자체**라고 할 수 있습니다.

---

### 1.2 Spatial Cross-Attention — 여러 카메라에서 공간 정보 모으기

#### (1) 왜 Deformable Attention을 쓰는가

multi-camera 3D perception을 일반적인 Transformer의 multi-head attention으로 처리하기에는 **계산량이 너무 크다**는 문제가 있습니다.
따라서 여기서 **Deformable Attention** 개념이 등장합니다.

**Deformable Attention**은 간단히 말해 **전체 픽셀 대신** 모델이 중요하게 생각하는 **특정 관심 영역 주변의 소수 샘플링 포인트에만 집중**(Sparse Attention)하는 방식입니다.

쿼리 $q$, 참조점 $p$, 입력 feature $x$ 에 대해 다음과 같이 정의됩니다.

$$
\mathrm{DeformAttn}(q,\, p,\, x) \;=\; \sum_{i=1}^{N_{head}} \mathcal{W}_i \sum_{j=1}^{N_{key}} A_{ij} \cdot \mathcal{W}'_i \, x \big( p + \Delta p_{ij} \big)
$$

여기서 $\Delta p_{ij}$ 는 **학습으로 예측되는 sampling offset**, $A_{ij}$ 는 **attention weight** 입니다.
즉 **참조점 $p$ 주변의 $N_{key}$ 개 지점만** 골라 보는 구조입니다.

#### (2) 동작 순서

![img](https://velog.velcdn.com/images/bibibip23/post/dfc54dcd-b8e4-491e-8312-5cb64229f462/image.png)

위 그림의 (b) 부분을 기준으로 보면 다음과 같습니다.

1. BEV plane 위의 각 Query $Q_p$ 를 **기둥(pillar) 형태**로 확장합니다.
2. 그 기둥에서 **3D reference points**를 $N_{ref}$ 개 뽑습니다.
3. 이 점들을 카메라 파라미터로 **2D 영상에 투영(projection)** 합니다.
4. 투영된 점이 실제로 화면 안에 들어오는 카메라, 즉 **hit view** $\mathcal{V}_{hit}$ 에서만 feature를 뽑아옵니다.
5. 이 과정을 거쳐 매 BEV Query마다 뽑아온 feature들의 **가중합**을 얻습니다.

**궁금해서 Claude에게 물어봄 — 1~3단계가 실제로 하는 일**

- **① "기둥 형태로 확장"이란?**
  BEV Query가 담당하는 위치 $p = (x, y)$ 는 **지면을 위에서 내려다본 2D 좌표**일 뿐이라 높이 정보가 없습니다.
  그런데 같은 $(x, y)$ 라도 실제로는 지면(도로 표시)일 수도, 차량 범퍼일 수도, 트럭 지붕일 수도 있습니다.
  그래서 그 위치에 **높이 값 $z'_j$ 여러 개를 붙여** 3D 점들을 만듭니다.

  $$
  (x', y') \;\longrightarrow\; \big\{ (x',\, y',\, z'_1),\; (x',\, y',\, z'_2),\; \dots,\; (x',\, y',\, z'_{N_{ref}}) \big\}
  $$

  $(x', y')$ 는 그대로 두고 높이만 다른 점들이 세로로 늘어서므로, 그 모양이 **기둥(pillar)** 처럼 보이는 것입니다. 
  별도의 신경망 연산이 아니라 **미리 정한 높이를 붙이는 것뿐**입니다.
- **② 무슨 기준으로 뽑는가?**
  학습되는 값이 아니라 **미리 고정해 둔 anchor 높이**입니다. 
  관심 3D 공간의 높이 범위(nuScenes 설정 기준 대략 $z \in [-5, 3]$ m)를 **균등 분할**해서 사용하며, **논문에서는 $N_{ref} = 4$ 개를 씁니다.**
  즉 "차량·보행자가 존재할 만한 높이 구간을 고르게 커버한다"가 기준입니다.
- **③ 투영 방향은 3D → 2D가 맞습니다.**
  ①에서 만든 3D 점 $(x', y', z'_j)$ 를 $i$ 번째 카메라의 **projection matrix $T_i$ 에 통과**시켜, 그 카메라 **이미지 평면 위의 2D 픽셀 좌표** $(x_{ij}, y_{ij})$ 로 보냅니다.
  깊이를 추정해서 2D를 3D로 올리는(ill-posed) 방향이 아니라, **이미 아는 3D 후보점을 2D로 내리는(well-posed) 방향**이라는 점이 BEVFormer의 핵심입니다.

> **hit view** : BEV Query를 3D 공간으로 올린 뒤 각 카메라 영상에 투영했을 때, 실제로 그 점이 화면 안에 존재하는 카메라(View)를 말합니다.

#### (3) 수식으로 보기

$$
\mathrm{SCA}\big( Q_p,\, F_t \big) \;=\; \frac{1}{\lvert \mathcal{V}_{hit} \rvert} \sum_{i \in \mathcal{V}_{hit}} \sum_{j=1}^{N_{ref}} \mathrm{DeformAttn}\Big( Q_p,\; \mathcal{P}(p, i, j),\; F_t^{\,i} \Big)
$$

- $F_t^{\,i}$ 는 $i$ 번째 카메라의 feature map이며, 각 카메라마다 feature map이 생성됩니다.
- $\mathcal{P}(p, i, j)$ 는 projection function으로, BEV Query를 $i$ 번째 카메라 영상 위의 좌표로 변환하는 함수입니다.
  (구체적인 함수식은 아래 **(4) ②** 참고)
- $N_{ref}$ 는 reference point의 개수입니다.
  실제 세계에서는 $(x', y')$ 가 같더라도 $z$ 축에 따라 높이가 다를 수 있기 때문에, 높이와 관련된 anchor 집합 $\{ z'_j \}_{j=1}^{N_{ref}}$ 를 미리 정의합니다.
  - 이 과정을 통해 앞에서 언급한 기둥 형태의 3D reference points $\big( x',\, y',\, z'_j \big)_{j=1}^{N_{ref}}$ 를 얻을 수 있으며,
    이 점들을 여러 카메라의 projection matrix에 통과시켜 서로 다른 image view 위의 좌표로 나타낼 수 있습니다.
- $\mathrm{DeformAttn}$ 은 투영된 위치 **주변**에서 feature를 sampling하는 연산입니다.
  그 결과를 모두 더해 reference point가 실제로 보이는 카메라, 즉 **hit view**에 대해서만 feature를 가져오고, 마지막으로 $\lvert \mathcal{V}_{hit} \rvert$ (hit view의 개수) 로 나누어 평균을 냅니다.

#### (4) 좌표 변환 두 단계

**① BEV Grid 좌표 → 실제 세계 좌표(Real World Coordinate)**

$$
x' = \Big( x - \frac{W}{2} \Big) \times s, \qquad y' = \Big( y - \frac{H}{2} \Big) \times s
$$

여기서 $W,\ H$ 는 BEV Grid의 크기이고, $s$ 는 Grid 하나가 나타내는 실제 거리(meter)를 의미합니다.
 $W/2,\ H/2$ 를 빼는 이유는 **BEV의 중앙(Ego Vehicle)** 을 원점으로 맞추기 위해서입니다.

**② 3D Reference Point → 카메라 영상 위의 2D 좌표**

$$
\mathcal{P}(p, i, j) = \big( x_{ij},\; y_{ij} \big)
$$

$$
z_{ij} \cdot \begin{bmatrix} x_{ij} \\ y_{ij} \\ 1 \end{bmatrix} \;=\; T_i \cdot \begin{bmatrix} x' \\ y' \\ z'_j \\ 1 \end{bmatrix}
$$

$T_i$ 는 $i$ 번째 카메라의 projection matrix, $z_{ij}$ 는 깊이(homogeneous scale)입니다.
①에서 얻은 3D Reference Point를 이 식으로 카메라 영상 위의 2D 좌표로 투영합니다.

* **보충 — projection function $\mathcal{P}$ 를 풀어 쓰면**

논문에서 $T_i$ 하나로 묶어 쓴 것은 사실 **표준 핀홀 카메라 모델**입니다. Extrinsic(외부 파라미터)과 Intrinsic(내부 파라미터) 두 개의 곱으로 분해됩니다.

$$
T_i \;=\; K_i \begin{bmatrix} R_i & t_i \end{bmatrix} \;\in\; \mathbb{R}^{3 \times 4}
$$

**1단계 · Extrinsic — Ego(BEV) 좌표계 → $i$ 번째 카메라 좌표계**

$$
\begin{bmatrix} x_c \\ y_c \\ z_c \end{bmatrix} \;=\; R_i \begin{bmatrix} x' \\ y' \\ z'_j \end{bmatrix} + t_i
$$

$R_i \in SO(3)$ 는 회전, $t_i \in \mathbb{R}^3$ 는 평행이동입니다. 
"차량 기준 좌표를 그 카메라가 보는 방향 기준으로 돌려놓는" 단계이며, 카메라 캘리브레이션으로 이미 알고 있는 값입니다.

**2단계 · Intrinsic — 카메라 좌표계 → 이미지 픽셀 좌표**

$$
K_i = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}, \qquad
x_{ij} = f_x \frac{x_c}{z_c} + c_x, \quad y_{ij} = f_y \frac{y_c}{z_c} + c_y
$$

* $f_x, f_y$ 는 초점거리, $(c_x, c_y)$ 는 주점(principal point)입니다.

| 기호                | 의미                                                                       |
| ------------------- | -------------------------------------------------------------------------- |
| $R_i,\ t_i$       | Extrinsic. Ego → 카메라 좌표계 변환 (캘리브레이션 값)                     |
| $K_i$             | Intrinsic. 카메라 좌표 → 픽셀 좌표 변환 (렌즈/센서 스펙)                  |
| $z_c\ (= z_{ij})$ | 카메라 좌표계에서의**깊이**. 위 행렬식의 homogeneous scale과 같은 값 |

**핵심은 $z_c$ 로 나누는 부분(perspective division)입니다.
**여기서 3D의 깊이 정보가 사라지기 때문에 역방향(2D → 3D)이 ill-posed가 되는 것이고, 
BEVFormer는 반대로 **$z'_j$ 를 미리 정해 두고 정방향으로만 계산**하므로 이 문제를 겪지 않습니다.

* **hit view 판정 조건**

$\mathcal{V}_{hit}$ 는 이 계산 결과로 정해집니다. $j$ 번째 reference point가 $i$ 번째 카메라의 hit view가 되려면

$$
z_c > 0 \quad \text{그리고} \quad 0 \le x_{ij} < W_{img},\quad 0 \le y_{ij} < H_{img}
$$

즉 **카메라 앞쪽에 있고**(뒤에 있는 점은 수식상으로는 투영되지만 실제로는 안 보입니다), **이미지 밖으로 벗어나지 않아야** 합니다.

> 실제 구현에서는 $(x_{ij}, y_{ij})$ 를 $[0, 1]$ 범위로 정규화한 뒤 `grid_sample` 로 feature를 뽑습니다.

---

### 1.3 Temporal Self-Attention — 시간 정보 이어받기

이 논문은 **움직이는 객체의 속도 추정**과 **가려진 객체 탐지** 문제를 해결하기 위해 Temporal Self-Attention을 제안했습니다.

#### (1) 직전 프레임 하나만 사용합니다

BEVFormer는 이전 정보를 모두 stacking하던 기존 방식과 달리, 바로 **이전 프레임의 History BEV Feature $B_{t-1}$ 만 활용**합니다.

#### (2) Ego motion 기반 정렬(Alignment)

여기서 주의할 점이 있습니다. **차량은 움직이기 때문에 같은 Grid라고 해서 같은 실제 위치를 나타내지 않을 수 있다**는 점입니다.
따라서 논문에서는 **Ego motion**에 따라 ***과거 BEV를 현재 좌표계에 맞게** **정렬(Alignment)*** 합니다. 이렇게 새롭게 정렬된 BEV가 $B'_{t-1}$ 입니다.

#### (3) 주변 객체의 움직임은 Attention이 처리합니다

이 과정으로 움직이는 자기 차량에 대한 문제는 해결되지만, **움직이는 주변 객체**는 어떻게 처리해야 할까요?
이를 해결하기 위해 Temporal Self-Attention이 사용됩니다.

$$
\mathrm{TSA}\Big( Q_p,\; \big\{ Q,\, B'_{t-1} \big\} \Big) \;=\; \sum_{V \in \{ Q,\; B'_{t-1} \}} \mathrm{DeformAttn}\big( Q_p,\; p,\; V \big)
$$

- **입력** : 현재 BEV Query $Q$ 와 정렬된 과거 BEV $B'_{t-1}$
- **동작** : $\mathrm{DeformAttn}$ 을 통해 현재 Query $Q_p$ 가 현재 BEV와 과거 BEV에서 **필요한 feature만 선택적으로** 가져옵니다.
- **예외** : 첫 프레임은 이전 정보가 존재하지 않으므로 $B'_{t-1} = Q$ 로 두어, 자기 자신에 대해 self-attention을 수행합니다.

#### (4) 참고 — Kalman Filter와의 비교

Multi-Object Tracking에서 사용하는 Kalman Filter와 비교하면 이해가 쉽습니다.

|           | Kalman Filter                             |                                         BEVFormer (Temporal Self-Attention)                                         |
| --------- | ----------------------------------------- | :-----------------------------------------------------------------------------------------------------------------: |
| 공통점    | 과거 정보를 현재 추론에 활용합니다        |                                                         〃                                                         |
| 접근 방식 | 객체의**상태(state)** 를 예측합니다 | 이전 시점의 **BEV Feature**를 <br />Attention을 통해 선택적으로 활용하여<br /> 현재의 공간 표현을 보완합니다 |

---

### 1.4 Applications of BEV Features

BEV features는 다양한 자율주행 인식 작업에서 범용적으로 사용될 수 있으며, **3D Object Detection**과 **Map Segmentation**은 기존의 2D 인식 모델을 약간만 수정하여 사용할 수 있습니다.

#### (1) 3D Object Detection

- 기존 **Deformable DETR**를 기반으로 한 **end-to-end 3D Detection Head**를 설계했습니다.
- Decoder의 입력으로 BEV features를 사용하여 **3D bounding box와 속도**를 예측합니다.
- 3D Bounding Box regression 시 $L_1$ **Loss만** 사용했습니다.
- Detection Head에서는 **NMS 후처리 단계 없이** 3D bounding box와 velocity를 end-to-end로 예측할 수 있습니다.

#### (2) Map Segmentation

- 기존 **Panoptic SegFormer**를 기반으로 **Segmentation Head**를 설계했습니다.
- BEV 기반 map segmentation은 semantic segmentation과 유사하여, 기존의 mask decoder를 거의 그대로 사용합니다.
- 차량, 도로, 주행 가능 영역, 차선 등 **각 클래스마다 고정된 Query**를 사용하여 Semantic Segmentation을 수행합니다.

#### (3) 정리 — BEVFormer는 Backbone입니다

위 내용을 보면 BEVFormer는 **BEV Features를 만들기 위한 Backbone**과 같다는 것을 알 수 있습니다. 따라서 최종적으로 수행하고자 하는 task에 적합한 Head를 덧붙이기만 하면, 약간의 수정만으로도 다양한 작업이 가능합니다.

---

### 1.5 Implementation Details

#### (1) Training Phase

1. 현재 시점 $t$ 의 프레임 하나를 선택합니다.
2. 그 이전 **2초** 동안의 연속된 프레임 중에서 **3개를 무작위(Random)로** 선택합니다. (예: $t-3,\ t-2,\ t-1,\ t$)
   - 이러한 무작위 샘플링은 차량의 다양한 움직임(Ego-motion)을 학습하는 데 도움이 됩니다.
3. 처음 3개의 프레임은 BEV features를 **순차적으로 생성하는 데만** 사용되며, 이 과정에서는 **역전파를 하지 않습니다(no gradients).** 실제로 Loss를 계산하는 대상은 현재 시점 $t$ 하나이기 때문입니다.
4. 현재 시점 $t$ 에서는 현재의 multi-camera 입력과 이전 BEV Feature $B_{t-1}$ 을 함께 사용하여 새로운 BEV Feature $B_t$ 를 생성합니다.
5. 마지막으로 $B_t$ 를 **detection이나 segmentation head에 입력**하고 손실함수를 계산합니다.

#### (2) Inference Phase

- video sequence의 각 프레임을 **시간 순서대로** 처리합니다.
- 이전 시점의 BEV features는 다음에 사용되기 위해 **저장**됩니다.
- 이러한 **online 방식**은 실시간 처리가 가능해 실제 환경에서도 사용될 수 있습니다.
- 시간 정보를 사용하면서도 **추론 속도는 기존 방법들과 비슷한 수준**을 유지합니다.

---

### 1.6 Experiments & Results

Experiments 부분은 이 글에서 자세하게 정리하지 않지만, **nuScenes**와 **Waymo Open Dataset**을 사용했으며 기존 연구들과 동일한 설정으로 공정하게 비교했다고 합니다.

#### (1) 기존 Camera 기반 방법과의 비교

- BEVFormer는 기존 Camera 기반 방법인 **DETR3D**보다 큰 폭의 성능 향상을 보였습니다.
- nuScenes Test 기준으로 DETR3D에 비해 **약 9.2 points 앞섰습니다.**
- 특히 **Temporal Self-Attention 덕분에** Velocity Estimation과 Occluded Object Detection 성능이 크게 향상되었습니다.

![](https://velog.velcdn.com/images/bibibip23/post/1dafa66d-d559-4c63-9e63-55132a32ee17/image.png)

#### (2) Multi-task Learning

- 같은 조건에서 여러 BEV encoder를 비교했을 때, **BEVFormer는 road segmentation 작업을 제외한 모든 작업에서 더 우수한 성능**을 보였습니다.
- **Detection Head와 Segmentation Head를 함께 학습하는 Multi-task Learning 환경**에서는 객체 탐지와 차량(Vehicles) 분할 성능이 더욱 향상되었습니다.
- 반면 **도로(Road)와 차선(Lane) 분할** 성능은 각각의 작업을 개별적으로 학습한 모델보다 다소 낮게 나타났는데, 이는 Multi-task Learning에서 흔히 발생하는 **Negative Transfer** 현상으로 설명할 수 있습니다.

> **Negative Transfer** : 여러 작업을 **동시에 학습했더니 서로 도움이 되는 것이 아니라 오히려 방해하는 현상**을 의미합니다.

#### (3) Ablation Study

> **Ablation Study** : 모델의 성능에 가장 큰 영향을 미치는 요소를 찾기 위해, 모델의 구성요소 및 feature들을 단계적으로 제거하거나 변경해 가며 성능의 변화를 관찰하는 방법입니다.

**① Spatial Cross-Attention 방식 비교**

논문에서는 Spatial Cross-Attention의 효과를 확인하기 위해 다음 3가지 방식을 비교했습니다.

| 방식                                       | 특징 / 한계                                                                                         |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| Global Attention                           | 모든 위치를 참조하므로**GPU 메모리 사용량이 매우 큽니다**                                     |
| Point-based Interaction                    | Reference Point만 사용하므로 **수용 영역(Receptive Field)이 제한됩니다**                     |
| **Deformable Attention (BEVFormer)** | **관심 영역(Local Region)만 선택적으로 참조**하여 성능과 연산량의 균형을 가장 잘 달성했습니다 |

실험 결과, **Deformable Attention 기반의 Spatial Cross-Attention이 가장 우수한 성능**을 보였습니다.

**② Temporal Self-Attention의 효과**

Temporal Self-Attention을 추가한 **BEVFormer**와 제거한 **BEVFormer-S**를 비교한 실험에서, 시간 정보를 활용할 때 다음이 개선되었습니다.

- 차량의 Velocity 추정 정확도 향상
- 객체의 위치 및 방향 예측 성능 개선
- **가려진 객체에 대한 Recall이 크게 증가**

**③ 성능-속도 Trade-off**

이 외에도 **BEV Resolution, Encoder Layer 수, Multi-scale Feature 사용 여부**를 조절하여 성능과 추론 속도의 Trade-off를 실험했습니다.

- Encoder Layer를 6개에서 1개로 줄이면 성능은 약간 감소하지만 **추론 속도는 크게 향상**됩니다.
- 가장 작은 모델은 **7ms**의 추론 속도를 달성했습니다.

즉, BEVFormer는 **응용 환경에 맞게 성능과 효율을 유연하게 조절**할 수 있습니다.

#### (4) 정성적 결과

![](https://velog.velcdn.com/images/bibibip23/post/6f7e74b8-2bd3-4944-a28e-1ccf43475009/image.png)

위 그림과 같이 BEVFormer는 대부분의 객체를 정확하게 탐지했으며, 오류는 주로 **매우 작은 객체나 먼 거리의 객체**에서 발생했습니다.

---
