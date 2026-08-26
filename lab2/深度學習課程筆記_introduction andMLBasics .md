# 深度學習課程學習紀錄

---

## 目錄

- [第一堂：Introduction (L1)](#第一堂introduction-l1)
- [第二、三堂：Machine Learning Basics (L5)](#第二三堂machine-learning-basics-l5)
- [Lab 作業：HW2 Binary Classification](#lab-作業hw2-binary-classification)

---

## 第一堂：Introduction (L1)

### 什麼是 Machine Learning？

機器學習是從原始資料中**提取模式**來獲取知識的方法。

**範例流程（以 MRI 預測健康為例）：**

```
x → [Rep.] → φ(x) → [Model] → y → (-) → Cost
                                     ↑
                                     t (真實標籤)
```

| 符號 | 意義 |
|------|------|
| x | 輸入（如 MRI 掃描） |
| φ(x) | 資料表示（特徵） |
| y | 模型預測值，y ∈ (0,1) |
| t | 真實標籤，t ∈ {0,1} |

**預測公式：**

$$y = f_{\mathbf{w}}(\phi(\mathbf{x})) = \sigma(\mathbf{w}^T \phi(\mathbf{x})), \quad \sigma(s) = \frac{1}{1+e^{-s}}$$

---

### Sigmoid 函數

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- 輸出範圍：(0, 1)
- 用於將實數值轉換為機率

---

### 監督式學習 (Supervised Learning)

- 每個輸入 x 都有對應的真實標籤 t
- 目標：找到函數 $f_{\mathbf{w}}(\phi(\mathbf{x}))$ 逼近 $t(\mathbf{x})$
- Cost：y 與 t 之間的距離，如 $\|y - t\|_2^2$，對所有 {x, t} 配對最小化 w

---

### 資料表示 Data Representation φ(x)

- 資料表示**關鍵決定**預測效能
- 傳統機器學習使用手工設計特徵，但很多任務難以判斷該用什麼特徵
- **Deep Learning** 的核心：讓模型自動學習特徵表示

---

### Deep Learning

深度學習是一種機器學習方法，其資料表示基於**概念的層次結構（hierarchy of concepts）**。

**數學形式：**

$$f_{\mathbf{w},\boldsymbol{\theta}_n,...,\boldsymbol{\theta}_1}(\mathbf{x}) = \sigma\left(\mathbf{w}^T \underbrace{\phi_{\boldsymbol{\theta}_n}(\phi_{\boldsymbol{\theta}_{n-1}}(\cdots\phi_{\boldsymbol{\theta}_1}(\mathbf{x})\cdots))}_{\text{Hierarchy of concepts/features}}\right)$$

- 從簡單函數的巢狀組合構建複雜函數 f(x)
- φ_θ 通常是 vector-valued functions，如 $\phi_\theta(\mathbf{x}) = \sigma(\boldsymbol{\theta}\mathbf{x})$

**Feedforward Deep Networks 範例（圖像識別）：**

```
輸入圖像 → 邊緣檢測(Layer 1) → 角點/輪廓(Layer 2) → 物件部位(Layer 3) → 輸出(CAR/PERSON/ANIMAL)
```

---

### Classic ML vs Deep Learning

| 方法 | 結構 |
|------|------|
| Rule-based systems | Input → Hand-designed program → Output |
| Classic ML | Input → Hand-designed features → Mapping → Output |
| Representation Learning | Input → Features → Mapping → Output |
| **Deep Learning** | Input → Simple features → More abstract layers → Output |

---

### 深度學習歷史

| 時期 | 名稱 | 重要發展 |
|------|------|---------|
| 1940s-1960s | **Cybernetics** | Perceptron (Rosenblatt, 1958)、ADALINE (Widrow and Hoff, 1960) |
| 1980s-1990s | **Connectionism** | RNN (Rumelhart et al., 1986)、CNN (LeCun et al., 1998)、LSTM (Hochreiter & Schmidhuber, 1997) |
| 2006s- | **Deep Learning** | Deep Belief Networks (Hinton, 2006)、VAE (Kingma et al., 2014)、GAN (Goodfellow et al., 2014) |

---

## 第二、三堂：Machine Learning Basics (L5)

### 學習演算法定義 (Mitchell, 1997)

> 若一個電腦程式在任務 T 上的表現 P 隨著經驗 E 的增加而提升，則稱它從經驗 E 中學習。

**以 Linear Regression 為例：**

- **Task T**：從 x 預測 y，輸出 $\hat{y} = \mathbf{w}^T\phi(\mathbf{x}) = \phi(\mathbf{x})^T\mathbf{w}$
- **Experience E**：在訓練集上最小化目標函數
- **Performance P**：在測試集上測量 MSE

---

### Linear Regression

**訓練目標（MSE）：**

$$\text{MSE}^{(\text{train})} = \frac{1}{m^{(\text{train})}} \|\hat{\mathbf{y}}^{(\text{train})} - \mathbf{y}^{(\text{train})}\|_2^2$$

**最小化方法**：令 $\nabla_\mathbf{w}\text{MSE}^{(\text{train})} = 0$，幾何解釋為 $\hat{\mathbf{y}}^{(\text{train})}$ 是 $\mathbf{y}^{(\text{train})}$ 投影到 $\Phi^{(\text{train})}$ 的列空間。

**解析解（Normal Equation）：**

$$\mathbf{w} = \left(\Phi^{(\text{train})T}\Phi^{(\text{train})}\right)^{-1}\Phi^{(\text{train})T}\mathbf{y}^{\text{train}}$$

**包含 bias 的擴展：**

$$\hat{y} = \mathbf{w}^T\phi(\mathbf{x}) + b = \tilde{\mathbf{w}}^T\tilde{\phi}(\mathbf{x}), \quad \tilde{\mathbf{w}} = \begin{bmatrix}\mathbf{w}\\b\end{bmatrix}, \; \tilde{\phi}(\mathbf{x}) = \begin{bmatrix}\phi(\mathbf{x})\\1\end{bmatrix}$$

---

### Generalization（泛化能力）

- 模型在**未見過**的新資料上表現好的能力
- **泛化誤差**：在新輸入上的預期誤差，通常用測試集估計
- **Bayes error**：當模型完全知道 $p_\text{data}(\mathbf{x},y)$ 時，能達到的最小泛化誤差

---

### Underfitting vs. Overfitting

| 問題 | 原因 | 表現 |
|------|------|------|
| **Underfitting** | 模型 capacity 太低 | 訓練誤差大 |
| **Overfitting** | 模型 capacity 太高 | 訓練誤差小，測試誤差大 |

- 通常 **test error ≥ training error**（過擬合的來源）
- 可透過調整 **model capacity** 來平衡

**範例（多項式次數 d）：**
- d=1：Underfitting
- d=2：Appropriate capacity
- d=9：Overfitting（尤其資料有雜訊時，E_out 可高達 9.00 vs 0.034 的 E_in）

---

### Regularization（正則化）

**目的**：修改學習演算法以降低泛化誤差（通常代價是提高訓練誤差）

**Weight Decay 範例：**

$$J(\mathbf{w}) = \text{MSE}^{(\text{train})} + \lambda \mathbf{w}^T\mathbf{w}$$

- λ 控制對小 w 的偏好
- λ 太大 → Underfitting，λ → 0 → Overfitting

**L2 vs L1 Regularizer：**

| 正則項 | 約束區域形狀 | 特性 |
|--------|------------|------|
| L2：$\mathbf{w}^T\mathbf{w} \leq C$ | 球形 | 平滑縮小所有權重 |
| L1：$\|\mathbf{w}\|_1 \leq C$ | 菱形（鑽石形） | 傾向產生稀疏解（部分權重為 0） |

---

### Estimators（估計量）

**Point Estimation**：從 i.i.d. 樣本 $\{x^{(1)}, x^{(2)}, \ldots, x^{(m)}\}$ 給出單一估計值

$$\hat{\theta}_m = g(x^{(1)}, x^{(2)}, \ldots, x^{(m)})$$

**Bias（偏差）：**

$$\text{bias}(\hat{\theta}_m) = E(\hat{\theta}_m) - \theta$$

- **Unbiased**：bias = 0
- **Asymptotically unbiased**：$\lim_{m\to\infty} \text{bias}(\hat{\theta}_m) = 0$

**Consistency（一致性）：**

$$\lim_{m\to\infty} P\left(|\hat{\theta}_m - \theta| > \varepsilon\right) = 0, \quad \varepsilon > 0$$

> 一個估計量要注意兩個特性：**Bias** 和 **Consistency**

**Gaussian Distribution 的估計量：**

- 樣本均值（unbiased）：$\hat{\mu}_m = \frac{1}{m}\sum_{i=1}^m x^{(i)}$
- 樣本變異數（asymptotically unbiased）：$\hat{\sigma}^2_m = \frac{1}{m}\sum_{i=1}^m(x^{(i)} - \hat{\mu}_m)^2$
- Unbiased 樣本變異數：$\tilde{\sigma}^2_m = \frac{m}{m-1}\hat{\sigma}^2_m$

**標準誤差（SE）：**

$$\text{SE}(\hat{\mu}_m) = \sqrt{\text{Var}\left[\frac{1}{m}\sum_{i=1}^m x^{(i)}\right]} = \frac{\sigma}{\sqrt{m}}$$

**95% 信賴區間：** $\left(\hat{\mu}_m - 1.96\cdot\text{SE}(\hat{\mu}_m),\; \hat{\mu}_m + 1.96\cdot\text{SE}(\hat{\mu}_m)\right)$

---

### Maximum Likelihood (ML) Estimation

$$\theta_\text{ML} = \arg\max_\theta \sum_{i=1}^m \log p_\text{model}(x^{(i)};\theta)$$

最大化 log-likelihood ≡ 最小化 KL 散度 ≡ 最小化 Cross-entropy：

$$D_{KL}(\hat{p}_\text{data} \| p_\text{model}) = E_{\mathbf{x}\sim\hat{p}_\text{data}}[\log\hat{p}_\text{data}(\mathbf{x}) - \log p_\text{model}(\mathbf{x};\theta)]$$

---

### Bayesian Statistics 與 MAP

**Bayesian approach**：將參數 θ 視為隨機變數，利用 Bayes rule：

$$p(\theta|\mathbf{X}) = \frac{p(\mathbf{X}|\theta)p(\theta)}{p(\mathbf{X})}$$

**Maximum A Posteriori (MAP)：**

$$\theta_\text{MAP} = \arg\max_\theta \log p(\mathbf{X}|\theta) + \log p(\theta)$$

- 許多正則化策略（如 ML + weight decay）可以解釋為 MAP 推斷
- weight decay 對應 Gaussian prior $p(\mathbf{w}) = \mathcal{N}(\mathbf{w};\mathbf{0}, \lambda^{-1}\mathbf{I})$

---

### Support Vector Machines (SVM)

- 監督式學習中最具影響力的方法之一
- **核心想法**：在特徵空間中找到一個超平面，使決策邊界的 margin 最大化

**決策函數：** $\hat{y}(\phi(\mathbf{x})) = \mathbf{w}^T\phi(\mathbf{x}) + b$

**最大化 margin 的最佳化問題：**

$$\min \frac{1}{2}\|\mathbf{w}\|_2^2, \quad \text{subject to } t_n(\mathbf{w}^T\phi(\mathbf{x}_n) + b) \geq 1, \; \forall n$$

利用 Lagrange multipliers（KKT conditions）求解：

$$\hat{y}(\phi(\mathbf{x})) = \sum_{i=1}^N a_n t_n \underbrace{\phi(\mathbf{x}_n)^T\phi(\mathbf{x})}_{\text{Kernel}} + b$$

- **Support vectors**：$a_n \neq 0$ 的訓練樣本

---

### Constrained Optimization (KKT Conditions)

**Lagrangian 函數：**

$$L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\alpha}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g^{(i)}(\mathbf{x}) + \sum_{j=1}^n \alpha_j h^{(j)}(\mathbf{x})$$

**最優解需滿足：**
1. $\nabla L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\alpha}) = 0$
2. 所有約束條件都滿足
3. **Complementary slackness**：$\alpha_j = 0$ for $h^{(j)}(\mathbf{x}) < 0$（inactive）；$\alpha_j \geq 0$ for $h^{(j)}(\mathbf{x}) = 0$（active）

---

### Principle Component Analysis (PCA)

**無監督學習**，學習資料表示：

- **Encoder**：$f(\mathbf{x}) = \mathbf{D}^T\mathbf{x}$，降維到 $\mathbf{c} \in \mathbb{R}^l$
- **Decoder**：$g(\mathbf{c}) = \mathbf{D}\mathbf{c}$，$\mathbf{D}$ 為正交矩陣

**目標函數：**

$$\arg\min_\mathbf{D} \|\mathbf{X}^{(\text{train})} - \mathbf{X}^{(\text{train})}\mathbf{D}\mathbf{D}^T\|_F^2, \quad \text{s.t. } \mathbf{D}^T\mathbf{D} = \mathbf{I}$$

**解法**：利用 SVD 分解 $\mathbf{X}^{(\text{train})} = \mathbf{U}\Sigma\mathbf{V}^T$，D 的列向量為 V 中對應最大 l 個奇異值的列向量。

---

### Gradient-based Learning

當沒有 closed-form solution 時，使用梯度法迭代更新。

**Steepest Descent（最速下降法）：**

$$\mathbf{w}^{(n+1)} = \mathbf{w}^{(n)} - \epsilon\nabla_\mathbf{w}J(\mathbf{w}^{(n)})$$

**Gradient Descent 演算法：**

1. 初始化 $w_0$
2. 對 t = 0, 1, ...：
   - 計算 $\nabla E_{in}(w_t)$
   - 更新：$w_{t+1} = w_t - \eta \nabla E_{in}(w_t)$
3. 直到 $\nabla E_{in}(w_{t+1}) = 0$ 或達到足夠迭代次數

**Learning Rate η 的選擇：**

| η 值 | 問題 |
|------|------|
| 太小 | 收斂太慢 |
| 太大 | 不穩定，震盪 |
| 適當（變動） | 最佳 |

**Stochastic Gradient Descent (SGD)：**

使用 mini-batch（$m' \ll m$）近似梯度：

$$\mathbf{g}' = -\frac{1}{m'}\sum_{i=1}^{m'}\nabla_\mathbf{w}\log p_\text{model}(y^{(i)}|\mathbf{x}^{(i)};\mathbf{w})$$

**SGD 的變體：**
- Momentum (Polyak, 1964)
- Nesterov Momentum (Sutskever et al., 2013)
- AdaGrad (Duchi et al., 2011)
- RMSProp (Hinton, 2012)
- Adam (Kingma and Ba, 2014)

---

## Lab 作業：HW2 Binary Classification

### 作業概述

- **題目**：使用 UCI Adult Income 資料集，從個人資料（年齡、教育程度、職業等）預測收入是否超過 5 萬美元（二元分類）
- **限制**：**禁止使用 PyTorch autograd**，需手動實作反向傳播

### 資料集

| 項目 | 數量 |
|------|------|
| 訓練集大小 | 32,561 筆 |
| 測試集大小 | 16,281 筆 |
| 特徵維度 | 106 |
| 標籤 | 0（poor）/ 1（rich） |

**資料前處理：** 標準化（Z-score normalization）

```python
class STDNorm:
    def __call__(self, x: torch.Tensor):
        mu = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mu) / (std + 1e-8)
```

---

### 模型架構

**2-layer MLP：**

```
Input (106) → Linear → ReLU → Linear → Sigmoid → Output (1)
```

---

### 核心元件實作

#### Sigmoid 函數

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```python
def sigmoid(z):
    out = 1 / (1 + torch.exp(-z))
    out = torch.clamp(out, 1e-6, 1-1e-6)
    return out
```

#### ReLU 函數

```python
class Relu:
    def forward(self, x):
        out = x.clone()
        out[out < 0] = 0
        return out
    
    def backward(self, upstream_grad):
        local_grad = torch.zeros_like(self.ori_tensor)
        local_grad[self.ori_tensor > 0] = 1.0
        return upstream_grad * local_grad
```

#### Linear Layer（含反向傳播）

```python
class LinearLayer:
    def forward(self, x):
        self.x = x.clone()
        return x @ self.w + self.b
    
    def backward(self, upstream_grad):
        self.dw = self.x.T @ upstream_grad       # ∂L/∂W
        self.db = torch.sum(upstream_grad, dim=0) # ∂L/∂b
        return upstream_grad @ self.w.T           # ∂L/∂x
```

#### Binary Cross-Entropy Loss

$$L = -\left[y \log(p) + (1-y)\log(1-p)\right]$$

$$\frac{\partial L}{\partial p} = \frac{p - y}{p(1-p)}$$

---

### 最佳化器

#### SGD

$$w \leftarrow w - \eta \cdot \nabla w$$

#### SGD + Weight Decay（L2 正則化）

$$w \leftarrow w - \eta(\nabla w + \lambda w)$$

#### SGD with Momentum

$$v \leftarrow \mu v + \eta \cdot \nabla w$$
$$w \leftarrow w - v$$

#### AdamW（Bonus）

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = m_t / (1-\beta_1^t), \quad \hat{v}_t = v_t / (1-\beta_2^t)$$
$$w \leftarrow w(1 - \eta\lambda) - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

---

### 訓練設定

| 超參數 | 值 |
|--------|---|
| Learning Rate | 1e-2 |
| Hidden Dim | 200 |
| Batch Size | 128 |
| Epochs | 20 |
| Weight Decay | 1e-2 |
| Momentum | 0.5（SGDM） |
| β1, β2, ε | 0.9, 0.999, 1e-8（AdamW） |

---

### 學習反思

1. **手動實作 Backpropagation**：透過自己計算每層的梯度，深刻理解 Chain Rule 的應用
2. **各種優化器比較**：SGD → Weight Decay → Momentum → AdamW，每一步都降低了訓練難度並提升了泛化能力
3. **Weight Decay 的意義**：在 SGD+Weight Decay 中體驗到了 L2 正則化實際上等同於對參數加上 Gaussian prior（MAP 推斷）
4. **AdamW vs Adam 的差異**：AdamW 將 weight decay 與梯度更新分離，使正則化更乾淨有效
