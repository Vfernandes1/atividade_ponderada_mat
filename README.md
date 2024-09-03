# Atividade Ponderada - Tipos de Otimização 

## Aluno: Vinícius Oliveira Fernandes

## 1. Momentum Gradient Descent

A atualização dos pesos utilizando o **Momentum Gradient Descent** é dada por:

$$
w_{t+1} = w_t + m_{t+1}, \quad m_{t+1} = \beta m_t - \eta \frac{\partial C}{\partial w_t}
$$

Para encontrar \( w_4 \):

- Começamos com \( w_0 \), sabendo que \( m_0 = 0 \) (como o termo do momento inicial é zero).

1. **Atualização 1**:
   $$m_1 = \beta m_0 - \eta \frac{\partial C}{\partial w_0} = - \eta \frac{\partial C}{\partial w_0}$$
   
   $$w_1 = w_0 + m_1 = w_0 - \eta \frac{\partial C}{\partial w_0}$$

4. **Atualização 2**:
   $$m_2 = \beta m_1 - \eta \frac{\partial C}{\partial w_1} = \beta(-\eta \frac{\partial C}{\partial w_0}) - \eta \frac{\partial C}{\partial w_1}$$
   
   $$w_2 = w_1 + m_2 = w_1 + \beta(-\eta \frac{\partial C}{\partial w_0}) - \eta \frac{\partial C}{\partial w_1}$$

6. **Atualização 3**:
   $$m_3 = \beta m_2 - \eta \frac{\partial C}{\partial w_2} = \beta(\beta(-\eta \frac{\partial C}{\partial w_0}) - \eta \frac{\partial C}{\partial w_1}) - \eta \frac{\partial C}{\partial w_2}$$

   $$w_3 = w_2 + m_3$$

8. **Atualização 4**:
   $$m_4 = \beta m_3 - \eta \frac{\partial C}{\partial w_3}$$
   
   $$w_4 = w_3 + m_4$$

---

## 2. Adaptive Gradient - AdaGrad

A atualização do peso no **AdaGrad** é:

$$
w_{t+1} = w_t - \frac{\eta}{\epsilon + \sqrt{v_{t+1}}} \frac{\partial C}{\partial w_t}, \quad v_{t+1} = v_t + \left( \frac{\partial C}{\partial w_t} \right)^2
$$

Para encontrar \( w_3 \):

1. **Atualização 1:**
   $$v_1 = v_0 + \left( \frac{\partial C}{\partial w_0} \right)^2$$
   $$w_1 = w_0 - \frac{\eta}{\epsilon + \sqrt{v_1}} \frac{\partial C}{\partial w_0}$$

2. **Atualização 2:**
   $$v_2 = v_1 + \left( \frac{\partial C}{\partial w_1} \right)^2$$
   $$w_2 = w_1 - \frac{\eta}{\epsilon + \sqrt{v_2}} \frac{\partial C}{\partial w_1}$$

3. **Atualização 3:**
   $$v_3 = v_2 + \left( \frac{\partial C}{\partial w_2} \right)^2$$
   $$w_3 = w_2 - \frac{\eta}{\epsilon + \sqrt{v_3}} \frac{\partial C}{\partial w_2}$$

---

## 3. Root Mean Square Propagation - RMSProp

A atualização no **RMSProp** é:

$$
w_{t+1} = w_t - \frac{\eta}{\epsilon + \sqrt{v_{t+1}}} \frac{\partial C}{\partial w_t}, \quad v_{t+1} = \beta v_t + (1 - \beta) \left( \frac{\partial C}{\partial w_t} \right)^2
$$

Para encontrar \( w_3 \):

1. **Atualização 1:**
   $$v_1 = \beta v_0 + (1 - \beta) \left( \frac{\partial C}{\partial w_0} \right)^2$$
   $$w_1 = w_0 - \frac{\eta}{\epsilon + \sqrt{v_1}} \frac{\partial C}{\partial w_0}$$

2. **Atualização 2:**
   $$v_2 = \beta v_1 + (1 - \beta) \left( \frac{\partial C}{\partial w_1} \right)^2$$
   $$w_2 = w_1 - \frac{\eta}{\epsilon + \sqrt{v_2}} \frac{\partial C}{\partial w_1}$$

3. **Atualização 3:**
   $$v_3 = \beta v_2 + (1 - \beta) \left( \frac{\partial C}{\partial w_2} \right)^2$$
   $$w_3 = w_2 - \frac{\eta}{\epsilon + \sqrt{v_3}} \frac{\partial C}{\partial w_2}$$

---

## 4. Adaptive Momentum - Adam

Para o **Adam**, a atualização do peso é:

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \hat{m}_{t+1}$$

onde:

$$v_{t+1} = \beta_2 v_t + (1 - \beta_2) \left( \frac{\partial C}{\partial w_t} \right)^2, \quad m_{t+1} = \beta_1 m_t + (1 - \beta_1) \frac{\partial C}{\partial w_t}$$
e
$$\hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}, \quad \hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}$$

Para encontrar \( w_2 \):

1. **Atualização 1:**
   $$m_1 = \beta_1 m_0 + (1 - \beta_1) \frac{\partial C}{\partial w_0}$$
   $$v_1 = \beta_2 v_0 + (1 - \beta_2) \left( \frac{\partial C}{\partial w_0} \right)^2$$
   $$\hat{m}_1 = \frac{m_1}{1 - \beta_1^1}, \quad \hat{v}_1 = \frac{v_1}{1 - \beta_2^1}$$
   $$w_1 = w_0 - \frac{\eta}{\sqrt{\hat{v}_1} + \epsilon} \hat{m}_1$$

2. **Atualização 2:**
   $$m_2 = \beta_1 m_1 + (1 - \beta_1) \frac{\partial C}{\partial w_1}$$
   $$v_2 = \beta_2 v_1 + (1 - \beta_2) \left( \frac{\partial C}{\partial w_1} \right)^2$$
   $$\hat{m}_2 = \frac{m_2}{1 - \beta_1^2}, \quad \hat{v}_2 = \frac{v_2}{1 - \beta_2^2}$$
   
   $$w_2 = w_1 - \frac{\eta}{\sqrt{\hat{v}_2} + \epsilon} \hat{m}_2$$
---

Referências:

OPENAI. ChatGPT. Formatação de Equações em Markdown. Mensagem recebida por Vinícius, em 31 ago. 2024.
