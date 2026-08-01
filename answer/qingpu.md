**结论先说：按当前 MarsDog 执行器参数，这条 `qingpu` 跳跃轨迹不能可靠复现。** 前腿 EL05（thigh_roll / calf）在速度或力矩上已经越界；落地冲击未计入时力矩已超，真实接触只会更严。

下面是**不训练**也能做的分析框架，以及刚才对 `qingpu.csv` 跑出来的结果。

---

## 方法论（轨迹 × URDF/MJCF × 电机规格）

把可行性拆成三层，由易到难：

### 1) 运动学包络（必要非充分）

对 CSV 关节角 \(q(t)\)（30 Hz）：

\[
\dot q \approx \nabla_t q,\quad
\ddot q \approx \nabla_t \dot q
\]

检查：

| 检查 | 判据 |
|------|------|
| 关节限位 | \(q(t) \in [q_{\min}, q_{\max}]\)（来自 MJCF `range`） |
| 速度预算 | \(\max\|\dot q\| \le \omega_{\lim}\)（来自 `ElectricActuator.velocity_limit`） |

任一关节 \(\max\|\dot q\|/\omega_{\lim} > 1\)，则**开环都跟不上**，谈不上完成动作。

### 2) 逆动力学力矩（接近充分，仍偏乐观）

在 MuJoCo 里对每一帧设 \(q,\dot q,\ddot q\)，调用 `mj_inverse`：

\[
\tau = M(q)\ddot q + C(q,\dot q)\dot q + g(q)
\]

与 `effort_limit` 比。注意：

- **无接触 ID 是下界**：落地撞击的接触力未进方程，真实 \(\tau\) 更大；
- 30 Hz 重定向 + 数值微分会放大 \(\ddot q\) 尖峰，尖峰力矩可能偏高，但**持续超限/多关节同时超**仍可信。

### 3) 开环回放（推荐你本地再做一眼）

`csv_to_npz --render True` 或把关节写成位置伺服开环追轨迹：看是否炸关节、倒地、明显跟丢。这比纯数字更直观，但仍不是 RL。

---

## `qingpu` × 当前模型：关键数字

电机映射以 `marsdog_constants.py` 为准（前小腿/大腿滚转 = **EL05**：\(6\,\mathrm{Nm}\)，\(10.47\,\mathrm{rad/s}\)）。

### A. 速度预算（已越界）

| 关节 | 电机 | \(\|\dot q\|_{\max}\) | \(\omega_{\lim}\) | 比值 |
|------|------|----------------------|-------------------|------|
| `fr_calf_joint` | EL05 | **15.1** | 10.47 | **1.44×** |
| `fl_calf_joint` | EL05 | **11.0** | 10.47 | **1.05×** |
| 其余主动关节 | — | — | — | < 0.6× |

峰值出现在 **落地后恢复段**（约 1.5–1.9 s），不是腾空中点。

另有少量**关节角出硬限位**：后腿 calf、前腿 thigh_roll 等（约 8–16% 帧），重定向本身已贴边。

### B. 无接触 ID 力矩（已越界，且偏乐观）

| 关节 | 电机 | \(\|\tau\|_{\max}\) | \(\tau_{\lim}\) | 比值 | 状态 |
|------|------|---------------------|-----------------|------|------|
| `fr_thigh_roll_joint` | EL05 | **20.5** | 6 | **3.4×** | OVER |
| `head_pitch_joint` | EL05 | 8.9 | 6 | 1.5× | OVER |
| `waist_pitch_joint` | PA43 | 22.4 | 18 | 1.2× | OVER |
| `fr_calf_joint` | EL05 | 6.5 | 6 | 1.1× | OVER |
| `fl_thigh_roll_joint` | EL05 | 6.1 | 6 | 1.0× | OVER |

后腿大关节（PA43/RS00）在无接触 ID 下很轻松；**瓶颈在前腿 EL05 + 腰 pitch**。计入落地冲击后，前腿只会更差。

---

## 直接回答「当前模型能不能完成」

| 层级 | 判定 |
|------|------|
| 关节空间能不能“写出”这条轨迹 | 勉强（有少量出限位） |
| 执行器速度能不能跟上 | **不能**（前 calf 超速） |
| 执行器力矩能不能跟上 | **不能**（前 thigh_roll 等已超，且未含撞击） |
| 结论 | **当前硬件规格下，不宜认为该动作可完整执行**；最多是“降速/削峰后的近似跳”或换更强前腿电机后再评估 |

---

## 你本地可复现的最短流程

```bash
# 1) 过滤
uv run python scripts/tools/filter_marsdog_active_dofs.py \
  --input-file src/mjlab/csv/qingpu.csv \
  --output-file src/mjlab/csv/qingpu_active.csv

# 2) 可视化回放（开环看形）
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file src/mjlab/csv/qingpu_active.csv \
  --output-name qingpu_motion \
  --input-fps 30 --output-fps 50 --render True
```

数值侧就是上面这套：**限位 → \(\dot q/\omega_{\lim}\) → `mj_inverse` 的 \(\tau/\tau_{\lim}\)**。若你要，我可以在 Agent 模式里把这段分析收成一个可复用脚本（例如 `scripts/tools/check_motion_actuator_budget.py`）。