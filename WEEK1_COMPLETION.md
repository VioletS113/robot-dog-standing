# Week 1 任务完成说明

## 完成内容

### 1. 新增文件：`source/robotDogStanding/robotDogStanding/utils/metrics.py`

实现了 `StandingSuccessMetric` 类，用于判定机器人是否成功站立：

**Success 判定条件：**
- 连续保持 `duration_sec`（默认 1.0 秒）满足以下所有条件：
  - `base_height > 0.35m`（可配置的 height_threshold）
  - `|roll| < 0.3 rad`（约17度，可配置的 angle_threshold）
  - `|pitch| < 0.3 rad`（约17度）

**主要功能：**
- 支持 vectorized env（num_envs >= 1）
- 跟踪每个环境的连续成功步数
- 提供 `compute_conditions()` 方法获取当前状态指标
- 提供 `update()` 方法更新成功状态
- 预留了 `contact_threshold` 参数用于未来添加接触力判定

**从环境获取状态的方式：**
```python
# Base height
base_height = env.scene["robot"].data.root_pos_w[:, 2]

# Roll, pitch, yaw from quaternion
quat = env.scene["robot"].data.root_quat_w
roll, pitch, yaw = euler_xyz_from_quat(quat)
```

### 2. 新增文件：`source/robotDogStanding/robotDogStanding/utils/__init__.py`

将 utils 目录设置为 Python 包，导出 `StandingSuccessMetric`。

### 3. 修改文件：`scripts/random_agent.py`

完全重写了 random_agent.py，实现以下功能：

**命令行参数：**
```bash
python scripts/random_agent.py \
  --headless \
  --task=Template-Robotdogstanding-v0 \
  --num_episodes=20 \
  --num_envs=1 \
  --step_log_interval=10
```

**环境创建方式：**
- 使用与 `train.py` 相同的方式：`parse_env_cfg()` + `gym.make()`
- 默认 task 为 `Template-Robotdogstanding-v0`
- 默认 num_envs=1（baseline evaluation）

**随机策略：**
- 每步采样 action：`2 * torch.rand(action_space.shape) - 1`
- 匹配 action space 范围 [-1, 1]

**日志记录：**

日志保存到：`outputs/logs/random_policy/<timestamp>/`

包含三个文件：

1. **`summary.json`** - 汇总统计：
   ```json
   {
     "num_episodes": 20,
     "success_rate": 0.05,
     "avg_return": -12.34,
     "avg_episode_length": 485.5,
     "std_return": 8.56,
     "min_return": -28.12,
     "max_return": 3.45
   }
   ```

2. **`episode_metrics.csv`** - 每个 episode 一行：
   - episode, total_reward, episode_length, success
   - final_height, final_roll, final_pitch
   - avg_height, avg_abs_roll, avg_abs_pitch

3. **`step_log.csv`** - 每 N 步一行（默认 N=10）：
   - episode, step
   - height, roll, pitch, yaw
   - reward, done, success_condition_met

**实现细节：**
- 使用 `StandingSuccessMetric` 跟踪成功状态
- 每个 episode 收集完整的 height/roll/pitch 轨迹
- 定期记录 step 数据（通过 `step_log_interval` 控制频率）
- Episode 结束时记录完整的 episode 指标
- 所有 episodes 完成后保存汇总 JSON

### 4. 新增文件：`outputs/logs/random_policy/README.md`

详细的使用说明文档，包含：
- 如何运行脚本
- 输出文件格式说明
- Success 判定标准
- 示例输出

## 使用方法

### 安装扩展（如果还未安装）：
```bash
python -m pip install -e source/robotDogStanding
```

### 运行 random policy baseline：
```bash
# 基础用法（20 episodes）
python scripts/random_agent.py --headless

# 自定义 episodes 数量
python scripts/random_agent.py --headless --num_episodes=50

# 更频繁的 step logging（每 5 步记录一次）
python scripts/random_agent.py --headless --step_log_interval=5
```

### 查看结果：
```bash
# 查看最新的日志目录
ls -lt outputs/logs/random_policy/

# 查看汇总统计
cat outputs/logs/random_policy/<timestamp>/summary.json

# 查看 episode 指标
cat outputs/logs/random_policy/<timestamp>/episode_metrics.csv | column -t -s,

# 查看 step 日志
head -20 outputs/logs/random_policy/<timestamp>/step_log.csv | column -t -s,
```

## 代码结构预留

1. **Contact force 支持：**
   - `StandingSuccessMetric` 已预留 `contact_threshold` 参数
   - 当可以获取 hind feet contact 时，可以轻松添加接触力判定：
   ```python
   if self.contact_threshold is not None:
       contact_sensor = self.env.scene["contact_forces"]
       hind_feet_contact = ...  # 获取后脚接触力
       contact_ok = hind_feet_contact > self.contact_threshold
       conditions_met = conditions_met & contact_ok
   ```

2. **Vectorized environment 支持：**
   - 所有代码都使用 tensor 操作，支持 num_envs > 1
   - 当前 baseline 使用 num_envs=1 以便逐个 episode 评估
   - 未来可以扩展到并行评估多个 episodes

## 验证

所有文件已通过语法检查：
```bash
python -m py_compile source/robotDogStanding/robotDogStanding/utils/metrics.py
python -m py_compile scripts/random_agent.py
```

## 下一步

1. 运行脚本获取 baseline 数据
2. 分析 random policy 的表现（预期 success rate 很低）
3. 使用这些 metrics 评估训练后的 RL policy
4. 根据需要调整 success 判定阈值
5. 如果可以获取接触力数据，添加 contact-based success 判定

## 文件清单

新增/修改的文件：
- ✅ `source/robotDogStanding/robotDogStanding/utils/metrics.py` (新增)
- ✅ `source/robotDogStanding/robotDogStanding/utils/__init__.py` (新增)
- ✅ `scripts/random_agent.py` (修改)
- ✅ `outputs/logs/random_policy/README.md` (新增)
- ✅ `WEEK1_COMPLETION.md` (新增，本文件)

临时测试文件（可删除）：
- `test_imports.py`
