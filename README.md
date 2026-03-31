# ARC-AGI-3 Starter — 基于 StochasticGoose 的竞赛起步方案

> 本仓库 fork 自 [DriesSmit/ARC3-solution](https://github.com/DriesSmit/ARC3-solution)（ARC-AGI-3 Agent Preview 竞赛第一名，得分 12.58%），已适配最新 v0.9.3 框架，可直接用于 [ARC Prize 2026](https://arcprize.org/competitions/2026) 竞赛开发。

## 核心思想

StochasticGoose **不尝试理解谜题规则，而是学习"哪些操作能引起画面变化"**，从而比随机探索更高效。

- **CNN 双头模型**：共享卷积骨干 → 动作头（预测 ACTION1-5 是否有效）+ 坐标头（预测 64×64 点击位置，纯卷积保留 2D 空间偏置）
- **分层采样**：先决定动作类型，再决定坐标；用 sigmoid 独立预测 + 公平缩放归一化
- **经验去重**：MD5 哈希确保 20 万样本缓冲中无重复，最大化样本多样性
- **关卡重置**：进入新关卡时清空缓冲并重建模型，避免旧知识干扰
- **可用动作掩码**：自动适配不同游戏的合法动作集（导航类 vs 点击类）

## 相比原仓库的改动

- 适配 ARC-AGI-3-Agents **v0.9.3**（`arcengine` 导入、`levels_completed` 字段）
- `make install` 自动补丁 submodule，无需手动修改框架代码
- 新增 `viewer.py` 实时可视化（游戏网格 + 动作概率 + 点击热力图 + 回看跳转）
- 修复非 git 仓库环境下的报错

## 快速开始

### 前置条件
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 包管理器
- [ARC API Key](https://three.arcprize.org/user)

### 安装与运行

```bash
git clone --recurse-submodules https://github.com/lxc902/ARC3-starter.git
cd ARC3-starter
make install
make setup-env
# 编辑 ARC-AGI-3-Agents/.env，填入你的 ARC_API_KEY
make action
```

### 指定游戏运行

```bash
uv run ARC-AGI-3-Agents/main.py --agent=action --game=vc33   # 点击操控类
uv run ARC-AGI-3-Agents/main.py --agent=action --game=ls20   # 地图导航类
uv run ARC-AGI-3-Agents/main.py --agent=action --game=ft09   # 逻辑匹配类
```

### 实时可视化

在另一个终端运行：

```bash
make viewer
```

操作：左右方向键逐帧、空格切换 LIVE/REVIEW 模式、拖动滑块回看。

## 原作者

- **Lead Developer**: [Dries Smit](https://driessmit.github.io/) @ [Tufa Labs](https://tufalabs.ai/)
- **Adviser/Reviewer**: [Jack Cole](https://x.com/MindsAI_Jack)

## 架构

### ActionModel（CNN）
- **输入**：16 通道 one-hot 编码帧（64×64）
- **骨干网络**：4 层 CNN（32→64→128→256 通道）
- **动作头**：预测 ACTION1-ACTION5 各自引起画面变化的概率
- **坐标头**：预测 64×64 点击位置概率（纯卷积，保留 2D 空间归纳偏置）

### 训练
- **监督学习**：(state, action) → frame_changed 二分类（BCE Loss）
- **经验缓冲**：20 万去重样本，MD5 哈希去重
- **动态重置**：进入新关卡时清空缓冲、重建模型
- **熵正则化**：轻量熵奖励防止过早收敛

### 探索策略
- **Sigmoid 独立预测**：每个动作独立预测"是否有效"
- **公平采样**：坐标概率除以 4096，使 Click 与按键动作公平竞争
- **可用动作掩码**：根据游戏返回的 `available_actions` 自动屏蔽非法动作

## Makefile 命令

| 命令 | 说明 |
|------|------|
| `make install` | 安装依赖 + 自动补丁 submodule |
| `make action` | 运行 agent（全部游戏） |
| `make setup-env` | 创建 .env 配置文件 |
| `make viewer` | 启动实时可视化 |
| `make tensorboard` | 启动 TensorBoard（http://localhost:6006）|
| `make clean` | 清理运行产物 |

## 文件结构

```
ARC3-starter/
├── ARC-AGI-3-Agents/      # 官方框架 (submodule, make install 自动补丁)
├── custom_agents/
│   ├── __init__.py        # 包初始化
│   ├── action.py          # 主 agent（CNN 模型 + 训练 + 采样）
│   └── view_utils.py      # 可视化工具函数
├── custom_agent.py        # Agent 导入入口
├── viewer.py              # 实时可视化查看器
├── utils.py               # 实验目录 / 日志工具
├── Makefile               # 构建与运行命令
├── requirements.txt       # Python 依赖
└── README.md              # 本文件
```

