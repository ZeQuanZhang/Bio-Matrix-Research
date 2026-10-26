# 从本地 PyCharm 启动 A800 的 v0.7 队列

> 归档说明：本文保留写作时点的设计或状态。项目已停止v0.9后续训练；最新执行状态见[实验索引](RESULTS_INDEX.md)。公开副本不含权重与语料。

在本项目选择 **Bio-Matrix v0.7 A800** 运行配置，或右键运行 `run_biomatrix_v07_a800.py`。使用本地 `.venv/bin/python`，勾选 **Emulate terminal in output console**。本地负责 SSH 和进度显示，训练在 A800 上运行。

默认 `MODE='follow'`，共享运行配置明确填写了 `--mode follow`：点击 Run 后每秒持续读取远端状态，直到按 Stop／Ctrl+C 或远端调度器退出。轮询会扣除 SSH 与显示耗时；网络或远端读取本身超过1秒时以实际可取得的状态为准。总进度末尾显示刷新时间，确认仍在跟踪。需要首次启动后台队列时显式使用 `--mode start`；远端已有同一调度器时会附着查看。

启动后立即打印 `[A800] v0.7 控制台已启动` 并强制刷新，首轮 SSH 返回前就显示连接提示。持续查看遇到首次连接失败或超时会显示原因并重连；`status`／`--json` 单次模式仍直接报错。PyCharm与普通终端都用本地入口同款tqdm：总进度在第1行、资源在第2行，各worker在后续固定位置原位更新。保留`leave=False`、`mininterval=1`、`maxinterval=5`，宽高由安全尺寸检测动态提供。首次连接和条件开始／完成／失败时记录一行事件，周期刷新不追加整块日志，也不清屏。更新前已运行的查看进程需要Stop后重新Run才能加载新实现；远端训练继续。

总进度按全部条件的已完成训练步数求和，正式核心网格总计45×24000步，显示完成、活跃、失败、中止条件数及时间。资源行显示最近采样target/s合计、GPU利用率、显存、功耗和温度。每个worker各有一条24,000步详细进度，显示模型、节点数、种子、step、loss、速度和首次编译／训练／验证／历史干预阶段；补位后同一行切换到新条件。当前14个worker加总进度和资源共16行，Run区域建议至少17行，留一行光标余量。吞吐为最近训练指标，GPU遥测每次读取。

PyCharm启动或调整窗口时，PTY尺寸可能短暂为0。tqdm 4.70.1会把它减为负数，导致隐藏进度条或在中文裁剪中死循环；入口现在在构造前及每次动态尺寸检测中都保护宽高，避免负数。已在实际PyCharm确认恢复可见及原位更新。

进度条仅显示完成数量和训练指标，关闭 tqdm 默认的 it/s 和 ETA。附着时已有的 step 并非本地查看期间完成，直接用它除以刚打开控制台的几秒会得到错误速度和剩余时间；训练速度以远端指标中的 target/s 为准。

如果已存在的个人 Run 配置仍保存了 `--mode status`、`--once` 或 `--json`，删除这些参数并改为 `--mode follow`。右键运行入口文件也会使用新的持续跟踪默认值。动态进度依赖 **Emulate terminal in output console**；共享配置已启用。

## 常用操作

给运行配置填写命令行参数；右键运行入口时也可以修改顶部 MODE。共享运行配置中的显式参数优先于顶部 MODE：

| MODE | 命令行参数 | 行为 |
|---|---|---|
| `start` | `--mode start` | 首次启动；已经运行则附着查看 |
| `status` | `--mode status` | 查看一次状态 |
| `follow` | `--mode follow` | 持续查看现有任务，不启动训练 |
| `pause` | `--mode pause` | 核对 PID、进程启动时间和命令后，仅向对应调度器发送 SIGTERM；等待 child 保存并退出 |
| `resume` | `--mode resume` | 显式启用 `--resume-aborted`，从原检查点继续中止条件 |
| `retry_failed` | `--mode retry_failed` | 显式启用 `--retry-failed`，重试失败条件 |
| `fetch` | `--mode fetch` | 取回原始记录和检查点的压缩包 |

`--once` 只查看一轮；与 start 连用仍会启动后台训练。`--json` 输出一份可检查的公共状态 JSON。需要同时恢复中止和失败条件时使用 `--mode resume --retry-failed`。普通 start 不隐式重试失败或中止条件。

**Ctrl+C 或停止本地查看不会暂停远端训练。** 需要保存并暂停时明确执行 pause。暂停请求由调度器转发给 child，调度器等待 child 退出；暂停期间仍可看到最后一轮训练／保存进度。不存在 `kill -9` 回退。若调度器不在但旧 child 仍存活，控制器拒绝另开任务并报告，避免重复训练或误杀进程。

## 已保存连接

本地连接配置为 `results/v07/remote_a800/connection.local.json`，权限必须为 600。该目录已被项目的 `results/` 忽略规则覆盖。配置包含 host、port、username、password、known_hosts、control_path、remote_python、release、workspace、scheduler、workers、cpu_threads 和 environment；不得提交到版本库。

密码按用户授权保存在这个本地文件中。native OpenSSH 通过临时、仅当前用户可执行的 askpass wrapper 调用本地 Python helper；密码只通过 OpenSSH 的 askpass 通道返回，不进入 SSH 命令行、远端脚本、远端环境或进度日志。wrapper 使用后清理。主机公钥由配置中的固定 known_hosts 文件校验，禁止自动接受新公钥。

environment 只用于后台 scheduler 的远端运行环境，保留本次核验的 NVIDIA MPS 路径和线程设置；这些设置被 child 继承。`compile_step=true` 时显式传入 scheduler 的 `--compile-step`；编译 adapter 和模式由远端 manifest 固定，不能在原 workspace 内临时切换。模型、数据或 source 变化仍由 scheduler 的哈希校验拒绝。

新调度器还读取本地连接配置的 `dispatch_policy`（`barrier`／`backfill`，缺省 `barrier`）及 `gpu_reserve_mib`（非负整数，缺省 12288），显式传给远端 `--dispatch-policy` 和 `--gpu-reserve-mib`。`start`／`resume` 附着已有调度器时同时核对命令行和manifest中的对应值；`follow`／`status`仅查看，允许读取运行中的其他调度策略。`backfill`在worker空闲时补入其他条件，显存预留由调度器守卫。调度器当前最多运行16个worker，显示会随worker数补齐详细条。对已存在的冻结队列切换调度方式必须经过单独核验的调度迁移，控制器不会隐式重写manifest。

2026-10-23已由12路扩到14路。独立的`v07_a800_live.py`接管适配器保留原12个训练子进程的PID、attempt和运行状态，只替换父监督进程，再启动slot12／13。基础调度器、训练manifest和已完成结果保持一致；适配器自身SHA及接管前后状态写入`live_adoptions/`。本地私有连接配置已更新，已打开的旧查看器也可识别新父进程并自动补齐14条。接管器只在显式操作信号下向身份核实后的训练进程组转发暂停；监督器异常退出保留训练子进程，须用显式接管流程恢复监督。

控制器只使用系统 `ssh` 和现有 `tqdm`，不需要 Paramiko，也不需要让 PyCharm 建立远程 Python SDK。

## 原始结果

远端状态为 `workspace/status.json`，每个条件的原始目录为：

```text
workspace/raw/CONDITION_ID/CONDITION_ID/
    metrics.jsonl
    validation.jsonl
    protocol.json
    last.pt
    best.pt
    result.json
```

并行调度器保留 `attempts/` 中每次尝试的 stdout、PID 和完成标记。控制器另存 `scheduler.pid.json`，每次启动追加 `scheduler.log`。恢复不会重新训练已经核验完成的条件。

fetch 将 manifest、status、configs、raw、attempts 与调度器记录下载到本地 `results/v07/remote_a800/downloads/a800-raw-时间.tar.gz`。调度迁移审计目录 `scheduling_migrations/`、活进程接管收据`live_adoptions/`和迁移前状态 `backfill_before.json` 存在时一并下载。每次使用新文件名，不覆盖以前的下载，不自动解压；不重复下载 worker 源码或冻结语料。每个普通文件先打开，再通过同一文件描述符取得 inode 和长度，因此训练中的 `last.pt` 原子替换不会把旧长度与新内容混在一起。跳过 `.tmp` 临时项和符号链接；遍历期间恰好消失的项可跳过，其他错误保留 `.part` 并报错。

运行中取回的是各文件独立快照，文件之间可能来自不同训练时刻。需要完整停止后的统一检查点快照时先 pause，等待远端退出，再 fetch。

断网时进度保留最后状态并尝试重连，远端 nohup 任务继续。只想重新连接时运行 follow；不用再次创建实验目录或更改训练配置。
