# AirFogSim源码阅读整合文档

## 目录
1. [平台概述](#平台概述)
2. [核心架构](#核心架构)
3. [仿真流程](#仿真流程)
4. [任务生命周期](#任务生命周期)
5. [核心组件详解](#核心组件详解)
6. [算法模块](#算法模块)
7. [常见问题与解答](#常见问题与解答)

---

## 平台概述

AirFogSim 是一个空地协同边缘计算仿真平台，以 `simulation_interval` 为单位进行离散时间步模拟。每经过一个 `simulation_interval`，仿真会进行一次"系统状态更新 + 算法调度"的循环。

### 关键特性
- **仿真时间步长**：以 `simulation_interval` 为单位（例如1.0秒）
- **仿真实体**：车辆(Vehicle)、无人机(UAV)、路边单元(RSU)、云服务器(CloudServer)
- **核心功能**：任务卸载、资源调度、通信管理、计算分配

---

## 核心架构

### 主要组件
- **AirFogSimEnv**：仿真环境的核心类
- **TaskManager**：任务管理器
- **ChannelManager**：信道管理器  
- **TrafficManager**：交通管理器
- **AlgorithmModule**：算法模块

### 实体管理
- `vehicles`：车辆节点字典
- `UAVs`：无人机节点字典
- `RSUs`：路边单元字典
- `cloudServers`：云服务器字典

---

## 仿真流程

### 每个时间步的执行流程

```
1. 环境状态更新
   ├── 任务生成
   ├── UAV移动
   └── 网络拓扑变化

2. 算法调度（scheduleStep()）
   ├── scheduleOffloading() - 任务卸载决策
   ├── scheduleCommunication() - 通信资源分配
   ├── scheduleComputing() - 计算资源分配
   └── scheduleUAV() - UAV轨迹优化

3. 系统执行
   ├── 任务调度
   ├── 资源分配
   └── 结果记录
```

### step()函数详细流程
```python
def step(self):
    # 1. 更新交通状态
    self._updateTraffics()
    
    # 2. 更新AI模型
    self._updateAIModels()
    
    # 3. 更新传感器
    self._updateSensor()
    
    # 4. 在每个交通步长内执行多个仿真步骤
    while 交通步长内:
        # 更新认证和隐私
        # 更新任务
        self._updateTask()
        # 更新无线通信
        self._updateWirelessCommunication()
        # 更新计算
        self._updateComputation()
        # 更新存储、能源、区块链
        
    # 5. 更新仿真时间并记录状态
```

---

## 任务生命周期

### 任务状态变迁图
```
to_generate → to_offload → offloading → computing → computed → returning → finished
                   ↓
              out_of_ddl (超时)
                   ↓
                removed
```

### 任务生成过程

#### 1. 任务生成模型
- **支持模型**：Poisson、Uniform、Normal、Exponential
- **生成参数**：`task_profile: {'lambda': 0.5, 'dag_edge_prob': 0.3}`
- **生成间隔**：每 `predictable_seconds` 集中生成一次任务（默认2秒）

#### 2. 任务属性生成
- **CPU需求**：任务所需的计算资源
- **任务大小**：上传和返回的数据量
- **截止时间**：任务的deadline
- **优先级**：任务的执行优先级

#### 3. 任务依赖关系
- 使用 `networkx.DiGraph` 管理任务依赖关系
- 支持DAG（有向无环图）结构
- 父任务失败会导致子任务无法执行

### 任务卸载过程

#### 2.1 卸载决策（OffloadingDecision）
由 [`scheduleOffloading`](airfogsim_algorithm.py) 函数实现：
1. 获取所有需要卸载的任务
2. 识别任务源节点（task_node_id）
3. 获取源节点附近的节点（最多5个，按距离排序）
4. 选择最近的节点作为卸载目标
5. 通过 [`taskScheduler.setTaskOffloading`]() 设置卸载目标

#### 2.2 通信资源分配（ResourceBlockAllocation）
[`scheduleCommunication`](airfogsim_algorithm.py) 函数：
1. 获取可用资源块(RB)总数
2. 获取所有正在卸载的任务
3. 平均分配资源块给每个任务
4. 设置每个任务的通信资源分配

#### 2.3 通信执行（CommunicationExecution）
[`_updateWirelessCommunication`](airfogsim_env.py) 函数执行实际数据传输：

**a. 分配通信资源**
- [`_allocate_communication_RBs`](airfogsim_env.py) 将资源块与通信链路关联
- 确定通信双方、通信类型（如V2I表示车辆到基础设施）

**b. 计算通信速率**
- [`_compute_communication_rate`](airfogsim_env.py) 计算各通信链路的数据传输速率

**c. 执行通信过程**
- [`_execute_communication`](airfogsim_env.py) 执行实际数据传输
- 根据通信速率在一个时间步长内传输相应数据
- 完成传输则标记为成功卸载，出错则标记为失败

#### 2.4 计算资源分配与执行
- [`scheduleComputing`](airfogsim_algorithm.py)：分配CPU资源
- [`_updateComputation`](airfogsim_env.py)：执行实际计算
- 采用简单的平均分配策略

#### 2.5 结果返回
- [`scheduleReturning`](airfogsim_algorithm.py)：设置返回路径
- 支持直接返回和中继返回两种方式

---

## 核心组件详解

### Task类（任务类）

#### 基本属性
```python
class Task:
    # 任务基本信息
    _task_id              # 任务唯一标识符
    _task_node_id         # 任务初始节点ID
    _task_cpu             # 所需CPU资源
    _task_size            # 任务数据大小
    _task_deadline        # 截止时间
    _task_priority        # 优先级
    _task_arrival_time    # 到达时间
    
    # 状态和路由
    _to_offload_route     # 卸载路径
    _to_return_route      # 返回路径
    _routes               # 经过的节点列表
    _routed_time          # 到达各节点的时间
    
    # 计算和传输状态
    _computed_size        # 已完成计算量
    _transmitted_size     # 已传输数据量
    _start_to_compute_time    # 开始计算时间
    _start_to_transmit_time   # 开始传输时间
```

#### 核心方法
- [`compute(allocated_cpu, simulation_interval, current_time)`](task.py:182)：执行计算
- [`transmit_to_Node(node_id, trans_data, current_time)`](task.py:259)：数据传输
- [`offloadTo(node_id, route, time)`](task.py:292)：任务卸载

#### 任务计算模式
每个 `simulation_interval` 执行一次 [`compute`](task.py) 函数：
```python
self._computed_size += allocated_cpu * simulation_interval
```

### TaskManager类（任务管理类）

#### 任务状态分类
```python
class TaskManager:
    _generated_task_history = {}        # 已生成任务历史
    _to_generate_task_infos = {}        # 待生成任务
    _waiting_to_offload_tasks = {}      # 等待卸载任务
    _offloading_tasks = {}              # 正在卸载任务
    _computing_tasks = {}               # 正在计算任务
    _waiting_to_return_tasks = {}       # 等待返回任务
    _returning_tasks = {}               # 正在返回任务
    _done_tasks = {}                    # 已完成任务
    _out_of_ddl_tasks = {}              # 超时任务
    _removed_tasks = {}                 # 已移除任务
```

#### 主要功能
1. **任务生成**：支持多种生成模型，动态生成任务及属性
2. **状态管理**：跟踪任务生命周期，从生成到完成或失败
3. **依赖管理**：通过DAG管理任务间依赖关系
4. **任务操作**：支持卸载、计算、返回等操作
5. **任务查询**：提供多种查询任务状态和历史的方法

### AirFogSimEnv类（环境类）

#### 核心管理器
- [`traffic_manager`](airfogsim_env.py)：交通管理器，负责车辆和UAV移动
- [`task_manager`](airfogsim_env.py)：任务管理器，负责任务生成和调度
- [`channel_manager`](airfogsim_env.py)：信道管理器，负责无线通信
- [`energy_manager`](airfogsim_env.py)：能源管理器
- [`node_state_manager`](airfogsim_env.py)：节点状态信息管理器

#### 关键方法解析
- [`__init__(config, interactive_mode)`](airfogsim_env.py)：初始化环境和各组件
- [`step()`](airfogsim_env.py)：执行一个仿真时间步长
- [`reset()`](airfogsim_env.py)：重置环境至初始状态
- [`_updateTraffics()`](airfogsim_env.py)：更新交通状态
- [`_updateTask()`](airfogsim_env.py)：更新和生成任务

---

## 算法模块

### BaseAlgorithmModule类

#### 核心调度流程
[`scheduleStep`](airfogsim_algorithm.py) 执行完整调度步骤：
```python
def scheduleStep(self, env: AirFogSimEnv):
    self.scheduleReturning(env)      # 任务返回路由调度
    self.scheduleOffloading(env)     # 任务卸载调度  
    self.scheduleCommunication(env)  # 通信资源调度
    self.scheduleComputing(env)      # 计算资源调度
    self.scheduleMission(env)        # 任务分配调度
    self.scheduleTraffic(env)        # 交通控制调度
```

#### 调度策略详解

**任务卸载策略（scheduleOffloading）**
- 获取源节点附近节点（最多5个，按距离排序）
- 选择最近节点作为卸载目标
- "最近节点优先"策略

**通信资源分配（scheduleCommunication）**
- 获取可用资源块总数和正在卸载的任务
- 平均分配资源块给每个任务

**计算资源分配（scheduleComputing）**
- 每个计算节点最多处理3个任务
- 平均分配CPU资源给任务
- 限制因子：节点任务数量上限

**返回路由选择（scheduleReturning）**
- 计算当前节点与所有RSU的距离
- 选择最近RSU作为返回路由

### NVHAUAlgorithmModule类

继承自 [`BaseAlgorithmModule`](airfogsim_algorithm.py)，实现更复杂策略：

**任务分配扩展**
- 50%概率分配给UAV，50%分配给车辆
- UAV：选择精度最低但满足要求的传感器
- 车辆：选择感知范围内且精度满足的最近传感器

**返回策略扩展**
- 车辆节点：50%概率选择UAV中继返回，50%直接返回
- UAV节点：直接选择最近RSU返回

---

## 常见问题与解答

### 1. CPU占用率不满但有任务等待的原因

**主要原因：**
在 [`scheduleComputing`](airfogsim_algorithm.py) 函数中，每个计算节点最多只处理3个任务，即使节点有足够CPU资源处理更多任务。当任务数超过限制时，多余任务会处于等待状态。

**次要原因：**
在 [`scheduleOffloading`](airfogsim_algorithm.py) 中，任务卸载总是选择最近节点，不考虑节点当前负载。这可能导致热门节点（位置优越）接收大量任务，而其他节点闲置。

### 2. 节点类型与数量确定

**生成节点和计算节点可以是同一个节点**
参见 [`airfogsim_env.py:92`](airfogsim_env.py:92) 注释说明。

**节点数量确定机制：**

**生成节点数量控制**
- 配置文件控制每种类型节点的最大数量(`max_node_num`)
- 定义节点成为任务节点的概率(`task_node_gen_poss`)
- 新节点加入时的判断条件：
  - 该类型节点在允许的任务节点类型中
  - 随机数小于成为任务节点的概率
  - 该类型任务节点数量未达到上限
  - 该节点不在当前任务节点列表中

**计算节点数量**
所有不是任务节点的节点都可以作为计算节点，参见 [`airfogsim_env.py:267`](airfogsim_env.py:267)。

### 3. 时间与性能

**任务总时间计算**
```
任务总时间 = 任务传输时间 + 任务计算时间 + 任务等待时间
```

**任务失败条件**
```
if 任务总时间 > task_deadline:
    任务失败
```

**平台时间机制**
- 以 `simulation_interval` 为单位的离散时间步
- 每个时间步执行完整的调度和执行流程
- 可通过 [`environment.getSimulationTime()`]() 获取当前仿真时间戳

---

## 总结

AirFogSim是一个功能完整的空地协同边缘计算仿真平台，具有以下特点：

1. **模块化设计**：环境、任务管理、算法模块分离，便于扩展
2. **完整的任务生命周期**：从生成到完成的全流程管理
3. **灵活的调度策略**：支持自定义算法模块
4. **丰富的实体类型**：车辆、无人机、RSU、云服务器
5. **全面的仿真功能**：通信、计算、存储、能源等

该平台适用于边缘计算、任务调度、资源分配等研究领域，为空地协同场景提供了完整的仿真环境。