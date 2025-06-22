general_prompt = """"I am working with a simulation platform called AirFogSim, which is designed for modeling and evaluating air-fog-cloud collaborative computing systems. This platform focuses on key problems such as UAV coordination, task offloading, radio resource (RB) allocation, and computing resource scheduling in integrated aerial and fog computing environments.

The architecture of AirFogSim consists of several modular components:

Environment: Simulates the network and task environment.

Schedulers: Manage task scheduling, communication, computing, and UAV mobility.

Algorithm Layer: Implements optimization strategies for decision-making.

Entity Modeling: Defines mobile devices, fog nodes, UAVs, and tasks.

Algorithm development is done by extending the BaseAlgorithmModule class. Key methods include:

Each method relies on APIs from corresponding schedulers (e.g., entity_scheduler, task_scheduler, comm_scheduler, comp_scheduler) to gather environmental data and entity states. All interactions between the algorithm and the environment are abstracted to ensure modularity and prevent direct state mutation.

Based on this platform structure, please help me derive relevant mathematical formulas or optimization objectives for algorithm design. For example, task offloading could involve minimizing latency and energy consumption under bandwidth and CPU constraints; UAV trajectory planning may optimize coverage and service continuity. Formulas should be clear, grounded in the system context, and aligned with real-time scheduling scenarios."""

config = """simulation:
  vehicle_count: 50
  max_simulation_time: 20
  # 每 0.1 秒进行一次调度，比如任务生成、计算、传输等。
  simulation_interval: 0.1 # in seconds, for transmission, computation, task generation and assignment, etc.
  # 同样是每 0.1 秒模拟一次交通（要和 SUMO 设置一致）。
  traffic_interval: 0.1 # in seconds, for traffic simulation. Remember to set the same value in the SUMO configuration file; otherwise, the simulation will not work properly
  
# 使用了五角场的 osm.sumocfg 配置文件进行交通仿真
sumo:
  sumo_config: "./sumo_wujiaochang/osm.sumocfg"
  sumo_osm: "./sumo_wujiaochang/osm_bbox.osm.xml"
  sumo_net: "./sumo_wujiaochang/osm.net.xml"
  sumo_port: 8813
  export_tripinfo: False # 如果True，则导出tripinfo.xml文件（很大）
  tripinfo_output: "./sumo_wujiaochang/tripinfo.xml"

# 可视化图标的路径
visualization:
  icon_path: "./icon"

traffic:
  traffic_mode: "SUMO" # "real" or "SUMO"
  tripinfo: "./sumo_wujiaochang/tripinfo.csv" # The path to the SUMO tripinfo file
  traffic_interval: 0.1 # in seconds, for traffic simulation. Remember to set the same value in the SUMO configuration file; otherwise, the simulation will not work properly
  # UAV 初始飞行高度 [100,200]，速度范围 [10,30]。
  UAV_z_range: [100, 200] # The range of z coordinates for UAVs
  UAV_speed_range: [10,30] # The speed of UAV (distance unit per timeslot)
  uav_traffic_file: "path/to/uav_traffic_file.csv" # UAV traffic file
  max_n_vehicles: 100 # Number of vehicles to simulate
  max_n_UAVs: 10 # Number of UAVs to simulate. UAVs are randomly generated in the simulation area initially
  RSU_positions: [[100, 100, 0], [100, 2000, 0], [2500, 100, 0], [2500, 2000, 0]] # List of RSU positions, each RSU is represented by a list of x, y, z coordinates
  max_n_cloudServers: 1 # Number of cloud servers
  # 泊松分布中车辆到达速率（每秒 1 辆）。
  arrival_lambda: 1 # The rate at which vehicles arrive at the network per second (in Poisson distribution)
  # 判断节点是否到达目标位置的误差阈值。
  distance_threshold: 50 # The distance threshold to check if the node is reach target position
  nonfly_zone_coordinates: [[[1000, 1200], [1200, 1200], [1200, 1000], [1000, 1000]],[[1500, 1700], [1700, 1700], [1700, 1500], [1500, 1500]]] # The non-fly zones in the simulation area, each zone is represented by a list of x1, y1, x2, y2, x3, y3 coordinates

task: # default task profile for all nodes
  # 任务调度的 TTI（通信时延）容忍阈值。
  tti_threshold: 0.5 # The threshold of TTI for task offloading
  hard_ddl: 2 # The hard deadline for tasks
  # 使用泊松分布（Poisson）生成任务，平均每 10 秒生成一个任务。
  task_generation_model: "Poisson" # supported model:['Poisson', 'Uniform', 'Normal', 'Exponential']
  task_generation_kwargs:
    lambda: 0.1 # Poission -> lambda; Uniform -> low, high; Normal -> mean, std; Exponential -> beta
  # 每个任务计算需求是 1.5（单位视代码而定，如GHz·s）。
  task_min_cpu: 1.5
  task_max_cpu: 1.5
  # 每个任务大小是 1.5（单位视代码而定，如MB）。
  task_min_size: 1.5
  task_max_size: 1.5
  task_min_required_returned_size: 0
  task_max_required_returned_size: 0
  # 每个任务的硬性截止时间是 2秒。
  task_min_deadline: 2
  task_max_deadline: 2
  task_min_priority: 1
  task_max_priority: 1
  cpu_model: "Uniform" # supported model:['Uniform', 'Normal']
  cpu_kwargs:
    low: 0.7
    high: 0.7
  size_model: "Uniform" # supported model:['Uniform', 'Normal']
  size_kwargs:
    low: 0.7
    high: 0.7
  deadline_model: "Uniform" # supported model:['Uniform', 'Normal']
  deadline_kwargs:
    low: 1
    high: 1
  priority_model: "Uniform" # supported model:['Uniform', 'Normal']
  priority_kwargs:
    low: 1
    high: 1
  required_returned_size_model: "Uniform" # supported model:['Uniform', 'Normal']
  required_returned_size_kwargs:
    low: 0
    high: 0

task_profile: # task profile for each node type
  # 初始化时每个节点有 50% 概率是“任务节点”。 
  task_node_gen_poss: 0.5 # The possibility of generating a task node when inializing the node
  # 每种类型最多生成多少任务节点。
  task_node_profiles: [{'type':'UAV', 'max_node_num': 30}, {'type':'vehicle', 'max_node_num': 20}] # The types of task nodes and the maximum number of nodes of each type
  vehicle: # The profile of vehicles
    lambda: 0.1
    dag_edge_prob: 0.3 # The probability of generating an edge in the DAG of task graph
  uav: # The profile of UAVs
    lambda: 0.1
    dag_edge_prob: 0.3

# 让所有实体的算力减小，提高失败概率
fog_profile:
  vehicle:
    cpu: 1 # CPU capacity of vehicles
    memory: 1 # Memory capacity of vehicles
    storage: 1 # Storage capacity of vehicles
  uav:
    cpu: 2 # CPU capacity of UAVs
    memory: 1 # Memory capacity of UAVs
    storage: 1 # Storage capacity of UAVs
  rsu:
    cpu: 5 # CPU capacity of RSUs
    memory: 1 # Memory capacity of RSUs
    storage: 1 # Storage capacity of RSUs
  cloud:
    cpu: 20 # CPU capacity of cloud servers
    memory: 1 # Memory capacity of cloud servers
    storage: 1 # Storage capacity of cloud servers

state_attribute:
  log_state: False # Whether to log the state attributes
  time_window: 3 # The time window for state attributes statistics
  fog_node_state_attributes: ['position_x', 'position_y', 'position_z', 'speed', 'fog_profile', 'node_type']
  task_node_state_attributes: ['position_x', 'position_y', 'position_z', 'speed', 'task_profile', 'node_type']
  task_state_attributes: ['task_node_id', 'task_size', 'task_cpu', 'required_returned_size', 'task_deadline', 'task_priority', 'task_arrival_time', 'task_lifecycle_state']

# 通信信道模型
channel:
  outage_model: 'Rayleigh' # supported model:['Rayleigh'], further implementation can be added in airfogsim/channel_callback/*_callback.py
  # 若 SNR < 10，就认为通信中断（outage）。
  outage_snr_threshold: 10 # The SNR threshold for outage detection
  V2V:
    pathloss_model: 'V2V_urban_tr37885'
    shadowing_model: '3GPP_LogNormal'
    fastfading_model: 'Rayleigh'
  V2I:
    pathloss_model: 'UMa_LOS_tr38901'
    shadowing_model: '3GPP_LogNormal'
    fastfading_model: 'Rayleigh'
  V2U:
    pathloss_model: 'free_space'
    shadowing_model: '3GPP_LogNormal'
    fastfading_model: 'Rayleigh'
  U2I:
    pathloss_model: 'free_space'
    shadowing_model: '3GPP_LogNormal'
    fastfading_model: 'Rayleigh'
  U2U:
    pathloss_model: 'free_space'
    shadowing_model: '3GPP_LogNormal'
    fastfading_model: 'Rayleigh'
  I2I:
    pathloss_model: 'free_space'
    shadowing_model: '3GPP_LogNormal'
    fastfading_model: 'Rayleigh'

sensing:
  # 每个 UAV/vehicle 上可以搭载最多 4 个传感器。
  sensors_per_node: 4 # The maximum number of sensors on a node
  node_type: ['UAV','vehicle']
  sensor_type_num: 4

# UAV 任务的执行高度、生命周期（TTL）、任务时长、数据大小等都可以从分布中随机生成。
mission:
  UAV_height: 100
  TTL_range: [100,200]
  duration_range: [5,10]
  mission_size_range: [10,20]
  sensor_accuracy_range: [0,1]
  distance_threshold: 150
  predictable_seconds: 2
  
  mission_generation_model: "Poisson" # supported model:['Poisson', 'Uniform', 'Normal', 'Exponential']
  generation_model_args:
    Poisson:
      lambda: 1
    Uniform:
      low: 0
      high: 1
    Normal:
      mean: 0
      std: 1
    Exponential:
      beta: 1

# 能量消耗模型，UAV 执行任务时电量是否够，将影响任务是否成功完成。
energy:
  initial_energy_range: [48000,52000]
  fly_unit_cost: 2.5
  hover_unit_cost: 1.2
  sensing_unit_cost: 0.5
  receive_unit_cost: 0.1
  send_unit_cost: 0.1"""

All_intersection_coordinates ='''
{'10487419-AddedOffRampNode': (2218.33, 1824.99), 
 '1123309048-AddedOffRampNode': (828.97, 1363.94), 
 '11999265104': (1623.04, 1996.52), 
 '1290439802#0-AddedOnRampNode': (1842.39, 1711.5), 
 '1290439804-AddedOnRampNode': (2356.25, 1855.71), 
 '1290439805#1-AddedOffRampNode': (1875.69, 1681.72), 
 '1290439805#1-AddedOnRampNode': (1854.47, 1674.82), 
 '1294773295-AddedOnRampNode': (1351.43, 1527.12), 
 '136362667#0-AddedOnRampNode': (829.59, 1319.18), 
 '1494769964': (253.88, 2178.51), 
 '1781735772': (558.13, 1830.67), 
 '1781735803': (681.89, 1922.49), 
 '1781735921': (554.38, 2041.89), 
 '1781939174': (561.82, 2050.36), 
 '1782735824': (1442.2, 2019.81), 
 '1782735842': (716.69, 1923.74), 
 '1871924355': (284.06, 1124.89), 
 '1871924379': (574.31, 1235.63), 
 '1871924414': (1445.93, 1559.83), 
 '1874530665': (1496.73, 1107.59), 
 '1876002870': (2092.13, 1573.83), 
 '24474366#3-AddedOnRampNode': (1530.75, 1013.56), 
 '2477681335': (565.34, 1035.97), 
 '266073268': (1399.81, 1513.49), '266073277': (1305.71, 1452.11), '266073292': (1361.67, 1446.83), '266073387': (1703.29, 580.24), '266073460': (1303.51, 153.12), '266073660': (557.1, 796.47), '296250717': (829.12, 1038.63), '298517010': (1181.05, 1848.3), '298517193': (1701.73, 2022.58), '298517194': (2024.55, 2131.67), '300744528': (2316.87, 817.08), '301862189': (2072.15, 2141.57), '301867172': (1441.22, 1569.85), '301874385': (1285.44, 1884.77), '301874388': (1235.7, 1989.15), '39150988#0-AddedOffRampNode': (1356.62, 1512.71), '39151773-AddedOnRampNode': (1697.23, 1864.51), '4322760749': (700.59, 1282.63), '4322760750': (733.33, 1292.56), '4322760824': (734.86, 1330.22), '4322760830': (677.89, 1328.07), '4342992774': (1451.03, 1545.67), '4342993100': (1698.07, 1988.14), '4342993107': (1693.82, 1964.42), '4342993116': (1598.28, 1399.21), '4342993130': (1743.94, 950.35), '4352514142': (1509.77, 1113.24), '4426438804': (1759.37, 1643.91), '4426438807': (1770.65, 1234.05), '4426450249': (1707.6, 1832.61), '461741552': (1907.65, 1705.59), '468731792': (1290.86, 1520.45), '468734868': (697.62, 1288.24), '468735237': (692.37, 1298.36), '468735907': (365.55, 1179.0), '468738040': (1421.04, 1529.27), '468741333': (2182.67, 1825.97), '468742514': (1701.55, 2004.56), '468811709': (1349.7, 1561.5), '469770338': (1565.37, 1695.35), '469770340': (1479.6, 1949.05), '469770345': (1407.3, 1642.33), '469770347': (1718.96, 1744.59), '472187895': (1326.39, 816.57), '472195891': (1700.92, 2137.84), '475384381': (0.0, 1576.98), '475845271': (1538.74, 1778.0), '475851328': (401.67, 886.61), '475851862': (485.19, 926.48), '475851871': (685.06, 1029.82), '475852014': (980.11, 949.75), '475852541': (592.02, 981.71), '475852543': (648.3, 873.23), '475852675': (733.1, 952.04), '486095071': (2469.45, 285.15), '558627333': (1202.28, 1856.32), '558631236': (1171.01, 866.12), '600129548': (778.27, 1332.13), '600130459': (1903.03, 1719.45), '600131735': (2123.59, 1792.98), '600132035': (1937.01, 1743.85), '600133474': (1971.39, 1710.74), '600133489': (2261.58, 1823.49), '600463137': (979.21, 1773.63), '600463140': (1118.02, 1460.73), '600512469': (1020.55, 1972.53), '600512526': (1111.8, 1995.56), '601150139': (2907.69, 1257.37), '601150148': (2901.51, 1820.47), '601626794': (1858.16, 637.07), '601626795': (1959.81, 362.41), '74427164#2-AddedOffRampNode': (1528.11, 1965.1), '826167789': (462.74, 1767.12), '826167995': (982.69, 2047.85), '826168011': (493.99, 1775.18), '826169818': (740.87, 1863.31), '826170194': (738.19, 1950.69), '826170716': (892.7, 1915.8), '826171421': (861.92, 1997.8), '826171425': (843.05, 2047.52), '826173462': (709.16, 2048.5), '86687514': (912.05, 1378.21), '92613981': (917.36, 1364.12), '9721775502': (1387.22, 1470.63), '9721775503': (1331.11, 1440.92), '9721775504': (1292.03, 1473.19), '9721775505': (1306.04, 1540.14), '9721775506': (1374.57, 1554.75), '9721777797': (1743.61, 1637.09), '9721777798': (1735.49, 1634.93), '9721777799': (1734.74, 1640.53), '9721777800': (1742.55, 1642.75), '9933045795': (1800.77, 1006.67), '9933045802': (1876.0, 1513.0), 'cluster_296250721_558624827': (1585.89, 887.73), 'cluster_4342993109_4342993124_4342993125': (1755.15, 1477.82), 'cluster_468728952_558631235': (1069.68, 910.77), 'cluster_4698504529_4698504532_4698504533': (1591.85, 1628.53), 'cluster_475851165_9873945210': (891.73, 533.73), 'cluster_9721777795_9721777796': (1733.04, 1671.98)}'''