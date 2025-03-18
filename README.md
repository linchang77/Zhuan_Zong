# 专业综合课程项目
**基于airfogsim的大模型负载预测**

仿真平台采用的是airfogsim，地址为 https://github.com/ZhiweiWei-NAMI/AirFogSim.git

## 如何运行

1. 首先下载docker
2. 构建docker镜像
```bash
docker build -t airfogsim .
```
**过程需要一点时间，有些下载需要挂梯子**

3. 运行容器
```bash
docker run -it --rm airfogsim
```
若需要让 Docker 里的代码和本机同步，可以选择本地挂载

```bash
docker run -it --rm -v $(pwd):/app airfogsim
```



