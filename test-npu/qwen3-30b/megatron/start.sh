pkill -9 python
ray stop --force
rm -rf /tmp/ray
rm -rf /root/.cache
#export HCCL_INTRA_PCIE_ENABLE=1
#export HCCL_INTRA_ROCE_ENABLE=0
# 通用环境变量
#export NCCL_IB_DISABLE=1
#export RAY_DEBUG=1  # 允许ray debug
#export RAY_DEBUG_POST_MORTEM=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 #异步
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
export RAY_DEDUP_LOGS=1  # Ray 日志去重
export HYDRA_FULL_ERROR=1

#修改为当前需要跑的用例路径
DEFAULT_SH="$(dirname "$0")/test.sh"
echo "Use $DEFAULT_SH"

ulimit -n 32768
export NNODES=1
export NPUS_PER_NODE=16
#修改为对应主节点IP
export MASTER_ADDR=90.90.97.78
# 修改为当前节点的通信网卡
SOCKET_IFNAME="enp194s0f0"
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME


######################
# GPU相关的环境变量
export NCCL_SOCKET_IFNAME=$SOCKET_IFNAME
# export CUDA_DEVICE_MAX_CONNECTIONS=1   # 利于TP+SP
######################



######################
# NPU相关的环境变量
export ENABLE_PROFILING=0
export ASCEND_PROCESS_LOG_PATH=./logs
#! HCCL 相关配置

#export ASCEND_LAUNCH_BLOCKING=1
export HCCL_BUFFSIZE=2048
export DISABLE_L2_CACHE=1 #规避主存OOM
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export SGLANG_DEEPEP_BF16_DISPATCH=1
#export MULTI_STREAM_MEMORY_REUSE=2

export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export TP_SOCKET_IFNAME=$SOCKET_IFNAME   # NPU？

export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"
export ASCEND_GLOBAL_LOG_LEVEL=3 # 3：error级？0：debug级？

#TASK_QUEUE_ENABLE，下发优化，图模式设置为1，非图模式设置为2。NPU参数？哪个包
export TASK_QUEUE_ENABLE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
# 加入source CANN相关的内容
######################

#################### Log 目录配置 ###########################
# * 确保 JOB_LOG_DIR 在共享盘下
export JOB_LOG_DIR=/mnt/share/m00876805/work/sgl/logs
export JOB_LOG_DIR_CURR=${JOB_LOG_DIR}/$(date +"%Y%m%d_%H%M%S")
mkdir -p $JOB_LOG_DIR_CURR
######################


#获取当前节点IP
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
#CURRENT_IP=141.61.29.107
echo $CURRENT_IP
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  mkdir -p $JOB_LOG_DIR_CURR
  cp $(dirname $0)/start.sh "${JOB_LOG_DIR_CURR}/."
  cp $(dirname $0)/test.sh "${JOB_LOG_DIR_CURR}/."

  # 主节点启动
  ray start --head  --dashboard-host="0.0.0.0" --node-ip-address=$CURRENT_IP --dashboard-port=4919 --disable-usage-stats

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*(NPU|GPU))' | head -n 1)pwd
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # 判断 device_count 是否与 NNODES 相等
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU/GPU resources), starting Python script."
          ray status
          bash $DEFAULT_SH 2>&1 | tee $JOB_LOG_DIR_CURR/test.log
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # 子节点尝试往主节点注册ray直到成功
  while true; do
      # 尝试连接 Ray 集群
      ray start --address="$MASTER_ADDR:6379" --node-ip-address=$CURRENT_IP

      # 检查连接是否成功
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

echo "start.sh ended"
