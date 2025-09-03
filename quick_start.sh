#!/bin/bash

# EPIC Docker快速启动脚本

echo "=== EPIC Docker 快速启动 ==="
echo ""

# 检查docker compose是否存在
if ! docker compose version &> /dev/null; then
    echo "错误: Docker Compose V2未安装或不可用"
    exit 1
fi

# 确保数据目录存在
mkdir -p collected_data datasets logs

echo "启动EPIC容器..."
docker compose up -d

echo ""
echo "✅ EPIC容器已启动！"
echo ""
echo "进入容器: docker compose exec epic bash"
echo "在容器内启动ROS:"
echo "  roscore &"
echo "  roslaunch epic_planner garage.launch"
echo ""
echo "停止容器: docker compose down"
