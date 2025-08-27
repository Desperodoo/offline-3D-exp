#!/bin/bash

# EPIC探索系统强制终止脚本
# 用于清理所有相关进程和RViz

echo "==================== EPIC 探索系统终止脚本 ===================="

# 定义要终止的进程关键字
PROCESS_KEYWORDS=(
    "exploration_node"
    "rviz"
    "roslaunch.*exploration_manager.*exploration_dungeon"
    "roslaunch.*exploration_manager.*exploration"
    "fast_exploration"
)

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 函数：检查进程是否存在
check_process() {
    local keyword="$1"
    pgrep -f "$keyword" > /dev/null
    return $?
}

# 函数：优雅终止进程
graceful_kill() {
    local keyword="$1"
    local pids=$(pgrep -f "$keyword")
    
    if [ -n "$pids" ]; then
        echo -e "${YELLOW}发现进程 '$keyword': $pids${NC}"
        echo "发送 SIGTERM 信号..."
        pkill -TERM -f "$keyword"
        
        # 等待最多10秒让进程优雅退出
        for i in {1..10}; do
            if ! check_process "$keyword"; then
                echo -e "${GREEN}进程 '$keyword' 已优雅退出${NC}"
                return 0
            fi
            sleep 1
            echo -n "."
        done
        echo ""
        
        # 如果还存在，强制终止
        if check_process "$keyword"; then
            echo -e "${RED}进程 '$keyword' 未响应，强制终止...${NC}"
            pkill -KILL -f "$keyword"
            sleep 1
            
            if check_process "$keyword"; then
                echo -e "${RED}警告: 进程 '$keyword' 可能仍在运行${NC}"
                return 1
            else
                echo -e "${GREEN}进程 '$keyword' 已强制终止${NC}"
                return 0
            fi
        fi
    else
        echo -e "${BLUE}未发现进程 '$keyword'${NC}"
        return 0
    fi
}

# 主程序
echo "正在搜索并终止探索相关进程..."

killed_any=false

# 逐个终止进程
for keyword in "${PROCESS_KEYWORDS[@]}"; do
    if check_process "$keyword"; then
        graceful_kill "$keyword"
        killed_any=true
    fi
done

# 特殊处理：终止所有RViz进程
if pgrep "rviz" > /dev/null; then
    echo -e "${YELLOW}发现RViz进程，正在终止...${NC}"
    pkill -TERM rviz
    sleep 2
    if pgrep "rviz" > /dev/null; then
        pkill -KILL rviz
        echo -e "${GREEN}RViz已强制终止${NC}"
    else
        echo -e "${GREEN}RViz已优雅退出${NC}"
    fi
    killed_any=true
fi

# 检查roscore状态
if pgrep "roscore\|rosmaster" > /dev/null; then
    echo -e "${BLUE}检测到roscore正在运行${NC}"
    read -p "是否也要终止roscore? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}终止roscore...${NC}"
        pkill -TERM -f "roscore\|rosmaster"
        sleep 2
        pkill -KILL -f "roscore\|rosmaster" 2>/dev/null
        echo -e "${GREEN}roscore已终止${NC}"
    fi
fi

# 清理共享内存（如果有的话）
if command -v ipcs >/dev/null 2>&1; then
    echo "清理共享内存..."
    ipcs -m | grep $USER | awk '{print $2}' | xargs -I {} ipcrm -m {} 2>/dev/null || true
fi

# 最终检查
echo ""
echo "================ 最终状态检查 ================"
remaining_processes=false

for keyword in "${PROCESS_KEYWORDS[@]}"; do
    if check_process "$keyword"; then
        echo -e "${RED}警告: 进程 '$keyword' 仍在运行${NC}"
        pgrep -f "$keyword" | head -5
        remaining_processes=true
    fi
done

if [ "$remaining_processes" = false ]; then
    if [ "$killed_any" = true ]; then
        echo -e "${GREEN}✓ 所有探索相关进程已成功终止${NC}"
    else
        echo -e "${GREEN}✓ 未发现运行中的探索进程${NC}"
    fi
else
    echo -e "${RED}✗ 部分进程可能仍在运行，请手动检查${NC}"
    echo "可以尝试运行: ps aux | grep -E 'exploration|rviz'"
fi

echo "==================== 脚本执行完成 ===================="
