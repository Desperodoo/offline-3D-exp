#!/bin/bash

# EPIC 数据自动采集脚本
# 功能：
# 1. 随机选择地图和初始位置
# 2. 修改launch文件和配置
# 3. 启动ROS进行数据采集
# 4. 自动终止并过滤数据

# 默认配置
EPIC_DIR="/home/amax/EPIC"
COLLECTION_TIME=900
MAP_TYPE=""
COUNT=1
VERBOSE=false

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# 帮助信息
show_help() {
    cat << EOF
EPIC 数据自动采集脚本

用法: $0 [选项]

选项:
  -h, --help              显示此帮助信息
  -t, --time SECONDS      单次采集时长(秒) (默认: 300)
  -c, --count COUNT       采集次数 (默认: 1)
  -m, --map-type TYPE     指定地图类型 (forest|partition|dungeon)
  -d, --dir PATH          EPIC项目目录 (默认: /home/amax/EPIC)
  -v, --verbose           显示详细日志
  --list-configs          列出所有可用配置
  --test-detection        测试探索完成检测系统

功能说明:
  • 智能探索完成检测: 监控ROS topics和节点状态自动判断探索是否完成
  • 安全退出机制: 启用EPIC系统的自动停止和安全退出功能
  • 多级检测策略: 综合topic消息、FSM状态、心跳信号等多种指标
  • 文件生成监控: 结合数据文件生成情况判断探索进度

示例:
  $0 -t 60 -m dungeon     # 60秒dungeon地图采集
  $0 -c 5 -t 120          # 5次随机采集，每次120秒
  $0 --test-detection     # 测试检测系统功能
  $0 --list-configs       # 列出所有配置
EOF
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -t|--time)
                COLLECTION_TIME="$2"
                shift 2
                ;;
            -c|--count)
                COUNT="$2"
                shift 2
                ;;
            -m|--map-type)
                MAP_TYPE="$2"
                shift 2
                ;;
            -d|--dir)
                EPIC_DIR="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --list-configs)
                list_available_configs
                exit 0
                ;;
            --test-detection)
                test_detection_system
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 测试探索状态检测系统
test_detection_system() {
    log_info "测试探索状态检测系统..."
    
    # 检查ROS环境
    if ! pgrep -f "roscore" > /dev/null; then
        log_warn "roscore未运行，启动roscore进行测试..."
        roscore &
        local roscore_pid=$!
        sleep 3
    fi
    
    echo ""
    echo "=== 探索状态检测测试 ==="
    echo ""
    
    # 测试基本检测功能
    log_info "1. 测试基本状态检测..."
    local status=$(check_exploration_status)
    echo "   当前检测状态: $status"
    
    # 测试topic监听能力
    log_info "2. 测试ROS topic访问..."
    
    echo "   测试 /exploration/auto_stop_status ..."
    timeout 3s rostopic echo /exploration/auto_stop_status -n 1 &>/dev/null
    if [ $? -eq 0 ]; then
        echo "   ✓ topic可访问"
    else
        echo "   ✗ topic不存在或无法访问"
    fi
    
    echo "   测试 /planning/state ..."
    timeout 3s rostopic echo /planning/state -n 1 &>/dev/null
    if [ $? -eq 0 ]; then
        echo "   ✓ topic可访问"
    else
        echo "   ✗ topic不存在或无法访问"
    fi
    
    echo "   测试 /planning/heartbeat ..."
    timeout 3s rostopic echo /planning/heartbeat -n 1 &>/dev/null
    if [ $? -eq 0 ]; then
        echo "   ✓ topic可访问"
    else
        echo "   ✗ topic不存在或无法访问"
    fi
    
    # 测试参数访问
    log_info "3. 测试ROS参数访问..."
    
    local test_params=(
        "/exploration_node/fsm/enable_auto_stop"
        "/exploration_node/fsm/enable_safe_exit"
        "/exploration_node/fsm/max_exploration_time"
        "/exploration_node/fsm/safe_exit_delay"
    )
    
    for param in "${test_params[@]}"; do
        local value=$(rosparam get "$param" 2>/dev/null || echo "N/A")
        echo "   $param = $value"
    done
    
    # 测试节点检查
    log_info "4. 测试节点状态检查..."
    if rosnode info /exploration_node &>/dev/null; then
        echo "   ✓ exploration_node存在并可访问"
        echo "   节点信息:"
        rosnode info /exploration_node 2>/dev/null | head -10 | sed 's/^/     /'
    else
        echo "   ✗ exploration_node不存在或无法访问"
    fi
    
    echo ""
    echo "=== 智能监控模拟测试 ==="
    echo ""
    
    # 模拟监控过程（短时间）
    log_info "5. 模拟监控过程（30秒）..."
    local test_start=$(date +%s)
    local test_end=$((test_start + 30))
    local test_count=0
    
    while [ $(date +%s) -lt $test_end ]; do
        local current=$(date +%s)
        local elapsed=$((current - test_start))
        local status=$(check_exploration_status)
        
        printf "\r   [%02ds/30s] 状态: %-20s 检查次数: %d" $elapsed "$status" $test_count
        
        ((test_count++))
        sleep 5
    done
    
    echo ""
    echo ""
    log_info "✓ 检测系统测试完成"
    echo ""
    echo "如果看到错误或topic不可访问，这是正常的（因为探索系统未运行）"
    echo "在实际探索过程中，这些topic和参数应该是可用的"
    
    # 清理测试用的roscore
    if [ -n "$roscore_pid" ]; then
        kill $roscore_pid 2>/dev/null
        log_info "清理测试环境"
    fi
}

# 验证环境
validate_environment() {
    log_info "验证环境..."
    
    if [ ! -d "$EPIC_DIR" ]; then
        log_error "EPIC目录不存在: $EPIC_DIR"
        exit 1
    fi
    
    if [ ! -d "$EPIC_DIR/src/MARSIM/map_generator/resource" ]; then
        log_error "地图资源目录不存在: $EPIC_DIR/src/MARSIM/map_generator/resource"
        exit 1
    fi
    
    # 检查必要的ROS工具
    for cmd in roscore roslaunch rosparam rosnode; do
        if ! command -v $cmd &> /dev/null; then
            log_error "ROS工具 $cmd 未找到，请确保ROS环境已正确设置"
            exit 1
        fi
    done
    
    # 检查Python脚本
    if [ ! -f "$EPIC_DIR/filter_topo_v2.py" ]; then
        log_warn "过滤脚本不存在: $EPIC_DIR/filter_topo_v2.py"
    fi
    
    log_info "环境验证完成"
}

# 列出所有可用配置
list_available_configs() {
    log_info "扫描可用配置..."
    
    local count=0
    for map_type in forest partition dungeon; do
        local resource_dir="$EPIC_DIR/src/MARSIM/map_generator/resource/$map_type"
        if [ -d "$resource_dir" ]; then
            for batch_dir in "$resource_dir"/batch_*; do
                if [ -d "$batch_dir" ]; then
                    local batch_id=$(basename "$batch_dir")
                    for txt_file in "$batch_dir"/*_map_free_points.txt; do
                        if [ -f "$txt_file" ]; then
                            local point_id=$(basename "$txt_file" | cut -d'_' -f1)
                            echo "  $map_type - $batch_id/$point_id"
                            ((count++))
                        fi
                    done
                fi
            done
        fi
    done
    
    if [ $count -eq 0 ]; then
        log_error "未找到任何配置文件"
        exit 1
    fi
    
    log_info "总计找到 $count 个可用配置"
}

# 随机选择配置
select_random_config() {
    local map_type="$1"
    local configs=()
    
    # 收集所有配置
    for mt in forest partition dungeon; do
        if [ -n "$map_type" ] && [ "$mt" != "$map_type" ]; then
            continue
        fi
        
        local resource_dir="$EPIC_DIR/src/MARSIM/map_generator/resource/$mt"
        if [ -d "$resource_dir" ]; then
            for batch_dir in "$resource_dir"/batch_*; do
                if [ -d "$batch_dir" ]; then
                    local batch_id=$(basename "$batch_dir")
                    for txt_file in "$batch_dir"/*_map_free_points.txt; do
                        if [ -f "$txt_file" ]; then
                            local point_id=$(basename "$txt_file" | cut -d'_' -f1)
                            configs+=("$mt:$batch_id:$point_id:$txt_file")
                        fi
                    done
                fi
            done
        fi
    done
    
    if [ ${#configs[@]} -eq 0 ]; then
        log_error "未找到可用的配置"
        exit 1
    fi
    
    # 随机选择
    local selected=${configs[$RANDOM % ${#configs[@]}]}
    echo "$selected"
}

# 从文件中随机选择位置
select_random_position() {
    local txt_file="$1"
    
    if [ ! -f "$txt_file" ]; then
        log_error "位置文件不存在: $txt_file"
        exit 1
    fi
    
    # 过滤注释和空行，随机选择一行
    local position=$(grep -v '^#' "$txt_file" | grep -v '^$' | shuf -n 1)
    if [ -z "$position" ]; then
        log_error "位置文件中没有有效位置: $txt_file"
        exit 1
    fi
    
    echo "$position"
}

# 创建输出目录
setup_output_directory() {
    local map_type="$1"
    local batch_id="$2"
    local point_id="$3"
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local output_dir="$EPIC_DIR/collected_data/${map_type}_${batch_id}_${point_id}_${timestamp}"
    
    mkdir -p "$output_dir/raw_data"
    
    # 保存配置信息
    cat > "$output_dir/config.txt" << EOF
# 数据采集配置 - $timestamp
map_type: $map_type
batch_id: $batch_id
point_id: $point_id
collection_time: $COLLECTION_TIME
script_version: shell_v1.0
EOF
    
    echo "$output_dir"
}

# 修改algorithm.xml
modify_algorithm_xml() {
    local map_type="$1"
    local output_dir="$2"
    
    local original="$EPIC_DIR/src/global_planner/exploration_manager/launch/algorithm.xml"
    local temp="$EPIC_DIR/src/global_planner/exploration_manager/launch/algorithm_temp.xml"
    
    log_debug "修改algorithm.xml: $original -> $temp"
    
    # 复制原文件并修改
    cp "$original" "$temp"
    
    # 修改config_file参数
    sed -i "s/default=\"cave2.yaml\"/default=\"${map_type}.yaml\"/" "$temp"
    
    # 修改export_dir参数
    sed -i "s|value=\"/home/amax/EPIC/topo_outputs\"|value=\"${output_dir}/raw_data\"|" "$temp"
    
    log_info "✓ 修改algorithm.xml: 使用${map_type}.yaml配置，输出到${output_dir}/raw_data" >&2
    
    # 只输出文件路径，不要其他信息
    echo "$temp"
}

# 修改MARSIM launch文件
modify_marsim_launch() {
    local map_type="$1"
    local init_x="$2"
    local init_y="$3"
    local init_z="$4"
    local init_yaw="$5"
    
    local original="$EPIC_DIR/src/MARSIM/mars_drone_sim/launch/${map_type}.launch"
    local temp="$EPIC_DIR/src/MARSIM/mars_drone_sim/launch/${map_type}_temp.launch"
    
    if [ ! -f "$original" ]; then
        log_warn "MARSIM launch文件不存在: $original"
        return
    fi
    
    log_debug "修改MARSIM launch: $original -> $temp"
    
    # 复制原文件并修改初始位置
    cp "$original" "$temp"
    
    # 修改初始位置参数
    sed -i "s/<arg name=\"init_x_\" value=\"[^\"]*\"/<arg name=\"init_x_\" value=\"${init_x}\"/" "$temp"
    sed -i "s/<arg name=\"init_y_\" value=\"[^\"]*\"/<arg name=\"init_y_\" value=\"${init_y}\"/" "$temp"
    sed -i "s/<arg name=\"init_z_\" value=\"[^\"]*\"/<arg name=\"init_z_\" value=\"${init_z}\"/" "$temp"
    sed -i "s/<arg name=\"init_yaw\" value=\"[^\"]*\"/<arg name=\"init_yaw\" value=\"${init_yaw}\"/" "$temp"
    
    # 修改地图路径 - 匹配正确的格式
    sed -i "s|<arg name=\"map_name\" value=\"[^\"]*\"|<arg name=\"map_name\" value=\"\$(find map_generator)/resource/${map_type}.pcd\"|" "$temp"
    
    log_info "✓ 修改MARSIM launch: 位置(${init_x}, ${init_y}, ${init_z}, ${init_yaw}), 地图${map_type}.pcd" >&2
    
    # 只输出文件路径到stdout
    echo "$temp"
}

# 修改epic_planner launch文件
modify_epic_launch() {
    local map_type="$1"
    local temp_algorithm="$2"
    local temp_marsim="$3"
    
    # 选择正确的launch文件进行修改
    local simulation_launch="$EPIC_DIR/src/global_planner/exploration_manager/launch/simulation_${map_type}.launch"
    local temp_simulation="$EPIC_DIR/src/global_planner/exploration_manager/launch/simulation_${map_type}_temp.launch"
    
    if [ ! -f "$simulation_launch" ]; then
        log_error "Simulation launch文件不存在: $simulation_launch"
        exit 1
    fi
    
    log_debug "修改Simulation launch: $simulation_launch -> $temp_simulation"
    
    # 复制并修改simulation launch文件
    cp "$simulation_launch" "$temp_simulation"
    
    # 如果有MARSIM引用，也替换
    if [ -n "$temp_marsim" ] && [ -f "$temp_marsim" ]; then
        sed -i "s|\$(find mars_drone_sim)/launch/${map_type}.launch|${temp_marsim}|" "$temp_simulation"
    fi
    
    # 在simulation launch文件中添加exploration_node
    sed -i '/<\/launch>/i\
    <!-- 探索节点 -->\
  <node name="exploration_node" pkg="epic_planner" type="exploration_node" output="screen">\
    <!-- 自动开始探索参数 -->\
    <param name="fsm/enable_auto_start" value="true"/>\
    <param name="fsm/auto_start_delay" value="10.0"/>\
    <param name="fsm/auto_start_condition" value="map_ready"/>\
    \
    <!-- 自动停止探索参数 -->\
    <param name="fsm/enable_auto_stop" value="true"/>\
    <param name="fsm/max_exploration_time" value="1800.0"/>  <!-- 30分钟 -->\
    <param name="fsm/completion_check_interval" value="5.0"/>  <!-- 5秒检查一次 -->\
    \
    <!-- 安全退出参数 -->\
    <param name="fsm/enable_safe_exit" value="true"/>\
    <param name="fsm/safe_exit_delay" value="15.0"/>  <!-- FINISH状态15秒后退出 -->\
  </node>' "$temp_simulation"
    
    # 现在修改主launch文件以使用临时simulation文件
    local original="$EPIC_DIR/src/global_planner/exploration_manager/launch/${map_type}.launch"
    local temp="$EPIC_DIR/src/global_planner/exploration_manager/launch/${map_type}_temp.launch"
    
    if [ ! -f "$original" ]; then
        log_error "Epic launch文件不存在: $original"
        exit 1
    fi
    
    log_debug "修改Epic launch: $original -> $temp"
    
    # 复制原文件并修改引用
    cp "$original" "$temp"
    
    # 替换algorithm.xml路径
    sed -i "s|\$(find epic_planner)/launch/algorithm.xml|${temp_algorithm}|" "$temp"
    
    # 替换simulation launch文件引用
    sed -i "s|\$(find epic_planner)/launch/simulation_${map_type}.launch|${temp_simulation}|" "$temp"
    
    log_info "✓ 修改Epic launch文件，启用自动停止和安全退出功能" >&2
    
    # 只输出文件路径
    echo "$temp"
}

# 启动roscore
start_roscore() {
    log_info "检查roscore状态..."
    
    if pgrep -f "roscore" > /dev/null; then
        log_info "✓ roscore已在运行"
        return 0
    fi
    
    log_info "启动roscore..."
    roscore &
    local roscore_pid=$!
    echo $roscore_pid > /tmp/epic_roscore.pid
    
    # 等待roscore启动
    local max_wait=30
    local count=0
    while [ $count -lt $max_wait ]; do
        if rostopic list &>/dev/null; then
            log_info "✓ roscore启动成功"
            return 0
        fi
        sleep 1
        ((count++))
    done
    
    log_error "roscore启动超时"
    return 1
}

# 启动ROS launch
start_ros_launch() {
    local temp_launch="$1"
    local output_dir="$2"
    
    cd "$EPIC_DIR"
    
    # 检查launch文件是否存在
    if [ ! -f "$temp_launch" ]; then
        log_error "Launch文件不存在: $temp_launch"
        return 1
    fi
    
    local launch_name=$(basename "$temp_launch")
    log_info "启动ROS launch: $launch_name"
    
    # 启动launch文件，后台运行
    roslaunch epic_planner "$launch_name" > "$output_dir/roslaunch.log" 2>&1 &
    local launch_pid=$!
    echo $launch_pid > /tmp/epic_launch.pid
    
    log_info "Launch进程PID: $launch_pid"
    
    # 等待系统启动
    log_info "等待系统启动..."
    sleep 20
    
    # 验证关键节点
    local max_wait=30
    local count=0
    while [ $count -lt $max_wait ]; do
        if rosnode list 2>/dev/null | grep -q "exploration_node"; then
            log_info "✓ exploration_node已启动"
            break
        fi
        sleep 2
        ((count++))
    done
    
    if [ $count -ge $max_wait ]; then
        log_warn "exploration_node启动检查超时，但继续执行"
    fi
    
    return 0
}

# 检查探索是否完成
check_exploration_status() {
    # 方法1: 检查核心节点是否还在运行
    if ! rosnode info /exploration_node &>/dev/null; then
        echo "NODE_STOPPED"
        return
    fi
    
    # 方法2: 监听自动停止状态topic (最可靠)
    local auto_stop_msg=$(timeout 2s rostopic echo /exploration/auto_stop_status -n 1 2>/dev/null | grep "data:" | cut -d'"' -f2)
    if [ -n "$auto_stop_msg" ]; then
        case "$auto_stop_msg" in
            *"TERMINATED_COMPLETED"*)
                echo "COMPLETED"
                return
                ;;
            *"TERMINATED_NO_FRONTIER"*)
                echo "NO_FRONTIERS"
                return
                ;;
            *"TERMINATED_TIMEOUT"*)
                echo "TIMEOUT"
                return
                ;;
        esac
    fi
    
    # 方法3: 检查FSM当前状态
    local planning_state_msg=$(timeout 2s rostopic echo /planning/state -n 1 2>/dev/null | grep "text:" | cut -d' ' -f2-)
    if [ -n "$planning_state_msg" ]; then
        case "$planning_state_msg" in
            *"FINISH"*)
                # 在FINISH状态，检查是否启用了安全退出
                local safe_exit=$(rosparam get /exploration_node/fsm/enable_safe_exit 2>/dev/null || echo "false")
                if [ "$safe_exit" = "true" ]; then
                    echo "FINISHING"  # 安全退出进程中
                else
                    echo "COMPLETED"  # 完成但可能需要手动终止
                fi
                return
                ;;
            *"PLAN_TRAJ"*|*"EXEC_TRAJ"*)
                echo "ACTIVE"
                return
                ;;
            *"WAIT_TRIGGER"*|*"INIT"*)
                echo "INITIALIZING"
                return
                ;;
            *"CAUTION"*|*"LAND"*)
                echo "EMERGENCY"
                return
                ;;
        esac
    fi
    
    # 方法4: 检查心跳信号
    local heartbeat=$(timeout 2s rostopic echo /planning/heartbeat -n 1 2>/dev/null)
    if [ -z "$heartbeat" ]; then
        echo "INACTIVE"
        return
    fi
    
    # 方法5: 检查是否启用自动停止并检查参数状态
    local auto_stop=$(rosparam get /exploration_node/fsm/enable_auto_stop 2>/dev/null || echo "false")
    if [ "$auto_stop" = "true" ]; then
        local max_time=$(rosparam get /exploration_node/fsm/max_exploration_time 2>/dev/null || echo "1800")
        # 如果能获取到开始时间，计算是否超时
        local exploration_start=$(rosparam get /exploration_node/exploration_start_time 2>/dev/null || echo "")
        if [ -n "$exploration_start" ]; then
            local current_time=$(date +%s)
            local start_seconds=$(echo "$exploration_start" | cut -d'.' -f1)
            local elapsed=$((current_time - start_seconds))
            if [ $elapsed -gt $(echo "$max_time" | cut -d'.' -f1) ]; then
                echo "TIMEOUT_EXPECTED"
                return
            fi
        fi
    fi
    
    # 方法6: 检查launch进程状态
    if [ -f /tmp/epic_launch.pid ]; then
        local launch_pid=$(cat /tmp/epic_launch.pid)
        if ! kill -0 $launch_pid 2>/dev/null; then
            echo "PROCESS_STOPPED"
            return
        fi
    fi
    
    # 默认认为还在运行
    echo "ACTIVE"
}

# 监控探索完成状态
monitor_exploration_completion() {
    local output_dir="$1"
    local max_time="$2"
    
    log_info "开始智能监控探索状态，最大时长: ${max_time}秒"
    
    local start_time=$(date +%s)
    local end_time=$((start_time + max_time))
    local file_count_start=$(find "$output_dir/raw_data" -name "topo_graph_*.txt" 2>/dev/null | wc -l)
    local last_file_count=$file_count_start
    local no_new_files_count=0
    local completion_check_count=0
    local finishing_count=0
    local inactive_count=0
    local check_interval=2  # 改为2秒检查一次，提高响应速度

    # 等待探索真正开始 (避免误检测初始化阶段)
    log_info "等待探索系统完全启动..."
    local startup_wait=0
    while [ $startup_wait -lt 10 ]; do  # 最多等待20秒
        local status=$(check_exploration_status)
        if [ "$status" = "ACTIVE" ]; then
            log_info "✓ 探索系统已激活，开始监控"
            break
        fi
        printf "\r启动检查: %ds - 状态: %s" $((startup_wait * 5)) "$status"
        sleep 5
        ((startup_wait++))
    done
        echo ""
    
    while [ $(date +%s) -lt $end_time ]; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        local remaining=$((end_time - current_time))
        
        # 检查文件生成情况
        local current_count=$(find "$output_dir/raw_data" -name "topo_graph_*.txt" 2>/dev/null | wc -l)
        local new_files=$((current_count - file_count_start))
        
        # 检查是否有新文件生成
        if [ $current_count -eq $last_file_count ]; then
            ((no_new_files_count++))
        else
            no_new_files_count=0
            last_file_count=$current_count
        fi
        
        # 检查探索状态
        local status=$(check_exploration_status)
        
        # 显示详细进度信息
        printf "\r[%ds/%ds] 文件:%d个 | 状态:%s | 静止:%ds | 剩余:%ds" \
               $elapsed $max_time $new_files "$status" $((no_new_files_count * check_interval)) $remaining
        
        # 根据状态决定是否退出
        case "$status" in
            "COMPLETED"|"NO_FRONTIERS")
                echo ""
                log_info "✓ 探索已完成（状态: $status），强制终止进程"
                force_terminate_exploration
                break
                ;;
            "TIMEOUT")
                echo ""
                log_info "✓ 探索因超时终止，强制清理进程"
                force_terminate_exploration
                break
                ;;
            "NODE_STOPPED"|"PROCESS_STOPPED")
                echo ""
                log_info "✓ 探索节点已停止（状态: $status），确认进程清理"
                force_terminate_exploration
                break
                ;;
            "FINISHING")
                ((finishing_count++))
                if [ $finishing_count -ge 8 ]; then  # FINISH状态持续16秒就强制退出
                    echo ""
                    log_info "✓ 探索处于FINISH状态过久，强制终止"
                    force_terminate_exploration
                    break
                fi
                ;;
            "INACTIVE")
                ((inactive_count++))
                if [ $inactive_count -ge 6 ]; then  # 12秒无心跳就开始检查
                    echo ""
                    log_info "⚠ 探索节点无响应超过12秒，检查是否已完成"
                    if [ $new_files -gt 10 ]; then
                        log_info "✓ 已收集到足够数据，判断探索已完成"
                        log_info "✓ 强制终止探索进程以确保退出"
                        # 强制终止ROS进程确保退出
                        force_terminate_exploration
                        break
                    else
                        log_warn "⚠ 数据量较少(${new_files}个)，可能刚开始或有问题"
                        if [ $inactive_count -ge 15 ]; then  # 30秒后即使数据少也退出
                            log_info "✓ 长时间无响应，强制退出"
                            force_terminate_exploration
                            break
                        fi
                    fi
                fi
                ;;
            "EMERGENCY")
                echo ""
                log_warn "⚠ 探索进入紧急状态，继续监控"
                ;;
            "TIMEOUT_EXPECTED")
                echo ""
                log_info "✓ 探索即将因超时终止，等待系统自动退出"
                sleep 10  # 等待系统自动处理
                break
                ;;
            "ACTIVE"|"INITIALIZING")
                # 重置所有异常计数器
                completion_check_count=0
                finishing_count=0
                inactive_count=0
                ;;
        esac
        
        # 检查是否长时间没有新文件且有足够的数据 - 降低阈值更快退出
        if [ $no_new_files_count -ge 15 ] && [ $new_files -gt 20 ]; then  # 30秒无新文件且有足够数据
            echo ""
            log_info "✓ 长时间无新文件生成且已收集足够数据，判断探索已完成"
            # 最后检查一次状态确认
            local final_check=$(check_exploration_status)
            log_info "✓ 最终状态检查：$final_check，强制退出"
            # 不管状态如何，都强制退出
            force_terminate_exploration
            break
        fi
        
        sleep $check_interval
    done
    
    echo ""
    if [ $(date +%s) -ge $end_time ]; then
        log_warn "⚠ 达到最大采集时间，强制停止采集"
    fi
    
    # 最终状态检查和文件统计
    local final_status=$(check_exploration_status)
    local final_count=$(find "$output_dir/raw_data" -name "topo_graph_*.txt" 2>/dev/null | wc -l)
    local total_files=$((final_count - file_count_start))
    
    log_info "监控完成 - 最终状态: $final_status, 生成文件: $total_files 个"
}

# 强制终止探索进程
force_terminate_exploration() {
    log_info "强制终止探索相关进程..."
    
    # 终止exploration_node
    pkill -f "exploration_node" 2>/dev/null && log_debug "✓ 终止exploration_node" || true
    
    # 终止roslaunch进程
    pkill -f "roslaunch.*epic_planner" 2>/dev/null && log_debug "✓ 终止roslaunch" || true
    
    # 终止其他相关进程
    pkill -f "simulation.*launch" 2>/dev/null || true
    
    # 等待进程完全终止
    sleep 2
    
    # 如果还有进程，使用KILL信号
    pkill -9 -f "exploration_node" 2>/dev/null || true
    pkill -9 -f "roslaunch.*epic_planner" 2>/dev/null || true
    
    log_info "✓ 强制终止完成"
}

# 清理ROS进程
cleanup_ros_processes() {
    log_info "清理ROS进程..."
    
    # 终止launch进程
    if [ -f /tmp/epic_launch.pid ]; then
        local launch_pid=$(cat /tmp/epic_launch.pid)
        if kill -0 $launch_pid 2>/dev/null; then
            log_debug "终止launch进程: $launch_pid"
            kill -TERM $launch_pid 2>/dev/null || true
            sleep 3
            kill -KILL $launch_pid 2>/dev/null || true
        fi
        rm -f /tmp/epic_launch.pid
    fi
    
    # 强制清理相关进程
    pkill -f "roslaunch.*epic_planner" 2>/dev/null || true
    pkill -f "exploration_node" 2>/dev/null || true
    
    # 终止roscore（谨慎操作）
    if [ -f /tmp/epic_roscore.pid ]; then
        local roscore_pid=$(cat /tmp/epic_roscore.pid)
        if kill -0 $roscore_pid 2>/dev/null; then
            log_debug "终止roscore进程: $roscore_pid"
            kill -TERM $roscore_pid 2>/dev/null || true
            sleep 2
        fi
        rm -f /tmp/epic_roscore.pid
    fi
    
    sleep 2
    log_info "✓ ROS进程清理完成"
}

# 过滤数据
filter_data() {
    local output_dir="$1"
    
    log_info "开始数据过滤..."
    
    local raw_dir="$output_dir/raw_data"
    local filtered_dir="$output_dir/filtered_data"
    
    if [ ! -d "$raw_dir" ]; then
        log_warn "原始数据目录不存在: $raw_dir"
        return
    fi
    
    local file_count=$(find "$raw_dir" -name "topo_graph_*.txt" | wc -l)
    if [ $file_count -eq 0 ]; then
        log_warn "未找到拓扑数据文件"
        return
    fi
    
    log_info "找到 $file_count 个原始文件"
    
    # 使用过滤脚本
    local filter_script="$EPIC_DIR/filter_topo_v2.py"
    if [ -f "$filter_script" ]; then
        mkdir -p "$filtered_dir"
        
        log_info "运行数据过滤脚本..."
        if python3 "$filter_script" "$raw_dir" -o "$filtered_dir" -t 0.5; then
            log_info "✓ 数据过滤完成"
            
            local filtered_count=$(find "$filtered_dir" -name "*.txt" 2>/dev/null | wc -l)
            log_info "过滤后文件数: $filtered_count"
        else
            log_warn "数据过滤失败，保留原始数据"
        fi
    else
        log_warn "过滤脚本不存在: $filter_script"
    fi
}

# 清理临时文件
cleanup_temp_files() {
    log_debug "清理临时文件..."
    
    # 清理launch文件
    rm -f "$EPIC_DIR"/src/global_planner/exploration_manager/launch/*_temp.launch
    rm -f "$EPIC_DIR"/src/global_planner/exploration_manager/launch/algorithm_temp.xml
    rm -f "$EPIC_DIR"/src/MARSIM/mars_drone_sim/launch/*_temp.launch
    
    log_debug "✓ 临时文件清理完成"
}

# 单次数据采集
run_single_collection() {
    local map_type="$1"
    
    log_info "=== 开始单次数据采集 ==="
    
    # 选择配置
    local config=$(select_random_config "$map_type")
    IFS=':' read -r selected_map batch_id point_id txt_file <<< "$config"
    
    log_info "选择配置: $selected_map - $batch_id/$point_id"
    
    # 选择初始位置
    local position=$(select_random_position "$txt_file")
    read -r init_x init_y <<< "$position"
    local init_z="1.0"
    local init_yaw="0.0"
    
    log_info "初始位置: ($init_x, $init_y, $init_z, $init_yaw)"
    
    # 设置输出目录
    local output_dir=$(setup_output_directory "$selected_map" "$batch_id" "$point_id")
    log_info "输出目录: $output_dir"
    
    # 记录配置到输出目录
    cat >> "$output_dir/config.txt" << EOF
init_position: $init_x $init_y $init_z
init_yaw: $init_yaw
pcd_file: $selected_map.pcd
txt_file: $txt_file
EOF
    
    # 修改launch文件
    local temp_algorithm=$(modify_algorithm_xml "$selected_map" "$output_dir")
    local temp_marsim=$(modify_marsim_launch "$selected_map" "$init_x" "$init_y" "$init_z" "$init_yaw")
    local temp_launch=$(modify_epic_launch "$selected_map" "$temp_algorithm" "$temp_marsim")
    
    local success=false
    
    # 改进的陷阱处理，确保数据保存
    cleanup_and_exit() {
        echo ""  # 换行
        log_info "收到中断信号，正在安全退出..."
        
        # 清理ROS进程
        cleanup_ros_processes
        
        # 尝试保存已收集的数据
        if [ -d "$output_dir/raw_data" ]; then
            filter_data "$output_dir"
            
            local topo_count=$(find "$output_dir/raw_data" -name "topo_graph_*.txt" 2>/dev/null | wc -l)
            local filtered_count=$(find "$output_dir/filtered_data" -name "*.txt" 2>/dev/null | wc -l || echo "0")
            
            log_info "中断前已收集数据:"
            log_info "  输出目录: $output_dir"
            log_info "  原始文件: $topo_count 个"
            log_info "  过滤后文件: $filtered_count 个"
        fi
        
        # 清理临时文件
        cleanup_temp_files
        
        exit 130  # 标准的Ctrl+C退出码
    }
    
    # 设置陷阱处理
    trap cleanup_and_exit INT TERM
    
    try_collection() {
        # 启动roscore
        if ! start_roscore; then
            return 1
        fi
        
        # 启动ROS系统
        if ! start_ros_launch "$temp_launch" "$output_dir"; then
            return 1
        fi
        
        # 监控探索完成状态
        monitor_exploration_completion "$output_dir" "$COLLECTION_TIME"
        
        success=true
        return 0
    }
    
    if try_collection; then
        log_info "数据采集完成，开始后处理..."
        
        # 清理ROS进程
        cleanup_ros_processes
        
        # 过滤数据
        filter_data "$output_dir"
        
        # 统计结果
        local topo_count=$(find "$output_dir/raw_data" -name "topo_graph_*.txt" 2>/dev/null | wc -l)
        local filtered_count=$(find "$output_dir/filtered_data" -name "*.txt" 2>/dev/null | wc -l || echo "0")
        
        log_info "✓ 采集完成:"
        log_info "  输出目录: $output_dir"
        log_info "  原始文件: $topo_count 个"
        log_info "  过滤后文件: $filtered_count 个"
        
    else
        log_error "数据采集失败"
        cleanup_ros_processes
    fi
    
    # 清理临时文件
    cleanup_temp_files
    
    # 清理陷阱
    trap - INT TERM
    
    if [ "$success" = true ]; then
        return 0
    else
        return 1
    fi
}

# 批量数据采集
run_batch_collection() {
    local count="$1"
    local map_type="$2"
    
    log_info "=== 开始批量数据采集 (共${count}次) ==="
    
    local successful=0
    local failed=0
    
    for ((i=1; i<=count; i++)); do
        log_info "--- 第 $i/$count 次采集 ---"
        
        if run_single_collection "$map_type"; then
            ((successful++))
            log_info "✓ 第 $i 次采集成功"
        else
            ((failed++))
            log_error "✗ 第 $i 次采集失败"
        fi
        
        # 批量采集间隔
        if [ $i -lt $count ]; then
            local wait_time=10
            log_info "等待 ${wait_time}秒 后进行下次采集..."
            sleep $wait_time
        fi
    done
    
    log_info "=== 批量采集完成 ==="
    log_info "成功: $successful 次"
    log_info "失败: $failed 次"
    log_info "输出目录: $EPIC_DIR/collected_data"
}

# 主函数
main() {
    # 解析参数
    parse_args "$@"
    
    # 验证环境
    validate_environment
    
    # 切换到工作目录
    cd "$EPIC_DIR"
    
    # 执行采集
    if [ $COUNT -eq 1 ]; then
        run_single_collection "$MAP_TYPE"
    else
        run_batch_collection "$COUNT" "$MAP_TYPE"
    fi
}

# 脚本入口
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
