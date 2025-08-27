#!/bin/bash
"""
数据采集脚本使用示例
"""

EPIC_DIR="/home/amax/EPIC"
SCRIPT_PATH="${EPIC_DIR}/data_collection_script.py"

echo "=== EPIC 数据自动采集脚本 ==="
echo "项目目录: ${EPIC_DIR}"
echo ""

# 显示使用方法
echo "使用方法:"
echo "1. 单次随机采集 (5分钟):"
echo "   python3 ${SCRIPT_PATH} -t 300"
echo ""
echo "2. 批量采集 (3次，每次5分钟):"
echo "   python3 ${SCRIPT_PATH} -c 3 -t 300"
echo ""
echo "3. 指定地图类型采集:"
echo "   python3 ${SCRIPT_PATH} --map-type forest -t 300"
echo ""
echo "4. 列出所有可用配置:"
echo "   python3 ${SCRIPT_PATH} --list-configs"
echo ""
echo "5. 快速测试 (1分钟):"
echo "   python3 ${SCRIPT_PATH} -t 60"
echo ""

# 检查是否有命令行参数
if [ $# -eq 0 ]; then
    echo "请选择操作 (输入数字):"
    echo "1) 单次随机采集 (5分钟)"
    echo "2) 快速测试 (1分钟)"
    echo "3) 批量采集 (3次)"
    echo "4) 列出可用配置"
    echo "5) 退出"
    
    read -p "选择: " choice
    
    case $choice in
        1)
            echo "开始单次随机采集..."
            python3 "${SCRIPT_PATH}" -t 300
            ;;
        2)
            echo "开始快速测试..."
            python3 "${SCRIPT_PATH}" -t 60
            ;;
        3)
            echo "开始批量采集..."
            python3 "${SCRIPT_PATH}" -c 3 -t 300
            ;;
        4)
            echo "列出可用配置..."
            python3 "${SCRIPT_PATH}" --list-configs
            ;;
        5)
            echo "退出"
            exit 0
            ;;
        *)
            echo "无效选择"
            exit 1
            ;;
    esac
else
    # 直接执行传入的参数
    python3 "${SCRIPT_PATH}" "$@"
fi
