#!/usr/bin/env python3
"""
EPIC 3D 文件管理分析
"""

import os
from pathlib import Path

def analyze_epic3d_files():
    """分析EPIC 3D相关文件的必要性"""
    
    print("=" * 60)
    print("EPIC 3D 文件管理分析")
    print("=" * 60)
    
    # 核心必需文件 (生产环境)
    core_files = {
        "/home/amax/EPIC/src/offline/epic3d_data_processor.py": "核心数据处理模块",
        "/home/amax/EPIC/src/offline/build_epic3d_dataset.py": "数据集构建脚本", 
        "/home/amax/EPIC/src/offline/epic3d_rl_config.yaml": "配置文件",
        "/home/amax/EPIC/src/offline/EPIC3D_Data_Processing_Design.md": "设计文档",
        "/home/amax/EPIC/src/offline/EPIC3D_System_Completion_Report.md": "完成报告"
    }
    
    # 辅助工具文件 (可选)
    utility_files = {
        "/home/amax/EPIC/epic3d_data_adapter.py": "格式转换适配器 (备用)"
    }
    
    # 测试和调试文件 (可以移除)
    test_debug_files = {
        "/home/amax/EPIC/duplicated/epic3d_demo.py": "演示脚本",
        "/home/amax/EPIC/duplicated/epic3d_final_test.py": "最终测试脚本",
        "/home/amax/EPIC/duplicated/test_epic3d_pipeline.py": "流程测试",
        "/home/amax/EPIC/duplicated/test_real_data.py": "真实数据测试",
        "/home/amax/EPIC/duplicated/test_rewards.py": "奖励测试",
        "/home/amax/EPIC/duplicated/debug_stats.py": "统计调试",
        "/home/amax/EPIC/duplicated/format_alignment_checker.py": "格式对齐检查",
        "/home/amax/EPIC/format_checker.py": "格式检查器",
        "/home/amax/EPIC/debug_viewpoint_benefits.py": "视点调试"
    }
    
    # 临时输出文件 (可以删除)
    temp_files = {
        "/home/amax/EPIC/src/offline/test_outputs/": "测试输出目录",
        "/tmp/epic3d_*.h5": "临时HDF5文件",
        "/tmp/test_*.h5": "临时测试文件"
    }
    
    print("📦 核心必需文件 (保留):")
    for file_path, description in core_files.items():
        status = "✅" if os.path.exists(file_path) else "❌"
        print(f"  {status} {file_path}")
        print(f"      {description}")
    
    print(f"\n🔧 辅助工具文件 (可选保留):")
    for file_path, description in utility_files.items():
        status = "✅" if os.path.exists(file_path) else "❌"
        print(f"  {status} {file_path}")
        print(f"      {description}")
    
    print(f"\n🧪 测试和调试文件 (可以移除/已移除):")
    for file_path, description in test_debug_files.items():
        status = "✅" if os.path.exists(file_path) else "❌"
        print(f"  {status} {file_path}")
        print(f"      {description}")
    
    print(f"\n🗑️  临时文件 (应该清理):")
    for file_pattern, description in temp_files.items():
        print(f"  📁 {file_pattern}")
        print(f"      {description}")
    
    # 检查还需要移动的文件
    files_to_move = []
    
    # 检查根目录下的epic3d相关文件
    root_files = [
        "/home/amax/EPIC/epic3d_data_adapter.py",
        "/home/amax/EPIC/format_checker.py", 
        "/home/amax/EPIC/debug_viewpoint_benefits.py"
    ]
    
    print(f"\n📋 需要清理的文件:")
    for file_path in root_files:
        if os.path.exists(file_path):
            files_to_move.append(file_path)
            print(f"  🔄 {file_path} -> ~/EPIC/duplicated/")
    
    # 检查src/offline下的非核心文件
    offline_files = [
        "/home/amax/EPIC/src/offline/data_collector.py",  # 来自offlineRL-exp复制
        "/home/amax/EPIC/src/offline/trainer_ddp.py",    # 来自offlineRL-exp复制
        "/home/amax/EPIC/src/offline/graph_buffer.py",   # 来自offlineRL-exp复制
        "/home/amax/EPIC/src/offline/exploration.py",    # 不相关
        "/home/amax/EPIC/src/offline/sgformer.py",       # 不相关
        "/home/amax/EPIC/src/offline/worker.py"          # 来自offlineRL-exp复制
    ]
    
    for file_path in offline_files:
        if os.path.exists(file_path):
            files_to_move.append(file_path)
            print(f"  🔄 {file_path} -> ~/EPIC/duplicated/")
    
    return files_to_move

if __name__ == "__main__":
    files_to_move = analyze_epic3d_files()
    
    if files_to_move:
        print(f"\n💡 建议操作:")
        print(f"总共需要移动 {len(files_to_move)} 个文件到 ~/EPIC/duplicated/")
        print("是否执行移动操作？")
    else:
        print(f"\n✅ 所有文件已正确整理，无需额外操作。")
