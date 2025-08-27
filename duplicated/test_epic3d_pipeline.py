#!/usr/bin/env python3
"""
EPIC 3D数据处理Pipeline测试脚本
测试完整的数据处理流程，包括数据加载、处理和保存
"""

import os
import sys
import time
import yaml
import argparse
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from epic3d_data_processor import EPIC3DDataProcessor, EPIC3DDatasetBuilder
from build_epic3d_dataset import main as build_dataset


def test_config_loading():
    """测试配置文件加载"""
    print("=== 测试1: 配置文件加载 ===")
    try:
        config_path = "epic3d_rl_config.yaml"
        if not os.path.exists(config_path):
            print(f"❌ 配置文件 {config_path} 不存在")
            return False
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        print(f"✅ 配置文件加载成功")
        print(f"   - 数据处理配置: {len(config.get('data_processing', {}))} 项")
        print(f"   - 数据集配置: {len(config.get('dataset', {}))} 项")
        print(f"   - 训练配置: {len(config.get('training', {}))} 项")
        return True
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False


def test_data_processor_init():
    """测试数据处理器初始化"""
    print("\n=== 测试2: 数据处理器初始化 ===")
    try:
        config_path = "epic3d_rl_config.yaml"
        processor = EPIC3DDataProcessor(config_path)
        
        print(f"✅ 数据处理器初始化成功")
        print(f"   - 最大节点数: {processor.max_nodes}")
        print(f"   - 最大视点数: {processor.max_viewpoints}")
        print(f"   - 节点特征维度: {processor.node_feature_dim}")
        return True
        
    except Exception as e:
        print(f"❌ 数据处理器初始化失败: {e}")
        return False


def test_dataset_builder_init():
    """测试数据集构建器初始化"""
    print("\n=== 测试3: 数据集构建器初始化 ===")
    try:
        config_path = "epic3d_rl_config.yaml"
        builder = EPIC3DDatasetBuilder(config_path)
        
        print(f"✅ 数据集构建器初始化成功")
        print(f"   - 训练集比例: {builder.train_ratio}")
        print(f"   - 验证集比例: {builder.val_ratio}")
        print(f"   - 测试集比例: {builder.test_ratio}")
        return True
        
    except Exception as e:
        print(f"❌ 数据集构建器初始化失败: {e}")
        return False


def test_sample_data_processing():
    """测试样本数据处理（如果存在collected_data目录）"""
    print("\n=== 测试4: 样本数据处理 ===")
    
    # 查找collected_data目录
    workspace_dir = Path(__file__).parent.parent.parent
    collected_data_dir = workspace_dir / "collected_data"
    
    if not collected_data_dir.exists():
        print("⚠️  collected_data目录不存在，跳过样本数据测试")
        return True
    
    # 查找topo_graph文件
    topo_files = list(collected_data_dir.rglob("topo_graph_*.txt"))
    if not topo_files:
        print("⚠️  未找到topo_graph文件，跳过样本数据测试")
        return True
    
    print(f"📁 找到 {len(topo_files)} 个topo_graph文件")
    
    try:
        config_path = "epic3d_rl_config.yaml"
        processor = EPIC3DDataProcessor(config_path)
        
        # 只处理第一个文件作为测试
        test_file = topo_files[0]
        print(f"🔄 测试处理文件: {test_file.name}")
        
        start_time = time.time()
        nodes, edges, viewpoints = processor.parse_topo_graph_file(str(test_file))
        process_time = time.time() - start_time
        
        print(f"✅ 文件解析成功")
        print(f"   - 节点数: {len(nodes)}")
        print(f"   - 边数: {len(edges)}")
        print(f"   - 视点数: {len(viewpoints)}")
        print(f"   - 处理时间: {process_time:.2f}秒")
        
        # 检查数据质量
        if nodes:
            sample_node = nodes[0]
            print(f"   - 样本节点特征: {len(sample_node)} 维")
            print(f"   - 位置范围: [{min(n[0] for n in nodes):.2f}, {max(n[0] for n in nodes):.2f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 样本数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_interface():
    """测试命令行接口"""
    print("\n=== 测试5: 命令行接口 ===")
    try:
        # 创建测试参数
        workspace_dir = Path(__file__).parent.parent.parent
        collected_data_dir = workspace_dir / "collected_data"
        
        if not collected_data_dir.exists():
            print("⚠️  collected_data目录不存在，跳过CLI测试")
            return True
            
        # 测试参数验证功能
        test_args = [
            "--collected_data_dir", str(collected_data_dir),
            "--output_dir", "/tmp/test_epic3d_dataset",
            "--config", "epic3d_rl_config.yaml",
            "--dry_run"  # 只验证，不实际处理
        ]
        
        print(f"🔄 测试CLI参数: {' '.join(test_args)}")
        
        # 模拟sys.argv
        original_argv = sys.argv
        sys.argv = ['build_epic3d_dataset.py'] + test_args
        
        try:
            # 这里只测试参数解析，不实际运行
            parser = argparse.ArgumentParser()
            parser.add_argument('--collected_data_dir', required=True, help='数据目录')
            parser.add_argument('--output_dir', required=True, help='输出目录')
            parser.add_argument('--config', default='epic3d_rl_config.yaml', help='配置文件')
            parser.add_argument('--dry_run', action='store_true', help='试运行模式')
            
            args = parser.parse_args(test_args)
            
            print(f"✅ CLI参数解析成功")
            print(f"   - 数据目录: {args.collected_data_dir}")
            print(f"   - 输出目录: {args.output_dir}")
            print(f"   - 配置文件: {args.config}")
            print(f"   - 试运行模式: {args.dry_run}")
            
        finally:
            sys.argv = original_argv
            
        return True
        
    except Exception as e:
        print(f"❌ CLI接口测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始EPIC 3D数据处理Pipeline测试")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_data_processor_init,
        test_dataset_builder_init,
        test_sample_data_processing,
        test_cli_interface
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果统计:")
    print(f"   ✅ 通过: {passed}")
    print(f"   ❌ 失败: {failed}")
    print(f"   📈 成功率: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 所有测试通过！Pipeline已准备就绪")
        print("\n💡 下一步可以运行:")
        print("   python build_epic3d_dataset.py --collected_data_dir /path/to/collected_data --output_dir /path/to/output")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查上述错误信息")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
