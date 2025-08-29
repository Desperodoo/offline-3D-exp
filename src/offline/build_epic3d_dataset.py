#!/usr/bin/env python3
"""
EPIC 3D数据集构建脚本

使用方法:
python build_epic3d_dataset.py --data_dirs /path/to/data1 /path/to/data2 --output dataset.h5 --config config.yaml
"""

import argparse
import logging
import yaml
import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from epic3d_data_processor import EPIC3DDatasetBuilder, setup_logging

def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise ValueError(f"Failed to load config file {config_path}: {e}")

def validate_config(config: dict) -> dict:
    """验证和补充配置参数"""
    # 确保必要的配置存在
    if 'data_processing' not in config:
        config['data_processing'] = {}
    
    # 设置默认值
    defaults = {
        'max_nodes': 500,
        'max_viewpoints': 100,
        'k_size': 20,
        'node_feature_dim': 9,
        'position_scale': 100.0,
        'observation_score_scale': 50.0,
        'cluster_distance_scale': 20.0,
        'distance_scale': 50.0
    }
    
    for key, default_value in defaults.items():
        if key not in config['data_processing']:
            config['data_processing'][key] = default_value
    
    return config

def validate_data_directories(data_dirs: list) -> list:
    """验证数据目录的存在和有效性"""
    valid_dirs = []
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            logging.warning(f"Data directory does not exist: {data_dir}")
            continue
        
        if not os.path.isdir(data_dir):
            logging.warning(f"Path is not a directory: {data_dir}")
            continue
        
        # 检查是否包含episode子目录
        subdirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
        
        if not subdirs:
            logging.warning(f"No subdirectories found in: {data_dir}")
            continue
        
        valid_dirs.append(data_dir)
        logging.info(f"Valid data directory: {data_dir} (found {len(subdirs)} subdirectories)")
    
    return valid_dirs

def main():
    parser = argparse.ArgumentParser(
        description='Build EPIC3D dataset for offline RL training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # 构建单个数据目录的数据集
  python build_epic3d_dataset.py \\
      --data_dirs /home/amax/EPIC/collected_data \\
      --output /tmp/epic3d_dataset.h5 \\
      --config config.yaml

  # 构建多个数据目录的数据集
  python build_epic3d_dataset.py \\
      --data_dirs /path/to/data1 /path/to/data2 \\
      --output dataset.h5 \\
      --config config.yaml \\
      --log_level DEBUG
        ''')
    
    parser.add_argument('--data_dirs', nargs='+', required=True,
                       help='List of directories containing EPIC3D collected data')
    parser.add_argument('--output', required=True,
                       help='Output HDF5 file path (e.g., dataset.h5)')
    parser.add_argument('--config', required=True,
                       help='Configuration YAML file path')
    parser.add_argument('--log_level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--validate_only', action='store_true',
                       help='Only validate input directories and config, do not build dataset')
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(level=log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("=== EPIC 3D Dataset Builder ===")
    
    try:
        # 验证输入参数
        logger.info("Validating input parameters...")
        
        # 验证配置文件
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        
        config = load_config(args.config)
        config = validate_config(config)
        logger.info(f"Loaded configuration from: {args.config}")
        
        # 验证数据目录
        valid_data_dirs = validate_data_directories(args.data_dirs)
        if not valid_data_dirs:
            raise ValueError("No valid data directories found")
        
        logger.info(f"Found {len(valid_data_dirs)} valid data directories")
        
        # 验证输出路径
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        if args.validate_only:
            logger.info("Validation completed successfully. Exiting without building dataset.")
            return
        
        # 构建数据集
        logger.info("Starting dataset construction...")
        builder = EPIC3DDatasetBuilder(config['data_processing'])
        
        output_path = builder.build_dataset_from_directories(valid_data_dirs, args.output)
        
        logger.info("=== Dataset Construction Completed ===")
        logger.info(f"Dataset saved to: {output_path}")
        logger.info(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        # 输出配置信息
        logger.info("Configuration used:")
        for key, value in config['data_processing'].items():
            logger.info(f"  {key}: {value}")
        
    except KeyboardInterrupt:
        logger.error("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to build dataset: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
