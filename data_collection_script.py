#!/usr/bin/env python3
"""
自动数据采集脚本
功能：
1. 随机选择地图和初始位置
2. 修改launch文件
3. 启动ROS进行数据采集
4. 自动终止并过滤数据
"""

import os
import sys
import time
import random
import subprocess
import shutil
import signal
import re
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

class DataCollectionScript:
    def __init__(self, epic_dir="/home/amax/EPIC", collection_time=300):
        self.epic_dir = Path(epic_dir)
        self.resource_dir = self.epic_dir / "src/MARSIM/map_generator/resource"
        
        # 使用epic_planner的launch目录，而不是MARSIM的
        self.launch_dir = self.epic_dir / "src/global_planner/exploration_manager/launch"
        
        self.collection_time = collection_time  # 数据采集时间（秒）
        self.output_base_dir = self.epic_dir / "collected_data"
        
        # 可用的地图类型
        self.map_types = ["forest", "partition", "dungeon"]
        
        # ROS进程相关
        self.ros_processes = []
        
        # 数据采集前的时间戳，用于过滤新文件
        self.collection_start_time = None
        
    def get_available_configs(self):
        """获取所有可用的配置（地图+初始位置组合）"""
        configs = []
        
        for map_type in self.map_types:
            map_dir = self.resource_dir / map_type
            if not map_dir.exists():
                continue
                
            # 检查每个batch目录
            for batch_dir in map_dir.glob("batch_*"):
                if not batch_dir.is_dir():
                    continue
                    
                # 查找txt文件
                for txt_file in batch_dir.glob("*_map_free_points.txt"):
                    batch_id = batch_dir.name
                    point_id = txt_file.stem.split('_')[0]
                    
                    configs.append({
                        'map_type': map_type,
                        'batch_id': batch_id,
                        'point_id': point_id,
                        'txt_file': txt_file,
                        'pcd_file': self.resource_dir / f"{map_type}.pcd"
                    })
        
        return configs
    
    def read_initial_positions(self, txt_file):
        """读取初始位置文件中的所有位置"""
        positions = []
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        positions.append((x, y))
        except Exception as e:
            print(f"错误读取位置文件 {txt_file}: {e}")
        
        return positions
    
    def select_random_config(self):
        """随机选择配置"""
        configs = self.get_available_configs()
        if not configs:
            raise ValueError("未找到可用的配置文件")
        
        config = random.choice(configs)
        positions = self.read_initial_positions(config['txt_file'])
        
        if not positions:
            raise ValueError(f"未找到有效的初始位置: {config['txt_file']}")
        
        # 随机选择一个初始位置
        init_x, init_y = random.choice(positions)
        init_z = 1.0  # 默认高度
        init_yaw = 0.0  # 默认朝向
        
        config.update({
            'init_x': init_x,
            'init_y': init_y, 
            'init_z': init_z,
            'init_yaw': init_yaw
        })
        
        return config
    
    def modify_launch_file(self, config):
        """修改launch文件"""
        map_type = config['map_type']
        
        # 使用exploration_manager的launch文件
        original_launch = self.epic_dir / f"src/global_planner/exploration_manager/launch/{map_type}.launch"
        temp_launch = self.epic_dir / f"src/global_planner/exploration_manager/launch/{map_type}_temp.launch"
        
        if not original_launch.exists():
            raise FileNotFoundError(f"Launch文件不存在: {original_launch}")
        
        try:
            # 创建修改后的algorithm.xml
            self.create_modified_algorithm_xml(config)
            
            # 创建修改后的MARSIM launch文件 
            self.create_temp_simulation_launch(config)
            
            # epic_planner的launch文件只需修改algorithm.xml的引用路径
            with open(original_launch, 'r') as f:
                content = f.read()
            
            # 替换algorithm.xml的包含路径为我们修改后的版本
            modified_content = content.replace(
                '$(find epic_planner)/launch/algorithm.xml',
                f'{self.epic_dir}/src/global_planner/exploration_manager/launch/algorithm_temp.xml'
            )
            
            # 如果有MARSIM launch文件的引用，也要替换
            if config.get('temp_marsim_launch'):
                marsim_original = f'$(find mars_drone_sim)/launch/{map_type}.launch'
                marsim_temp = str(config['temp_marsim_launch'])
                modified_content = modified_content.replace(marsim_original, marsim_temp)
            
            with open(temp_launch, 'w') as f:
                f.write(modified_content)
                
        except Exception as e:
            print(f"修改launch文件失败: {e}")
            # 备用方法：直接复制
            shutil.copy2(original_launch, temp_launch)
        
        print(f"✓ 修改launch文件: {temp_launch}")
        print(f"  地图: {config['map_type']} ({config['batch_id']}/{config['point_id']})")
        print(f"  初始位置: ({config['init_x']:.2f}, {config['init_y']:.2f}, {config['init_z']:.2f})")
        
        return temp_launch
    
    def create_modified_algorithm_xml(self, config):
        """创建修改后的algorithm.xml文件，修改export_dir和config_file"""
        output_dir = config['output_dir']
        map_type = config['map_type']
        
        # 原始algorithm.xml路径
        original_algorithm = self.epic_dir / "src/global_planner/exploration_manager/launch/algorithm.xml"
        temp_algorithm = self.epic_dir / "src/global_planner/exploration_manager/launch/algorithm_temp.xml"
        
        try:
            # 读取原始algorithm.xml
            with open(original_algorithm, 'r') as f:
                content = f.read()
            
            # 修改config_file参数以使用正确的yaml配置
            config_file_pattern = r'<arg name="config_file" default="[^"]*"/>'
            new_config_file = f'<arg name="config_file" default="{map_type}.yaml"/>'
            
            if re.search(config_file_pattern, content):
                content = re.sub(config_file_pattern, new_config_file, content)
                print(f"  修改config_file: {map_type}.yaml")
            
            # 修改topo_extraction的export_dir
            export_dir_pattern = r'name="topo_extraction/export_dir"\s+value="[^"]*"'
            new_export_dir = f'name="topo_extraction/export_dir" value="{output_dir}/raw_data"'
            
            if re.search(export_dir_pattern, content):
                modified_content = re.sub(export_dir_pattern, new_export_dir, content)
                print(f"  修改export_dir: {output_dir}/raw_data")
            else:
                # 如果没有找到export_dir参数，在topo_extraction部分添加
                topo_section_pattern = r'(<!-- 拓扑图提取配置 -->)'
                if re.search(topo_section_pattern, content):
                    insert_text = f'''<!-- 拓扑图提取配置 -->
		<param name="topo_extraction/export_dir" value="{output_dir}/raw_data" type="string" />'''
                    modified_content = re.sub(topo_section_pattern, insert_text, content)
                    print(f"  添加export_dir: {output_dir}/raw_data")
                else:
                    modified_content = content
                    print("  警告: 未找到topo_extraction部分，使用原始配置")
            
            with open(temp_algorithm, 'w') as f:
                f.write(modified_content)
            
            config['temp_algorithm'] = temp_algorithm
            print(f"✓ 创建修改后的algorithm.xml: {temp_algorithm}")
            
        except Exception as e:
            print(f"警告: 修改algorithm.xml失败: {e}")
            # 备用：复制原文件
            shutil.copy2(original_algorithm, temp_algorithm)
            config['temp_algorithm'] = temp_algorithm
    
    def create_temp_simulation_launch(self, config):
        """创建临时的MARSIM launch文件，修改初始位置和地图路径"""
        map_type = config['map_type']
        
        # MARSIM的partition.launch文件路径
        original_marsim_launch = self.epic_dir / f"src/MARSIM/mars_drone_sim/launch/{map_type}.launch"
        temp_marsim_launch = self.epic_dir / f"src/MARSIM/mars_drone_sim/launch/{map_type}_temp.launch"
        
        if not original_marsim_launch.exists():
            print(f"警告: MARSIM launch文件不存在: {original_marsim_launch}")
            return
        
        try:
            # 读取原始launch文件
            with open(original_marsim_launch, 'r') as f:
                content = f.read()
            
            # 修改地图路径
            map_path_pattern = r'<arg name="map_name" value="[^"]*" />'
            new_map_path = f'<arg name="map_name" value="$(find map_generator)/resource/{config["pcd_file"].name}" />'
            content = re.sub(map_path_pattern, new_map_path, content)
            
            # 修改初始位置
            init_replacements = [
                (r'<arg name="init_x_" value="[^"]*" />', f'<arg name="init_x_" value="{config["init_x"]}" />'),
                (r'<arg name="init_y_" value="[^"]*" />', f'<arg name="init_y_" value="{config["init_y"]}" />'),
                (r'<arg name="init_z_" value="[^"]*" />', f'<arg name="init_z_" value="{config["init_z"]}" />'),
                (r'<arg name="init_yaw" value="[^"]*" />', f'<arg name="init_yaw" value="{config["init_yaw"]}" />'),
            ]
            
            for pattern, replacement in init_replacements:
                content = re.sub(pattern, replacement, content)
            
            # 保存临时文件
            with open(temp_marsim_launch, 'w') as f:
                f.write(content)
            
            config['temp_marsim_launch'] = temp_marsim_launch
            print(f"✓ 创建修改后的MARSIM launch: {temp_marsim_launch}")
            print(f"  地图: {config['pcd_file'].name}")
            print(f"  初始位置: ({config['init_x']}, {config['init_y']}, {config['init_z']}, {config['init_yaw']})")
            
        except Exception as e:
            print(f"警告: 修改MARSIM launch失败: {e}")
            shutil.copy2(original_marsim_launch, temp_marsim_launch)
            config['temp_marsim_launch'] = temp_marsim_launch
    
    def setup_output_directory(self, config):
        """设置输出目录"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = self.output_base_dir / f"{config['map_type']}_{config['batch_id']}_{config['point_id']}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置信息
        config_file = output_dir / "config.txt"
        with open(config_file, 'w') as f:
            f.write(f"# 数据采集配置 - {timestamp}\n")
            f.write(f"map_type: {config['map_type']}\n")
            f.write(f"batch_id: {config['batch_id']}\n")
            f.write(f"point_id: {config['point_id']}\n")
            f.write(f"init_position: {config['init_x']:.6f} {config['init_y']:.6f} {config['init_z']:.6f}\n")
            f.write(f"init_yaw: {config['init_yaw']:.6f}\n")
            f.write(f"pcd_file: {config['pcd_file']}\n")
            f.write(f"txt_file: {config['txt_file']}\n")
            f.write(f"collection_time: {self.collection_time}\n")
        
        return output_dir
    
    def setup_ros_parameters(self, output_dir):
        """设置ROS参数以启用数据采集功能"""
        print("设置ROS参数...")
        
        # 确保输出目录存在
        raw_data_dir = output_dir / "raw_data"
        raw_data_dir.mkdir(exist_ok=True)
        
        # 只设置必要的参数，自动启动参数应该从yaml配置文件中读取
        params = {
            # 拓扑图提取参数
            '/exploration_node/topo_extraction/export_dir': str(raw_data_dir),
        }
        
        for param, value in params.items():
            try:
                result = subprocess.run(['rosparam', 'set', param, value], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(f"  ✓ 设置 {param} = {value}")
                else:
                    print(f"  ✗ 设置 {param} 失败: {result.stderr}")
            except Exception as e:
                print(f"  ✗ 设置参数 {param} 时出错: {e}")
    
    def setup_ros_parameters_after_launch(self, output_dir):
        """在launch启动后设置ROS参数"""
        print("在launch启动后设置ROS参数...")
        
        # 等待节点启动
        time.sleep(5)
        
        # 设置参数
        params = {
            'topo_extraction/enable_extraction': 'true',
            'topo_extraction/extraction_rate': '2.0',
            'topo_extraction/export_dir': str(output_dir / "raw_data"),
            'topo_extraction/debug_output': 'true',
            'exploration_stats/enable_stats': 'true',
            'exploration_stats/publish_rate': '1.0'
        }
        
        for param, value in params.items():
            try:
                result = subprocess.run(['rosparam', 'set', param, value], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(f"  ✓ 设置 {param} = {value}")
                else:
                    print(f"  ✗ 设置 {param} 失败: {result.stderr}")
            except Exception as e:
                print(f"  ✗ 设置参数 {param} 时出错: {e}")
    
    def start_ros_launch(self, temp_launch, output_dir):
        """启动ROS launch文件"""
        try:
            # 切换到EPIC目录
            os.chdir(self.epic_dir)
            
            # 启动roscore（如果还没有运行）
            try:
                subprocess.check_output(['pgrep', '-f', 'roscore'], stderr=subprocess.DEVNULL)
                print("✓ roscore已在运行")
            except subprocess.CalledProcessError:
                print("启动roscore...")
                roscore_proc = subprocess.Popen(['roscore'], 
                                              stdout=subprocess.DEVNULL, 
                                              stderr=subprocess.DEVNULL)
                self.ros_processes.append(roscore_proc)
                time.sleep(5)  # 等待roscore启动
            
            # 在启动launch之前设置参数
            self.setup_ros_parameters(output_dir)
            
            # 启动launch文件 - 使用epic_planner包
            launch_cmd = [
                'roslaunch', 'epic_planner', temp_launch.name
            ]
            
            print(f"启动launch文件: {' '.join(launch_cmd)}")
            
            launch_proc = subprocess.Popen(launch_cmd,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                                         cwd=str(temp_launch.parent))
            self.ros_processes.append(launch_proc)
            
            # 等待launch文件启动
            print("等待系统启动...")
            time.sleep(20)  # 给更多时间让整个系统启动
            
            # 检查关键ROS参数
            self.check_ros_parameters()
            
            return launch_proc
            
        except Exception as e:
            print(f"错误启动ROS: {e}")
            raise
    
    def check_ros_parameters(self):
        """检查ROS参数设置"""
        print("检查ROS参数设置...")
        
        params_to_check = [
            "/exploration_node/fsm/enable_auto_start",
            "/exploration_node/fsm/auto_start_delay", 
            "/exploration_node/fsm/auto_start_condition",
            "/exploration_node/topo_extraction/enable",
            "/exploration_node/topo_extraction/export_dir",
            "/exploration_node/exploration_stats/enable"
        ]
        
        for param in params_to_check:
            try:
                result = subprocess.run(['rosparam', 'get', param], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    value = result.stdout.strip()
                    print(f"  {param}: {value}")
                else:
                    print(f"  {param}: 未设置")
            except Exception as e:
                print(f"  检查参数 {param} 时出错: {e}")
        
        # 检查运行的ROS节点
        try:
            result = subprocess.run(['rosnode', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                nodes = result.stdout.strip().split('\n')
                print(f"  运行的ROS节点数: {len(nodes)}")
                exploration_nodes = [node for node in nodes if 'exploration' in node.lower() or 'epic' in node.lower()]
                if exploration_nodes:
                    for node in exploration_nodes:
                        print(f"    关键节点: {node}")
                        
                        # 检查节点状态
                        try:
                            info_result = subprocess.run(['rosnode', 'info', node], 
                                                       capture_output=True, text=True, timeout=5)
                            if 'ERROR' not in info_result.stderr:
                                print(f"      节点状态: 正常运行")
                            else:
                                print(f"      节点状态: 可能有问题")
                        except:
                            pass
                else:
                    print("    警告: 未找到exploration相关节点")
            else:
                print("  无法获取ROS节点列表")
        except Exception as e:
            print(f"  检查ROS节点时出错: {e}")
    
    def wait_for_completion(self, process, timeout):
        """等待数据采集完成"""
        print(f"开始数据采集，时长: {timeout}秒")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            # 检查进程是否还在运行
            if process.poll() is not None:
                print("ROS进程提前终止")
                break
                
            time.sleep(1)
            
            # 显示进度
            elapsed = time.time() - start_time
            remaining = max(0, timeout - elapsed)
            print(f"\r采集进度: {elapsed:.1f}s / {timeout}s (剩余 {remaining:.1f}s)", end='', flush=True)
        
        print("\n数据采集完成")
    
    def cleanup_ros_processes(self):
        """清理ROS进程"""
        print("正在终止ROS进程...")
        
        # 首先尝试优雅地终止进程
        for proc in self.ros_processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                except Exception as e:
                    print(f"警告: 终止进程时出错: {e}")
        
        # 强制杀死相关ROS进程
        try:
            subprocess.run(['pkill', '-f', 'roslaunch'], timeout=5)
            subprocess.run(['pkill', '-f', 'roscore'], timeout=5)
            subprocess.run(['pkill', '-f', 'epic_planner'], timeout=5)
        except Exception as e:
            print(f"警告: 强制终止进程时出错: {e}")
        
        time.sleep(2)  # 等待进程完全终止
        self.ros_processes.clear()
        print("✓ ROS进程已清理")
    
    def find_collected_data(self, output_dir=None):
        """查找采集到的数据文件（优先在指定输出目录查找）"""
        print("正在查找数据文件...")
        
        # 构建搜索路径列表，优先级从高到低
        search_paths = []
        
        # 如果指定了输出目录，优先在其raw_data子目录中搜索
        if output_dir:
            search_paths.append(output_dir / "raw_data")
        
        # 其他可能的位置
        search_paths.extend([
            self.epic_dir / "topo_outputs",  # 默认输出目录
            Path("/tmp"),                     # 临时目录（兜底）
            self.epic_dir,                    # EPIC项目根目录
            Path.home()                       # 用户主目录
        ])
        
        topo_files = []
        stats_files = []
        
        for search_path in search_paths:
            if search_path.exists():
                print(f"  搜索路径: {search_path}")
                
                # 查找拓扑图数据
                topo_patterns = ["topo_graph_*.txt", "topo_*.txt"]
                for pattern in topo_patterns:
                    found_files = list(search_path.glob(pattern))
                    for f in found_files:
                        # 检查文件修改时间，只收集新文件
                        if (self.collection_start_time and 
                            f.stat().st_mtime > self.collection_start_time and
                            f not in topo_files):
                            topo_files.append(f)
                            print(f"    找到新拓扑文件: {f}")
                
                # 查找探索统计数据
                stats_patterns = ["exploration_stats_*.txt", "exploration_*.txt"]
                for pattern in stats_patterns:
                    found_files = list(search_path.glob(pattern))
                    for f in found_files:
                        if (self.collection_start_time and 
                            f.stat().st_mtime > self.collection_start_time and
                            f not in stats_files):
                            stats_files.append(f)
                            print(f"    找到新统计文件: {f}")
                
                # 如果在高优先级目录找到文件，就不用搜索其他目录了
                if topo_files or stats_files:
                    print(f"    在 {search_path} 找到数据文件，停止搜索其他目录")
                    break
        
        print(f"  总计找到: {len(topo_files)} 个拓扑文件, {len(stats_files)} 个统计文件")
        return topo_files, stats_files
    
    def move_and_filter_data(self, output_dir):
        """移动并过滤数据"""
        print("正在整理和过滤数据...")
        
        # 查找数据文件
        topo_files, stats_files = self.find_collected_data(output_dir)
        
        # 移动原始数据到输出目录
        raw_dir = output_dir / "raw_data"
        raw_dir.mkdir(exist_ok=True)
        
        moved_topo_files = []
        for file in topo_files:
            dest = raw_dir / file.name
            shutil.move(str(file), str(dest))
            moved_topo_files.append(dest)
            print(f"  移动: {file.name} -> {dest}")
        
        moved_stats_files = []
        for file in stats_files:
            dest = raw_dir / file.name
            shutil.move(str(file), str(dest))
            moved_stats_files.append(dest)
            print(f"  移动: {file.name} -> {dest}")
        
        # 过滤拓扑图数据
        if moved_topo_files:
            filtered_dir = output_dir / "filtered_data"
            filtered_dir.mkdir(exist_ok=True)
            
            # 使用filter_topo_v2.py进行过滤
            filter_script = self.epic_dir / "filter_topo_v2.py"
            if filter_script.exists():
                try:
                    # 运行过滤脚本
                    subprocess.run([
                        'python3', str(filter_script),
                        str(raw_dir),
                        '-o', str(filtered_dir),
                        '-t', '0.5'  # 时间间隔阈值
                    ], check=True)
                    print(f"✓ 数据过滤完成: {filtered_dir}")
                except subprocess.CalledProcessError as e:
                    print(f"警告: 数据过滤失败: {e}")
            else:
                print(f"警告: 过滤脚本不存在: {filter_script}")
        
        print(f"✓ 数据整理完成: {output_dir}")
        return len(moved_topo_files), len(moved_stats_files)
    
    def cleanup_temp_files(self, temp_launch, config=None):
        """清理临时文件"""
        try:
            if temp_launch.exists():
                temp_launch.unlink()
                print(f"✓ 清理临时文件: {temp_launch}")
        except Exception as e:
            print(f"警告: 清理临时文件失败: {e}")
        
        # 清理临时algorithm.xml文件
        if config and 'temp_algorithm' in config:
            try:
                temp_algorithm = config['temp_algorithm']
                if temp_algorithm.exists():
                    temp_algorithm.unlink()
                    print(f"✓ 清理临时algorithm文件: {temp_algorithm}")
            except Exception as e:
                print(f"警告: 清理临时algorithm文件失败: {e}")
        
        # 清理临时MARSIM launch文件
        if config and 'temp_marsim_launch' in config:
            try:
                temp_marsim_launch = config['temp_marsim_launch']
                if temp_marsim_launch.exists():
                    temp_marsim_launch.unlink()
                    print(f"✓ 清理临时MARSIM文件: {temp_marsim_launch}")
            except Exception as e:
                print(f"警告: 清理临时MARSIM文件失败: {e}")

    def run_single_collection(self, config=None):
        """运行单次数据采集"""
        try:
            # 选择配置
            if config is None:
                config = self.select_random_config()
            
            print(f"=== 开始数据采集 ===")
            print(f"配置: {config['map_type']} - {config['batch_id']}/{config['point_id']}")
            
            # 设置输出目录
            output_dir = self.setup_output_directory(config)
            
            # 将输出目录添加到配置中
            config['output_dir'] = output_dir
            
            # 修改launch文件
            temp_launch = self.modify_launch_file(config)
            
            # 设置ROS参数以启用数据采集
            self.setup_ros_parameters(output_dir)
            
            # 记录采集开始时间（用于过滤新文件）
            self.collection_start_time = time.time()
            print(f"记录采集开始时间: {self.collection_start_time}")
            
            try:
                # 启动ROS (不再重复设置参数)
                process = self.start_ros_launch(temp_launch, output_dir)
                
                # 等待数据采集完成 (不干扰正在运行的探索)
                self.wait_for_completion(process, self.collection_time)
                
                # 终止ROS进程
                self.cleanup_ros_processes()
                
                # 移动和过滤数据
                topo_count, stats_count = self.move_and_filter_data(output_dir)
                
                print(f"✓ 采集完成:")
                print(f"  输出目录: {output_dir}")
                print(f"  拓扑文件: {topo_count} 个")
                print(f"  统计文件: {stats_count} 个")
                
                return output_dir
                
            finally:
                # 清理临时文件
                self.cleanup_temp_files(temp_launch, config)
                
        except KeyboardInterrupt:
            print("\n用户中断，正在清理...")
            self.cleanup_ros_processes()
            sys.exit(1)
        except Exception as e:
            print(f"采集过程出错: {e}")
            self.cleanup_ros_processes()
            raise

    def run_batch_collection(self, count=10):
        """运行批量数据采集"""
        print(f"=== 开始批量数据采集 (共{count}次) ===")
        
        successful = 0
        failed = 0
        
        for i in range(count):
            print(f"\n--- 第 {i+1}/{count} 次采集 ---")
            
            try:
                output_dir = self.run_single_collection()
                successful += 1
                print(f"✓ 第 {i+1} 次采集成功")
                
                # 等待一段时间再进行下次采集
                if i < count - 1:
                    wait_time = 10
                    print(f"等待 {wait_time} 秒后进行下次采集...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                print(f"✗ 第 {i+1} 次采集失败: {e}")
                failed += 1
                
                # 确保进程被清理
                self.cleanup_ros_processes()
                
                # 短暂等待后继续
                time.sleep(5)
        
        print(f"\n=== 批量采集完成 ===")
        print(f"成功: {successful} 次")
        print(f"失败: {failed} 次")
        print(f"输出目录: {self.output_base_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='EPIC数据自动采集脚本')
    parser.add_argument('-c', '--count', type=int, default=1, help='采集次数 (默认: 1)')
    parser.add_argument('-t', '--time', type=int, default=300, help='单次采集时长(秒) (默认: 300)')
    parser.add_argument('-d', '--dir', default='/home/amax/EPIC', help='EPIC项目目录 (默认: /home/amax/EPIC)')
    parser.add_argument('--map-type', choices=['forest', 'partition', 'dungeon'], help='指定地图类型')
    parser.add_argument('--list-configs', action='store_true', help='列出所有可用配置')
    
    args = parser.parse_args()
    
    # 创建采集脚本实例
    collector = DataCollectionScript(args.dir, args.time)
    
    if args.list_configs:
        configs = collector.get_available_configs()
        print(f"找到 {len(configs)} 个可用配置:")
        for config in configs:
            print(f"  {config['map_type']} - {config['batch_id']}/{config['point_id']}")
        return
    
    try:
        if args.count == 1:
            # 单次采集
            config = None
            if args.map_type:
                # 过滤指定地图类型的配置
                configs = [c for c in collector.get_available_configs() if c['map_type'] == args.map_type]
                if configs:
                    config = random.choice(configs)
                    positions = collector.read_initial_positions(config['txt_file'])
                    if positions:
                        init_x, init_y = random.choice(positions)
                        config.update({'init_x': init_x, 'init_y': init_y, 'init_z': 1.0, 'init_yaw': 0.0})
            
            collector.run_single_collection(config)
        else:
            # 批量采集
            collector.run_batch_collection(args.count)
            
    except KeyboardInterrupt:
        print("\n用户中断程序")
        collector.cleanup_ros_processes()
    except Exception as e:
        print(f"程序出错: {e}")
        collector.cleanup_ros_processes()
        sys.exit(1)


if __name__ == "__main__":
    main()
