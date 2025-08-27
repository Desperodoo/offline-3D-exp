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
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

class DataCollectionScript:
    def __init__(self, epic_dir="/home/amax/EPIC", collection_time=300):
        self.epic_dir = Path(epic_dir)
        self.resource_dir = self.epic_dir / "src/MARSIM/map_generator/resource"
        self.launch_dir = self.epic_dir / "src/MARSIM/mars_drone_sim/launch"
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
        original_launch = self.launch_dir / f"{map_type}.launch"
        temp_launch = self.launch_dir / f"{map_type}_temp.launch"
        
        if not original_launch.exists():
            raise FileNotFoundError(f"Launch文件不存在: {original_launch}")
        
        try:
            # 方法1: 使用XML解析和修改
            tree = ET.parse(original_launch)
            root = tree.getroot()
            
            # 查找并修改初始位置参数
            include_elem = root.find(".//include")
            if include_elem is not None:
                # 查找或创建初始位置参数
                for param_name, value in [
                    ('init_x_', config['init_x']),
                    ('init_y_', config['init_y']),
                    ('init_z_', config['init_z']),
                    ('init_yaw', config['init_yaw'])
                ]:
                    # 查找现有的arg元素
                    existing_arg = include_elem.find(f"arg[@name='{param_name}']")
                    if existing_arg is not None:
                        existing_arg.set('value', str(value))
                    else:
                        # 创建新的arg元素
                        new_arg = ET.SubElement(include_elem, 'arg')
                        new_arg.set('name', param_name)
                        new_arg.set('value', str(value))
            
            # 保存到临时文件
            tree.write(temp_launch, encoding='utf-8', xml_declaration=True)
            
        except ET.ParseError:
            # 方法2: 如果XML解析失败，使用字符串替换
            print("XML解析失败，使用字符串替换方法...")
            self._modify_launch_file_text(original_launch, temp_launch, config)
        
        print(f"✓ 修改launch文件: {temp_launch}")
        print(f"  地图: {config['map_type']} ({config['batch_id']}/{config['point_id']})")
        print(f"  初始位置: ({config['init_x']:.2f}, {config['init_y']:.2f}, {config['init_z']:.2f})")
        
        return temp_launch
    
    def _modify_launch_file_text(self, original_launch, temp_launch, config):
        """使用文本替换方式修改launch文件"""
        with open(original_launch, 'r') as f:
            content = f.read()
        
        # 替换初始位置参数
        replacements = [
            (r'<arg name="init_x_" value="[^"]*"', f'<arg name="init_x_" value="{config["init_x"]}"'),
            (r'<arg name="init_y_" value="[^"]*"', f'<arg name="init_y_" value="{config["init_y"]}"'),
            (r'<arg name="init_z_" value="[^"]*"', f'<arg name="init_z_" value="{config["init_z"]}"'),
            (r'<arg name="init_yaw" value="[^"]*"', f'<arg name="init_yaw" value="{config["init_yaw"]}"'),
        ]
        
        import re
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        with open(temp_launch, 'w') as f:
            f.write(content)
    
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
    
    def start_ros_launch(self, launch_file, output_dir):
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
                time.sleep(3)  # 等待roscore启动
            
            # 设置ROS参数（在启动launch前）
            self.setup_ros_parameters(output_dir)
            
            # 启动launch文件
            launch_cmd = [
                'roslaunch', str(launch_file),
                'sensing_horizon:=15.0'  # 默认感知范围
            ]
            
            print(f"启动launch文件: {' '.join(launch_cmd)}")
            
            launch_proc = subprocess.Popen(launch_cmd,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
            self.ros_processes.append(launch_proc)
            
            # 等待launch文件启动并检查状态
            print("等待系统启动...")
            for i in range(15):  # 15秒内检查进程状态
                time.sleep(1)
                if launch_proc.poll() is not None:
                    # 进程已经终止，获取错误信息
                    stdout, stderr = launch_proc.communicate()
                    print(f"Launch进程提前终止 (返回码: {launch_proc.returncode})")
                    if stderr:
                        print(f"错误输出: {stderr.decode('utf-8', errors='ignore')}")
                    if stdout:
                        print(f"标准输出: {stdout.decode('utf-8', errors='ignore')}")
                    raise RuntimeError(f"Launch文件启动失败")
                
                print(f"  启动中... ({i+1}/15)")
            
            print("✓ Launch进程运行中")
            
            # 检查关键ROS参数
            self.check_ros_parameters()
            
            return launch_proc
            
        except Exception as e:
            print(f"错误启动ROS: {e}")
            raise
    
    def setup_ros_parameters(self, output_dir):
        """设置ROS参数以启用数据采集功能"""
        print("设置ROS参数...")
        
        # 创建raw_data目录
        raw_data_dir = output_dir / "raw_data"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置拓扑图提取参数
        params = {
            'topo_extraction/enable_extraction': 'true',
            'topo_extraction/extraction_rate': '2.0',
            'topo_extraction/export_dir': str(raw_data_dir),
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
    
    def check_ros_parameters(self):
        """检查ROS参数设置"""
        print("检查ROS参数设置...")
        
        params_to_check = [
            "topo_extraction/enable_extraction",
            "topo_extraction/extraction_rate", 
            "topo_extraction/export_dir",
            "exploration_stats/enable_stats"
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
    
    def find_collected_data(self, output_dir):
        """查找采集到的数据文件（只查找本次采集生成的新文件）"""
        print("正在查找数据文件...")
        
        # 主要搜索路径
        search_paths = [
            output_dir / "raw_data",  # 首选位置
            Path("/tmp"),             # 备用位置
            self.epic_dir,            # EPIC目录
            Path.home()               # 用户主目录
        ]
        
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
            if file.parent != raw_dir:  # 只有当文件不在目标目录时才移动
                shutil.move(str(file), str(dest))
                print(f"  移动: {file.name} -> {dest}")
            moved_topo_files.append(dest)
        
        moved_stats_files = []
        for file in stats_files:
            dest = raw_dir / file.name
            if file.parent != raw_dir:  # 只有当文件不在目标目录时才移动
                shutil.move(str(file), str(dest))
                print(f"  移动: {file.name} -> {dest}")
            moved_stats_files.append(dest)
        
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
    
    def cleanup_temp_files(self, temp_launch):
        """清理临时文件"""
        try:
            if temp_launch.exists():
                temp_launch.unlink()
                print(f"✓ 清理临时文件: {temp_launch}")
        except Exception as e:
            print(f"警告: 清理临时文件失败: {e}")
    
    def run_single_collection(self, config=None, sensing_horizon=15.0):
        """运行单次数据采集"""
        try:
            # 选择配置
            if config is None:
                config = self.select_random_config()
            
            print(f"=== 开始数据采集 ===")
            print(f"配置: {config['map_type']} - {config['batch_id']}/{config['point_id']}")
            
            # 设置输出目录
            output_dir = self.setup_output_directory(config)
            
            # 修改launch文件
            temp_launch = self.modify_launch_file(config)
            
            # 记录采集开始时间（用于过滤新文件）
            self.collection_start_time = time.time()
            print(f"记录采集开始时间: {self.collection_start_time}")
            
            try:
                # 启动ROS
                process = self.start_ros_launch(temp_launch, output_dir)
                
                # 等待数据采集完成
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
                self.cleanup_temp_files(temp_launch)
                
        except KeyboardInterrupt:
            print("\n用户中断，正在清理...")
            self.cleanup_ros_processes()
            sys.exit(1)
        except Exception as e:
            print(f"采集过程出错: {e}")
            self.cleanup_ros_processes()
            raise
    
    def run_batch_collection(self, count=10, sensing_horizon=15.0):
        """运行批量数据采集"""
        print(f"=== 开始批量数据采集 (共{count}次) ===")
        
        successful = 0
        failed = 0
        
        for i in range(count):
            print(f"\n--- 第 {i+1}/{count} 次采集 ---")
            
            try:
                output_dir = self.run_single_collection(sensing_horizon=sensing_horizon)
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
    parser.add_argument('-s', '--sensing-horizon', type=float, default=15.0, help='感知范围 (默认: 15.0)')
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
            
            collector.run_single_collection(config, args.sensing_horizon)
        else:
            # 批量采集
            collector.run_batch_collection(args.count, args.sensing_horizon)
            
    except KeyboardInterrupt:
        print("\n用户中断程序")
        collector.cleanup_ros_processes()
    except Exception as e:
        print(f"程序出错: {e}")
        collector.cleanup_ros_processes()
        sys.exit(1)


if __name__ == "__main__":
    main()
