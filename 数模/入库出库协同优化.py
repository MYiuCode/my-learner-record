NEW_FILE_CODE
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

class DCCOptimizer:
    """复合作业(DCC)协同优化器 - 改进版
    
    改进点：
    1. 完善二部图匹配模型
    2. 添加时间窗约束检查
    3. 增加货位适配性验证
    4. 提供可视化匹配结果
    """
    
    def __init__(self):
        self.matching_results = []
        
    def calculate_nonlinear_time(self, pos1: Tuple[float, float],
                                 pos2: Tuple[float, float],
                                 is_loaded: bool = False) -> float:
        """非线性运动学时间计算（考虑加减速）
        
        Args:
            pos1: 起始位置 (col, level)
            pos2: 目标位置 (col, level)
            is_loaded: 是否满载
            
        Returns:
            运动时间（秒）
        """
        # 动力学参数
        if not is_loaded:
            ax, vx = 0.5, 3.0  # 空载
            ay, vy = 0.15, 0.75
        else:
            ax, vx = 0.4, 2.3  # 满载
            ay, vy = 0.10, 0.58
        
        # 实际位移（米）
        s_x = abs(pos2[0] - pos1[0]) * 0.8  # 列宽0.8米
        s_y = abs(pos2[1] - pos1[1]) * 0.4  # 层高0.4米
        
        def get_axis_time(S, a_max, v_max):
            if S <= 0.001:
                return 0.0
            S_crit = (v_max ** 2) / a_max
            if S <= S_crit:
                # 三角形速度波形
                return 2.0 * np.sqrt(S / a_max)
            else:
                # 梯形速度波形
                return (S / v_max) + (v_max / a_max)
        
        t_x = get_axis_time(s_x, ax, vx)
        t_y = get_axis_time(s_y, ay, vy)
        
        return max(t_x, t_y)
    
    def check_time_window_constraint(self, inbound_time: float,
                                     outbound_time: float,
                                     travel_time: float,
                                     delta_t: float = 300.0) -> bool:
        """检查时间窗约束
        
        Args:
            inbound_time: 入库任务到达时间
            outbound_time: 出库任务计划时间
            travel_time: 从入库点到出库点的转移时间
            delta_t: 出库延迟容忍度（默认300秒）
            
        Returns:
            是否满足时间窗约束
        """
        crane_ready_time = inbound_time + travel_time
        return crane_ready_time <= outbound_time + delta_t
    
    def check_slot_compatibility(self, box_type: str, level: int) -> bool:
        """检查货位层高兼容性
        
        Args:
            box_type: 箱子类型 (E1/E3/E4)
            level: 层号
            
        Returns:
            是否兼容
        """
        if level <= 8:
            return box_type == 'E1'
        elif level <= 42:
            return box_type in ['E1', 'E3']
        else:
            return box_type in ['E1', 'E3', 'E4']
    
    def build_saving_matrix(self, inbound_list: List[Dict],
                           outbound_list: List[Dict],
                           use_nonlinear: bool = True) -> np.ndarray:
        """构建节约矩阵
        
        Args:
            inbound_list: 入库任务列表
            outbound_list: 出库任务列表
            use_nonlinear: 是否使用非线性时间模型
            
        Returns:
            节约矩阵
        """
        n_in = len(inbound_list)
        n_out = len(outbound_list)
        
        saving_matrix = np.zeros((n_in, n_out))
        
        io_pos = (1, 1)  # I/O起点
        
        for i in range(n_in):
            for j in range(n_out):
                # 获取位置信息
                loc_in = (inbound_list[i]['col'], inbound_list[i]['level'])
                loc_out = (outbound_list[j]['Col'], outbound_list[j]['Level'])
                
                # 检查层高兼容性
                if not self.check_slot_compatibility(
                    inbound_list[i].get('box_type', 'E1'),
                    inbound_list[i]['level']
                ):
                    saving_matrix[i, j] = -np.inf
                    continue
                
                # 检查时间窗约束
                if 'inbound_time' in inbound_list[i] and 'outbound_time' in outbound_list[j]:
                    if use_nonlinear:
                        t_io_to_in = self.calculate_nonlinear_time(
                            io_pos, loc_in, is_loaded=True
                        )
                    else:
                        t_io_to_in = max(
                            loc_in[0] * 0.8 / 3.0,
                            loc_in[1] * 0.4 / 0.75
                        )
                    
                    if not self.check_time_window_constraint(
                        inbound_list[i]['inbound_time'],
                        outbound_list[j]['outbound_time'],
                        t_io_to_in
                    ):
                        saving_matrix[i, j] = -np.inf
                        continue
                
                # 计算时间节约
                if use_nonlinear:
                    t_out_only = self.calculate_nonlinear_time(
                        io_pos, loc_out, is_loaded=False
                    )
                    t_inter = self.calculate_nonlinear_time(
                        loc_in, loc_out, is_loaded=False
                    )
                else:
                    t_out_only = max(
                        loc_out[0] * 0.8 / 3.0,
                        loc_out[1] * 0.4 / 0.75
                    )
                    t_inter = max(
                        abs(loc_in[0] - loc_out[0]) * 0.8 / 3.0,
                        abs(loc_in[1] - loc_out[1]) * 0.4 / 0.75
                    )
                
                saving = t_out_only - t_inter
                saving_matrix[i, j] = max(0.0, saving)
        
        return saving_matrix
    
    def optimize_matching(self, inbound_list: List[Dict],
                         outbound_list: List[Dict],
                         use_nonlinear: bool = True) -> List[Dict]:
        """执行最优匹配
        
        Args:
            inbound_list: 入库任务列表
            outbound_list: 出库任务列表
            use_nonlinear: 是否使用非线性模型
            
        Returns:
            匹配结果列表
        """
        print(f"入库任务数: {len(inbound_list)}")
        print(f"出库任务数: {len(outbound_list)}")
        
        # 构建节约矩阵
        saving_matrix = self.build_saving_matrix(
            inbound_list, outbound_list, use_nonlinear
        )
        
        # 处理无穷大值
        saving_matrix_clean = np.where(
            np.isinf(saving_matrix), 
            -1e10, 
            saving_matrix
        )
        
        # 匈牙利算法求解最大权匹配
        row_ind, col_ind = linear_sum_assignment(-saving_matrix_clean)
        
        pairs = []
        total_saving = 0.0
        
        for r, c in zip(row_ind, col_ind):
            if saving_matrix[r, c] > 0.01:  # 只保留有效配对
                pairs.append({
                    'inbound_id': inbound_list[r].get('id', r),
                    'outbound_sku': outbound_list[c].get('SKU', 
                                        outbound_list[c].get('原材料号', c)),
                    'inbound_location': (inbound_list[r]['col'], 
                                       inbound_list[r]['level']),
                    'outbound_location': (outbound_list[c]['Col'], 
                                        outbound_list[c]['Level']),
                    'time_saving': saving_matrix[r, c],
                    'use_nonlinear': use_nonlinear
                })
                total_saving += saving_matrix[r, c]
        
        print(f"成功配对数: {len(pairs)}")
        print(f"总时间节约: {total_saving:.2f} 秒")
        print(f"平均每对节约: {total_saving/len(pairs) if pairs else 0:.2f} 秒")
        
        return pairs
    
    def compare_linear_vs_nonlinear(self, inbound_list: List[Dict],
                                   outbound_list: List[Dict]) -> Dict:
        """对比线性和非线性模型的匹配效果"""
        print("\n" + "="*60)
        print("线性模型匹配")
        print("="*60)
        pairs_linear = self.optimize_matching(
            inbound_list, outbound_list, use_nonlinear=False
        )
        
        print("\n" + "="*60)
        print("非线性模型匹配")
        print("="*60)
        pairs_nonlinear = self.optimize_matching(
            inbound_list, outbound_list, use_nonlinear=True
        )
        
        # 统计对比
        total_saving_linear = sum(p['time_saving'] for p in pairs_linear)
        total_saving_nonlinear = sum(p['time_saving'] for p in pairs_nonlinear)
        
        comparison = {
            '线性模型': {
                '配对数': len(pairs_linear),
                '总节约时间(秒)': round(total_saving_linear, 2),
                '平均节约时间(秒)': round(
                    total_saving_linear / len(pairs_linear) if pairs_linear else 0, 2
                )
            },
            '非线性模型': {
                '配对数': len(pairs_nonlinear),
                '总节约时间(秒)': round(total_saving_nonlinear, 2),
                '平均节约时间(秒)': round(
                    total_saving_nonlinear / len(pairs_nonlinear) if pairs_nonlinear else 0, 2
                )
            }
        }
        
        return comparison
    
    def visualize_matching(self, pairs: List[Dict],
                          save_path: str = '问题三_DCC匹配可视化.png'):
        """可视化DCC匹配结果"""
        if not pairs:
            print("无匹配结果可可视化")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 提取数据
        inbound_cols = [p['inbound_location'][0] for p in pairs]
        inbound_levels = [p['inbound_location'][1] for p in pairs]
        outbound_cols = [p['outbound_location'][0] for p in pairs]
        outbound_levels = [p['outbound_location'][1] for p in pairs]
        savings = [p['time_saving'] for p in pairs]
        
        # 左图：匹配连线图
        ax1 = axes[0]
        for i in range(len(pairs)):
            ax1.plot([inbound_cols[i], outbound_cols[i]], 
                    [inbound_levels[i], outbound_levels[i]],
                    'b-', alpha=0.3, linewidth=0.5)
        
        ax1.scatter(inbound_cols, inbound_levels, c='green', 
                   s=100, marker='s', label='入库位置', alpha=0.7, edgecolors='black')
        ax1.scatter(outbound_cols, outbound_levels, c='red', 
                   s=100, marker='o', label='出库位置', alpha=0.7, edgecolors='black')
        
        ax1.set_xlabel('列号', fontsize=12)
        ax1.set_ylabel('层号', fontsize=12)
        ax1.set_title('DCC复合作业匹配示意图', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 右图：时间节约分布
        ax2 = axes[1]
        ax2.hist(savings, bins=20, color='steelblue', alpha=0.7, 
                edgecolor='black', density=True)
        ax2.axvline(np.mean(savings), color='red', linestyle='--', 
                   linewidth=2, label=f'平均值: {np.mean(savings):.2f}s')
        ax2.set_xlabel('时间节约 (秒)', fontsize=12)
        ax2.set_ylabel('频率', fontsize=12)
        ax2.set_title('DCC时间节约分布', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 可视化结果已保存至: {save_path}")
        plt.show()


def main():
    """主函数"""
    optimizer = DCCOptimizer()
    
    # Step 1: 加载数据
    print("=" * 60)
    print("Step 1: 加载数据")
    print("=" * 60)
    
    try:
        df_alloc = pd.read_csv("问题一_库位分配结果.csv")
        df_orders = pd.read_csv("表2_生产线订料数据.csv")
        df_inbound = pd.read_csv("表3_入库材料数据.csv")
    except FileNotFoundError as e:
        print(f"错误: 缺少数据文件 - {e}")
        print("使用模拟数据进行演示")
        
        # 生成模拟数据
        np.random.seed(42)
        n_inbound = 50
        n_outbound = 50
        
        df_inbound = pd.DataFrame({
            'id': range(n_inbound),
            'col': np.random.randint(1, 69, n_inbound),
            'level': np.random.randint(1, 51, n_inbound),
            'box_type': np.random.choice(['E1', 'E3', 'E4'], n_inbound),
            'inbound_time': np.arange(n_inbound) * 60  # 每分钟一个入库
        })
        
        df_orders = pd.DataFrame({
            '原材料号': [f'SKU_{i}' for i in range(n_outbound)],
            'Col': np.random.randint(1, 69, n_outbound),
            'Level': np.random.randint(1, 51, n_outbound),
            'outbound_time': np.arange(n_outbound) * 90  # 每1.5分钟一个出库
        })
    
    # Step 2: 准备数据
    print("\n" + "=" * 60)
    print("Step 2: 准备入库出库任务列表")
    print("=" * 60)
    
    inbound_list = df_inbound.head(30).to_dict('records')
    outbound_list = df_orders.head(30).to_dict('records')
    
    print(f"选取入库任务: {len(inbound_list)} 个")
    print(f"选取出库任务: {len(outbound_list)} 个")
    
    # Step 3: 执行优化匹配
    print("\n" + "=" * 60)
    print("Step 3: 执行DCC协同优化匹配")
    print("=" * 60)
    
    pairs = optimizer.optimize_matching(inbound_list, outbound_list, use_nonlinear=True)
    
    # Step 4: 对比线性和非线性模型
    print("\n" + "=" * 60)
    print("Step 4: 线性vs非线性模型对比")
    print("=" * 60)
    
    comparison = optimizer.compare_linear_vs_nonlinear(inbound_list, outbound_list)
    
    print("\n对比结果:")
    for model, metrics in comparison.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    # Step 5: 可视化
    print("\n" + "=" * 60)
    print("Step 5: 生成可视化结果")
    print("=" * 60)
    
    optimizer.visualize_matching(pairs)
    
    # Step 6: 保存结果
    if pairs:
        df_pairs = pd.DataFrame(pairs)
        df_pairs.to_csv("问题三_DCC匹配结果.csv", index=False, encoding='utf-8-sig')
        print("\n✓ 匹配结果已保存至: 问题三_DCC匹配结果.csv")


if __name__ == "__main__":
    main()
