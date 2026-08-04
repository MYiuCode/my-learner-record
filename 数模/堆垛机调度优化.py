NEW_FILE_CODE
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

class StackCraneScheduler:
    """堆垛机调度优化器 - 改进版
    
    改进点：
    1. 实现完整的FIFO、SSTF、SCAN三种策略
    2. 添加死锁检测与安全阀机制
    3. 增加可视化对比功能
    4. 支持多巷道并行仿真
    """
    
    def __init__(self, safety_buffer_ratio: float = 0.05):
        """
        Args:
            safety_buffer_ratio: 安全缓冲区比例（默认5%）
        """
        self.safety_buffer_ratio = safety_buffer_ratio
        self.results = {}
        
    def load_data(self, alloc_file: str, order_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载库位分配和订单数据"""
        try:
            df_alloc = pd.read_csv(alloc_file)
        except FileNotFoundError:
            print(f"警告: 未找到 {alloc_file}，使用模拟数据")
            df_alloc = self._generate_mock_allocation()
        
        df_orders = pd.read_csv(order_file)
        
        # 关联库位信息
        df_orders_mapped = df_orders.merge(
            df_alloc, 
            left_on='原材料号', 
            right_on='SKU' if 'SKU' in df_alloc.columns else '原材料编号',
            how='inner'
        )
        
        return df_alloc, df_orders_mapped
    
    def _generate_mock_allocation(self) -> pd.DataFrame:
        """生成模拟库位分配数据"""
        np.random.seed(42)
        n_samples = 500
        
        return pd.DataFrame({
            'SKU': [f'B6T{i:011d}' for i in range(n_samples)],
            'Aisle': np.random.randint(1, 8, n_samples),
            'Shelf': np.random.randint(1, 15, n_samples),
            'Level': np.random.randint(1, 51, n_samples),
            'Col': np.random.randint(1, 69, n_samples),
            'Depth': np.random.choice([1, 2], n_samples, p=[0.6, 0.4])
        })
    
    def calculate_travel_time(self, pos1: Tuple[float, float], 
                             pos2: Tuple[float, float]) -> float:
        """计算两点间行进时间（线性模型）"""
        col_diff = abs(pos2[0] - pos1[0])
        level_diff = abs(pos2[1] - pos1[1])
        
        t_horizontal = (col_diff * 0.8) / 3.0
        t_vertical = (level_diff * 0.4) / 0.75
        
        return max(t_horizontal, t_vertical)
    
    def simulate_fifo(self, orders: pd.DataFrame) -> Dict:
        """FIFO先进先出策略"""
        if orders.empty:
            return {'total_time': 0, 'sequence': [], 'empty_run_ratio': 0}
        
        t_now = 0.0
        current_pos = (1.0, 1.0)  # I/O起点
        total_time = 0.0
        empty_distance = 0.0
        total_distance = 0.0
        
        sequence = []
        
        for idx, order in orders.iterrows():
            target_pos = (order['Col'], order['Level'])
            
            # 移动时间
            t_move = self.calculate_travel_time(current_pos, target_pos)
            
            # 取货时间 + 倒腾惩罚
            t_pick = 10.0 + (15.0 if order['Depth'] == 2 else 0.0)
            
            step_time = t_move + t_pick
            total_time += step_time
            
            # 距离统计
            dist = np.sqrt(
                (target_pos[0] - current_pos[0])**2 + 
                (target_pos[1] - current_pos[1])**2
            )
            total_distance += dist
            
            sequence.append({
                'order_idx': idx,
                'sku': order.get('原材料号', order.get('SKU', '')),
                'start_pos': current_pos,
                'end_pos': target_pos,
                'move_time': t_move,
                'pick_time': t_pick,
                'cumulative_time': total_time
            })
            
            current_pos = target_pos
        
        # 计算空驶率（简化估算）
        avg_distance = total_distance / len(orders) if len(orders) > 0 else 0
        optimal_distance = self._estimate_optimal_distance(orders)
        empty_run_ratio = max(0, 1 - optimal_distance / total_distance) if total_distance > 0 else 0
        
        return {
            'total_time': total_time,
            'avg_time_per_order': total_time / len(orders) if len(orders) > 0 else 0,
            'sequence': sequence,
            'empty_run_ratio': empty_run_ratio,
            'strategy': 'FIFO'
        }
    
    def simulate_sstf(self, orders: pd.DataFrame) -> Dict:
        """SSTF最短寻找时间优先策略"""
        if orders.empty:
            return {'total_time': 0, 'sequence': [], 'empty_run_ratio': 0}
        
        remaining_orders = orders.copy()
        current_pos = (1.0, 1.0)
        total_time = 0.0
        sequence = []
        
        while not remaining_orders.empty:
            # 计算到所有待处理订单的距离
            distances = remaining_orders.apply(
                lambda row: self.calculate_travel_time(
                    current_pos, 
                    (row['Col'], row['Level'])
                ),
                axis=1
            )
            
            # 选择最近的订单
            nearest_idx = distances.idxmin()
            order = remaining_orders.loc[nearest_idx]
            
            target_pos = (order['Col'], order['Level'])
            t_move = distances[nearest_idx]
            t_pick = 10.0 + (15.0 if order['Depth'] == 2 else 0.0)
            
            step_time = t_move + t_pick
            total_time += step_time
            
            sequence.append({
                'order_idx': nearest_idx,
                'sku': order.get('原材料号', order.get('SKU', '')),
                'start_pos': current_pos,
                'end_pos': target_pos,
                'move_time': t_move,
                'pick_time': t_pick,
                'cumulative_time': total_time
            })
            
            current_pos = target_pos
            remaining_orders = remaining_orders.drop(nearest_idx)
        
        avg_time = total_time / len(orders) if len(orders) > 0 else 0
        
        return {
            'total_time': total_time,
            'avg_time_per_order': avg_time,
            'sequence': sequence,
            'empty_run_ratio': self._estimate_empty_ratio(sequence),
            'strategy': 'SSTF'
        }
    
    def simulate_scan(self, orders: pd.DataFrame, direction: str = 'forward') -> Dict:
        """SCAN电梯扫描策略
        
        Args:
            orders: 订单数据
            direction: 扫描方向 ('forward' 或 'backward')
        """
        if orders.empty:
            return {'total_time': 0, 'sequence': [], 'empty_run_ratio': 0}
        
        # 按列排序（模拟电梯单向扫描）
        if direction == 'forward':
            orders_sorted = orders.sort_values(by='Col', ascending=True)
        else:
            orders_sorted = orders.sort_values(by='Col', ascending=False)
        
        current_pos = (1.0, 1.0)
        total_time = 0.0
        sequence = []
        
        for idx, order in orders_sorted.iterrows():
            target_pos = (order['Col'], order['Level'])
            
            t_move = self.calculate_travel_time(current_pos, target_pos)
            t_pick = 10.0 + (15.0 if order['Depth'] == 2 else 0.0)
            
            step_time = t_move + t_pick
            total_time += step_time
            
            sequence.append({
                'order_idx': idx,
                'sku': order.get('原材料号', order.get('SKU', '')),
                'start_pos': current_pos,
                'end_pos': target_pos,
                'move_time': t_move,
                'pick_time': t_pick,
                'cumulative_time': total_time
            })
            
            current_pos = target_pos
        
        avg_time = total_time / len(orders) if len(orders) > 0 else 0
        
        return {
            'total_time': total_time,
            'avg_time_per_order': avg_time,
            'sequence': sequence,
            'empty_run_ratio': self._estimate_empty_ratio(sequence),
            'strategy': 'SCAN'
        }
    
    def check_deadlock_risk(self, orders: pd.DataFrame, 
                           aisle_capacity: int) -> Dict:
        """死锁风险评估
        
        Args:
            orders: 订单数据
            aisle_capacity: 巷道总容量
            
        Returns:
            死锁风险评估报告
        """
        total_orders = len(orders)
        deep_orders = orders[orders['Depth'] == 2]
        
        # 估算当前装载率
        estimated_load = total_orders / aisle_capacity if aisle_capacity > 0 else 0
        
        # 安全检查：保留安全缓冲区
        safe_threshold = 1.0 - self.safety_buffer_ratio
        
        risk_level = "LOW"
        if estimated_load > 0.95:
            risk_level = "CRITICAL"
        elif estimated_load > 0.90:
            risk_level = "HIGH"
        elif estimated_load > 0.85:
            risk_level = "MEDIUM"
        
        return {
            'total_orders': total_orders,
            'deep_position_orders': len(deep_orders),
            'estimated_load_ratio': round(estimated_load, 4),
            'safe_threshold': safe_threshold,
            'risk_level': risk_level,
            'recommendation': self._get_safety_recommendation(risk_level)
        }
    
    def _estimate_optimal_distance(self, orders: pd.DataFrame) -> float:
        """估算最优路径距离（下界）"""
        if orders.empty:
            return 0.0
        
        # 简化的TSP下界估算
        positions = orders[['Col', 'Level']].values
        if len(positions) < 2:
            return 0.0
        
        # 最小生成树近似
        from scipy.spatial.distance import pdist
        distances = pdist(positions, metric='euclidean')
        mst_approx = np.sum(np.sort(distances)[:len(positions)-1])
        
        return mst_approx
    
    def _estimate_empty_ratio(self, sequence: List[Dict]) -> float:
        """估算空驶率"""
        if not sequence:
            return 0.0
        
        total_moves = len(sequence)
        # 简化：假设长距离移动中约40%为空驶
        long_moves = sum(
            1 for s in sequence 
            if np.sqrt((s['end_pos'][0]-s['start_pos'][0])**2 + 
                      (s['end_pos'][1]-s['start_pos'][1])**2) > 10
        )
        
        return long_moves / total_moves if total_moves > 0 else 0
    
    def _get_safety_recommendation(self, risk_level: str) -> str:
        """获取安全建议"""
        recommendations = {
            "LOW": "装载率安全，无需特殊处理",
            "MEDIUM": "建议监控深位货格使用情况",
            "HIGH": "建议启用邻近释放协议，优先使用同列空位",
            "CRITICAL": "立即停止入库操作，执行紧急清仓"
        }
        return recommendations.get(risk_level, "未知风险等级")
    
    def compare_strategies(self, orders: pd.DataFrame) -> pd.DataFrame:
        """对比三种调度策略"""
        print("正在执行FIFO策略...")
        fifo_result = self.simulate_fifo(orders)
        
        print("正在执行SSTF策略...")
        sstf_result = self.simulate_sstf(orders)
        
        print("正在执行SCAN策略...")
        scan_result = self.simulate_scan(orders)
        
        comparison = pd.DataFrame([
            {
                '策略': 'FIFO',
                '总耗时(秒)': round(fifo_result['total_time'], 2),
                '平均单单耗时(秒)': round(fifo_result['avg_time_per_order'], 2),
                '空驶率(%)': round(fifo_result['empty_run_ratio'] * 100, 2)
            },
            {
                '策略': 'SSTF',
                '总耗时(秒)': round(sstf_result['total_time'], 2),
                '平均单单耗时(秒)': round(sstf_result['avg_time_per_order'], 2),
                '空驶率(%)': round(sstf_result['empty_run_ratio'] * 100, 2)
            },
            {
                '策略': 'SCAN',
                '总耗时(秒)': round(scan_result['total_time'], 2),
                '平均单单耗时(秒)': round(scan_result['avg_time_per_order'], 2),
                '空驶率(%)': round(scan_result['empty_run_ratio'] * 100, 2)
            }
        ])
        
        # 计算提效比
        fifo_time = fifo_result['total_time']
        comparison['较FIFO提效比(%)'] = round(
            (1 - comparison['总耗时(秒)'] / fifo_time) * 100, 2
        )
        
        return comparison
    
    def visualize_comparison(self, comparison: pd.DataFrame, 
                            save_path: str = '问题二_策略对比.png'):
        """可视化策略对比结果"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        strategies = comparison['策略'].values
        times = comparison['总耗时(秒)'].values
        avg_times = comparison['平均单单耗时(秒)'].values
        empty_ratios = comparison['空驶率(%)'].values
        
        # 总耗时对比
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars1 = axes[0].bar(strategies, times, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('总耗时 (秒)', fontsize=12)
        axes[0].set_title('总耗时对比', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # 在柱状图上标注数值
        for bar, time in zip(bars1, times):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500,
                        f'{time:.0f}', ha='center', va='bottom', fontsize=10)
        
        # 平均单耗时对比
        bars2 = axes[1].bar(strategies, avg_times, color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('平均单单耗时 (秒)', fontsize=12)
        axes[1].set_title('平均单耗时对比', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        for bar, time in zip(bars2, avg_times):
            axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{time:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 空驶率对比
        bars3 = axes[2].bar(strategies, empty_ratios, color=colors, alpha=0.7, edgecolor='black')
        axes[2].set_ylabel('空驶率 (%)', fontsize=12)
        axes[2].set_title('空驶率对比', fontsize=14, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        
        for bar, ratio in zip(bars3, empty_ratios):
            axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{ratio:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 对比图已保存至: {save_path}")
        plt.show()


def main():
    """主函数"""
    scheduler = StackCraneScheduler(safety_buffer_ratio=0.05)
    
    # Step 1: 加载数据
    print("=" * 60)
    print("Step 1: 加载数据")
    print("=" * 60)
    df_alloc, df_orders = scheduler.load_data(
        "问题一_库位分配结果.csv",
        "表2_生产线订料数据.csv"
    )
    
    # Step 2: 按巷道分组测试
    print("\n" + "=" * 60)
    print("Step 2: 多巷道调度仿真")
    print("=" * 60)
    
    all_comparisons = []
    
    for aisle in range(1, 8):
        print(f"\n--- 巷道 {aisle} ---")
        df_aisle = df_orders[df_orders['Aisle'] == aisle]
        
        if df_aisle.empty:
            print(f"巷道 {aisle} 无订单，跳过")
            continue
        
        print(f"订单数量: {len(df_aisle)}")
        
        # 死锁风险评估
        risk_report = scheduler.check_deadlock_risk(df_aisle, aisle_capacity=13600)
        print(f"装载率: {risk_report['estimated_load_ratio']*100:.2f}%")
        print(f"风险等级: {risk_report['risk_level']}")
        print(f"建议: {risk_report['recommendation']}")
        
        # 策略对比（使用前100个订单进行快速测试）
        sample_orders = df_aisle.head(min(100, len(df_aisle)))
        comparison = scheduler.compare_strategies(sample_orders)
        
        print("\n策略对比结果:")
        print(comparison.to_string(index=False))
        
        comparison['巷道'] = aisle
        all_comparisons.append(comparison)
    
    # Step 3: 汇总结果
    if all_comparisons:
        final_comparison = pd.concat(all_comparisons, ignore_index=True)
        final_comparison.to_csv("问题二_调度策略对比结果.csv", 
                               index=False, encoding='utf-8-sig')
        print("\n✓ 调度对比结果已保存至: 问题二_调度策略对比结果.csv")
        
        # 可视化
        avg_comparison = final_comparison.groupby('策略')[
            ['总耗时(秒)', '平均单单耗时(秒)', '空驶率(%)']
        ].mean().reset_index()
        
        scheduler.visualize_comparison(avg_comparison)


if __name__ == "__main__":
    main()
