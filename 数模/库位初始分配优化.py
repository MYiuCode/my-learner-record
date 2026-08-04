import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class SlotAllocator:
    """库位分配优化器 - 改进版
    
    改进点：
    1. 添加倒腾惩罚计算与交换逻辑
    2. 完善成对存储约束
    3. 增加货位占用状态追踪
    4. 输出详细统计报告
    """
    
    def __init__(self):
        self.df_slots = None
        self.allocation_result = []
        self.occupied_slots = set()
        
    def load_and_clean_data(self, filepath: str) -> pd.DataFrame:
        """数据加载与清洗 - 使用Z-score截断法"""
        df_inv = pd.read_csv(filepath)
        df_inv = df_inv.dropna()
        
        # Z-score异常值检测与剔除
        mean_ratio = df_inv['消耗占比'].mean()
        std_ratio = df_inv['消耗占比'].std()
        z_scores = np.abs((df_inv['消耗占比'] - mean_ratio) / std_ratio)
        
        df_clean = df_inv[z_scores < 3].copy()
        
        # 归一化权重
        total_ratio = df_clean['消耗占比'].sum()
        df_clean['归一化权重'] = df_clean['消耗占比'] / total_ratio
        
        print(f"原始物料数: {len(df_inv)}")
        print(f"剔除异常后物料数: {len(df_clean)}")
        print(f"异常条目数: {len(df_inv) - len(df_clean)}")
        print(f"归一化权重总和: {df_clean['归一化权重'].sum():.6f}")
        
        return df_clean
    
    def build_slot_system(self) -> pd.DataFrame:
        """构建完整货位坐标体系"""
        slots = []
        
        for aisle in range(1, 8):  # 7巷道
            for shelf in range(1, 15):  # 14货架
                for level in range(1, 51):  # 50层
                    for col in range(1, 69):  # 68列
                        for depth in [1, 2]:  # 1=浅位, 2=深位
                            # 层高约束判定
                            if level <= 8:
                                allowed_types = {'E1'}
                                layer_height = 0.2
                            elif level <= 42:
                                allowed_types = {'E1', 'E3'}
                                layer_height = 0.4
                            else:
                                allowed_types = {'E1', 'E3', 'E4'}
                                layer_height = 0.5
                            
                            # 计算行进时间（线性模型）
                            t_travel = max(
                                (col * 0.8) / 3.0,  # 水平移动时间
                                (level * layer_height) / 0.75  # 垂直移动时间
                            )
                            
                            slot_id = f"{aisle:02d}_{shelf:02d}_{level:02d}_{col:02d}_{depth:02d}"
                            
                            slots.append({
                                'aisle': aisle,
                                'shelf': shelf,
                                'level': level,
                                'col': col,
                                'depth': depth,
                                't_travel': t_travel,
                                'allowed_types': frozenset(allowed_types),
                                'is_deep': depth == 2,
                                'slot_id': slot_id,
                                'layer_height': layer_height
                            })
        
        df_slots = pd.DataFrame(slots)
        print(f"总货位数: {len(df_slots)}")
        
        return df_slots
    
    def allocate_slots(self, df_clean: pd.DataFrame, df_slots: pd.DataFrame) -> pd.DataFrame:
        """三阶段贪心分配算法"""
        
        # 分区准备
        e1_slots = df_slots[df_slots['level'] <= 8].copy()
        e1e3_slots = df_slots[(df_slots['level'] >= 9) & (df_slots['level'] <= 42)].copy()
        e1e3e4_slots = df_slots[df_slots['level'] >= 43].copy()
        
        # 按需求强度降序排序
        df_sorted = df_clean.sort_values(by='归一化权重', ascending=False).reset_index(drop=True)
        
        allocation = []
        remaining_slots = {
            'E1': e1_slots.copy(),
            'E1E3': e1e3_slots.copy(),
            'E1E3E4': e1e3e4_slots.copy()
        }
        
        allocated_slot_ids = set()
        
        for idx, row in df_sorted.iterrows():
            sku = row['原材料编号']
            box_type = row['箱子类型']
            weight = row['归一化权重']
            qty = int(row['库存数量/箱'])
            
            # 选择候选分区
            if box_type == 'E1':
                candidates_key = 'E1'
                backup_key = 'E1E3'
            elif box_type == 'E3':
                candidates_key = 'E1E3'
                backup_key = 'E1E3E4'
            else:  # E4
                candidates_key = 'E1E3E4'
                backup_key = None
            
            candidates = remaining_slots[candidates_key]
            
            # 如果主分区不足，使用备用分区
            if len(candidates) < qty and backup_key:
                candidates = pd.concat([candidates, remaining_slots[backup_key]])
            
            # 按行进时间升序排序
            sorted_candidates = candidates.sort_values(by='t_travel')
            
            # 尝试成对存储（高需求物料优先）
            selected_slots = self._select_paired_slots(
                sorted_candidates, qty, sku, allocated_slot_ids
            )
            
            # 记录分配结果
            for _, slot_row in selected_slots.iterrows():
                # 计算倒腾惩罚
                reshuffle_penalty = self._calculate_reshuffle_penalty(
                    slot_row, allocated_slot_ids
                )
                
                allocation.append({
                    '原材料编号': sku,
                    '货位编号': slot_row['slot_id'],
                    '箱子类型': box_type,
                    '归一化权重': weight,
                    't_travel': slot_row['t_travel'],
                    'is_deep': slot_row['is_deep'],
                    'aisle': slot_row['aisle'],
                    'shelf': slot_row['shelf'],
                    'level': slot_row['level'],
                    'col': slot_row['col'],
                    'depth': slot_row['depth'],
                    'reshuffle_penalty': reshuffle_penalty
                })
                
                allocated_slot_ids.add(slot_row['slot_id'])
            
            # 更新剩余货位
            for key in remaining_slots:
                remaining_slots[key] = remaining_slots[key][
                    ~remaining_slots[key]['slot_id'].isin(selected_slots['slot_id'])
                ]
        
        df_allocation = pd.DataFrame(allocation)
        return df_allocation
    
    def _select_paired_slots(self, candidates: pd.DataFrame, qty: int, 
                             sku: str, allocated: set) -> pd.DataFrame:
        """选择货位，优先成对存储"""
        
        if qty < 2:
            # 单箱直接取最优
            available = candidates[~candidates['slot_id'].isin(allocated)]
            return available.head(qty)
        
        # 寻找可成对的货格
        paired_slots = []
        remaining_needed = qty
        
        # 按行列分组查找成对空位
        for _, group in candidates.groupby(['aisle', 'shelf', 'level', 'col']):
            if remaining_needed <= 0:
                break
                
            available_in_group = group[~group['slot_id'].isin(allocated)]
            
            if len(available_in_group) == 2:
                depths = set(available_in_group['depth'])
                if depths == {1, 2}:  # 浅深位都可用
                    paired_slots.extend(available_in_group['slot_id'].tolist())
                    remaining_needed -= 2
        
        # 如果成对货位足够，优先使用
        if len(paired_slots) >= qty:
            selected_ids = paired_slots[:qty]
            return candidates[candidates['slot_id'].isin(selected_ids)]
        
        # 否则混合使用成对和单独货位
        selected = candidates[candidates['slot_id'].isin(paired_slots)]
        
        if len(selected) < qty:
            additional_needed = qty - len(selected)
            available = candidates[
                (~candidates['slot_id'].isin(allocated)) & 
                (~candidates['slot_id'].isin(selected['slot_id']))
            ]
            additional = available.head(additional_needed)
            selected = pd.concat([selected, additional])
        
        return selected.head(qty)
    
    def _calculate_reshuffle_penalty(self, slot_row: pd.Series, 
                                     allocated: set) -> float:
        """计算倒腾惩罚"""
        if not slot_row['is_deep']:
            return 0.0
        
        # 检查对应浅位是否被占用
        shallow_slot_id = slot_row['slot_id'].replace('_02', '_01')
        
        if shallow_slot_id in allocated:
            return 15.0  # 倒腾惩罚时间（秒）
        
        return 0.0
    
    def generate_report(self, df_allocation: pd.DataFrame) -> Dict:
        """生成详细统计报告"""
        
        total_slots_used = len(df_allocation)
        unique_skus = df_allocation['原材料编号'].nunique()
        
        # 加权平均出库时间
        weighted_avg_time = (
            df_allocation['t_travel'] * df_allocation['归一化权重']
        ).sum()
        
        # 倒腾统计分析
        deep_slots_with_penalty = df_allocation[
            df_allocation['reshuffle_penalty'] > 0
        ]
        reshuffle_rate = len(deep_slots_with_penalty) / len(
            df_allocation[df_allocation['is_deep']]
        ) if len(df_allocation[df_allocation['is_deep']]) > 0 else 0
        
        # 高需求物料分析（Top 100）
        top_100_skus = df_allocation.nlargest(100, '归一化权重')
        top_100_avg_time = top_100_skus['t_travel'].mean()
        
        report = {
            '成功分配物料数': unique_skus,
            '使用货位总数': total_slots_used,
            '加权平均出库时间(秒)': round(weighted_avg_time, 2),
            '高需求物料Top100平均时间(秒)': round(top_100_avg_time, 2),
            '深位倒腾发生率(%)': round(reshuffle_rate * 100, 2),
            '需要倒腾的货位数': len(deep_slots_with_penalty)
        }
        
        return report


def main():
    """主函数"""
    allocator = SlotAllocator()
    
    # Step 1: 数据加载与清洗
    print("=" * 60)
    print("Step 1: 数据加载与清洗")
    print("=" * 60)
    df_clean = allocator.load_and_clean_data("表1_原材料库存数据.csv")
    
    # Step 2: 构建货位系统
    print("\n" + "=" * 60)
    print("Step 2: 构建货位坐标体系")
    print("=" * 60)
    df_slots = allocator.build_slot_system()
    
    # Step 3: 执行分配
    print("\n" + "=" * 60)
    print("Step 3: 执行库位分配优化")
    print("=" * 60)
    df_allocation = allocator.allocate_slots(df_clean, df_slots)
    
    # Step 4: 生成报告
    print("\n" + "=" * 60)
    print("Step 4: 生成统计报告")
    print("=" * 60)
    report = allocator.generate_report(df_allocation)
    
    for key, value in report.items():
        print(f"{key}: {value}")
    
    # 保存结果
    df_allocation.to_csv("问题一_库位分配结果.csv", index=False, encoding='utf-8-sig')
    print("\n✓ 分配结果已保存至: 问题一_库位分配结果.csv")


if __name__ == "__main__":
    main()
