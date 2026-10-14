#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO标签统计程序
功能：
1. 统计COCO格式JSON文件中的类别数量、名称和索引
2. 统计每个类别的实例数量

使用方法：
python coco_analyzer.py

或者在代码中调用：
from coco_analyzer import analyze_coco_annotations
result = analyze_coco_annotations("path/to/your/coco.json")
"""

import json
import os
import sys
from collections import defaultdict

def analyze_coco_annotations(json_file_path):
    """
    分析COCO格式的标注文件
    
    Args:
        json_file_path (str): COCO JSON文件的路径
    
    Returns:
        dict: 包含统计信息的字典，如果失败返回None
        {
            'total_categories': int,           # 总类别数
            'categories': list,                # [(id, name), ...] 类别列表
            'total_annotations': int,          # 总标注数
            'category_instances': dict,        # {category_id: count, ...} 每个类别的实例数
            'category_names': dict             # {category_id: name, ...} 类别ID到名称的映射
        }
    """
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误：文件 {json_file_path} 不存在")
        return None
    
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        print(f"成功读取文件: {json_file_path}")
        print("=" * 50)
        
        # 检查必要的字段
        if 'categories' not in coco_data:
            print("错误：JSON文件中缺少'categories'字段")
            return None
        
        if 'annotations' not in coco_data:
            print("错误：JSON文件中缺少'annotations'字段")
            return None
        
        # 1. 统计类别信息
        categories = coco_data['categories']
        num_categories = len(categories)
        
        print(f"1. 类别统计:")
        print(f"   总类别数: {num_categories}")
        print(f"   类别详情:")
        
        # 创建类别ID到名称的映射
        category_id_to_name = {}
        for category in categories:
            if 'id' not in category or 'name' not in category:
                print("警告：发现格式不完整的类别信息")
                continue
            cat_id = category['id']
            cat_name = category['name']
            category_id_to_name[cat_id] = cat_name
            print(f"   - 索引 {cat_id}: {cat_name}")
        
        print("\n" + "=" * 50)
        
        # 2. 统计每个类别的实例数量
        annotations = coco_data['annotations']
        category_instance_count = defaultdict(int)
        
        # 统计每个类别的实例数
        valid_annotations = 0
        for annotation in annotations:
            if 'category_id' not in annotation:
                print("警告：发现缺少category_id的标注")
                continue
            category_id = annotation['category_id']
            category_instance_count[category_id] += 1
            valid_annotations += 1
        
        print(f"2. 每个类别的实例数量:")
        print(f"   总标注数: {len(annotations)}")
        if valid_annotations != len(annotations):
            print(f"   有效标注数: {valid_annotations}")
        print(f"   各类别实例统计:")
        
        # 按类别ID排序显示
        total_instances = 0
        for cat_id in sorted(category_id_to_name.keys()):
            cat_name = category_id_to_name[cat_id]
            instance_count = category_instance_count[cat_id]
            total_instances += instance_count
            print(f"   - {cat_name} (ID: {cat_id}): {instance_count} 个实例")
        
        # 检查是否有未知类别的标注
        unknown_categories = set(category_instance_count.keys()) - set(category_id_to_name.keys())
        if unknown_categories:
            print(f"\n   警告：发现未定义的类别ID: {unknown_categories}")
            for unknown_id in unknown_categories:
                count = category_instance_count[unknown_id]
                print(f"   - 未知类别 (ID: {unknown_id}): {count} 个实例")
                total_instances += count
        
        print(f"\n   验证: 总实例数 = {total_instances}")
        
        # 返回统计结果
        result = {
            'total_categories': num_categories,
            'categories': [(cat['id'], cat['name']) for cat in categories if 'id' in cat and 'name' in cat],
            'total_annotations': len(annotations),
            'valid_annotations': valid_annotations,
            'category_instances': dict(category_instance_count),
            'category_names': category_id_to_name
        }
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return None
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return None

def main():
    """
    主函数，处理用户输入的文件路径
    """
    print("COCO标签统计程序")
    print("=" * 50)
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
        print(f"使用命令行参数指定的文件: {json_file_path}")
    
    # 如果路径被引号包围，去除引号
    if json_file_path.startswith('"') and json_file_path.endswith('"'):
        json_file_path = json_file_path[1:-1]
    elif json_file_path.startswith("'") and json_file_path.endswith("'"):
        json_file_path = json_file_path[1:-1]
    
    print(f"\n正在分析文件: {json_file_path}")
    print("=" * 50)
    
    # 分析文件
    result = analyze_coco_annotations(json_file_path)
    
    if result:
        print("\n" + "=" * 50)
        print("分析完成！")
    else:
        print("\n分析失败，请检查文件路径和格式。")

if __name__ == "__main__":
    main()