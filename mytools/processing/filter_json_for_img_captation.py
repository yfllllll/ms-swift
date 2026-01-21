import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import json
import shutil

def filter_description_json_files(input_dir, output_dir, max_boxes=135):
    """
    筛选符合条件的JSON标注文件和对应的图像文件，并复制到指定文件夹
    
    参数:
        input_dir: 输入目录，包含JSON文件和对应的图像文件
        output_dir: 输出目录，用于存放符合条件的文件
        max_boxes: 最大允许的框数量，超过此数量的文件将被过滤
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    valid_count = 0
    filtered_count = 0
    
    # 遍历所有JSON文件
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue
            
        json_path = os.path.join(input_dir, filename)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取JSON中记录的图像文件名
            image_filename = data.get('imagePath', '')
            if not image_filename:
                # 如果没有imagePath字段，尝试使用同名图像文件
                base_name = os.path.splitext(filename)[0]
                # 查找匹配的图像文件
                image_filename = None
                for ext in image_extensions:
                    potential_image = base_name + ext
                    if os.path.exists(os.path.join(input_dir, potential_image)):
                        image_filename = potential_image
                        break
            
            # 检查描述生成脚本产生的JSON格式
            # 主要检查descriptions字段
            descriptions = data.get('descriptions', [])
            
            # 规则1: 必须有descriptions字段且是列表
            if not isinstance(descriptions, list):
                print(f"过滤文件: {filename} (descriptions字段不是列表)")
                filtered_count += 1
                continue
            
            # 规则2: 描述数量必须是5个
            if len(descriptions) != 5:
                print(f"过滤文件: {filename} (描述数量不是5个，实际为{len(descriptions)})")
                filtered_count += 1
                continue
            
            # 规则3: 检查每个描述的质量
            valid_descriptions_count = 0
            has_error_description = False
            
            for desc in descriptions:
                # 检查描述结构
                if not isinstance(desc, dict):
                    has_error_description = True
                    print(f"  警告: {filename} 中的描述不是字典格式")
                    break
                
                # 检查是否有description字段
                desc_text = desc.get('description', '')
                if not desc_text or not isinstance(desc_text, str):
                    has_error_description = True
                    print(f"  警告: {filename} 中的描述文本无效")
                    break
                
                # 检查描述是否太短或包含错误信息
                # desc_text_lower = desc_text.lower()
                # if (len(desc_text.strip()) < 10 or 
                #     'error' in desc_text_lower or 
                #     '出错' in desc_text_lower or 
                #     '无法' in desc_text_lower):
                #     has_error_description = True
                #     print(f"  警告: {filename} 中的描述质量不佳: {desc_text[:50]}...")
                #     break
                
                valid_descriptions_count += 1
            
            if has_error_description or valid_descriptions_count != 5:
                print(f"过滤文件: {filename} (描述质量不合格)")
                filtered_count += 1
                continue
            
            # 规则4: 检查图像尺寸是否合理
            image_height = data.get('imageHeight', 0)
            image_width = data.get('imageWidth', 0)
            
            if image_height <= 0 or image_width <= 0:
                print(f"过滤文件: {filename} (图像尺寸无效: {image_width}x{image_height})")
                filtered_count += 1
                continue
            
            # 规则5: 检查对应的图像文件是否存在
            if not image_filename or not os.path.exists(os.path.join(input_dir, image_filename)):
                print(f"过滤文件: {filename} (找不到对应的图像文件: {image_filename})")
                filtered_count += 1
                continue
            
            # # 规则6: 如果存在shapes字段，检查框数量（可选）
            # shapes = data.get('shapes', [])
            # if shapes and len(shapes) > max_boxes:
            #     print(f"过滤文件: {filename} (框数量过多: {len(shapes)})")
            #     filtered_count += 1
            #     continue
            
            # 复制JSON文件和对应的图像文件到输出目录
            shutil.copy2(json_path, os.path.join(output_dir, filename))
            shutil.copy2(
                os.path.join(input_dir, image_filename), 
                os.path.join(output_dir, image_filename)
            )
            
            valid_count += 1
            print(f"保留文件: {filename} 和 {image_filename} (描述数量: 5, 图像尺寸: {image_width}x{image_height})")
        
        except json.JSONDecodeError as e:
            print(f"过滤文件: {filename} (JSON格式错误: {str(e)})")
            filtered_count += 1
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            filtered_count += 1
    
    print(f"\n处理完成:")
    print(f"- 输入目录: {input_dir}")
    print(f"- 输出目录: {output_dir}")
    print(f"- 保留文件: {valid_count} 对 (JSON+图像)")
    print(f"- 过滤文件: {filtered_count} 个")
    
    # 返回统计信息
    return {
        'total_processed': valid_count + filtered_count,
        'valid_count': valid_count,
        'filtered_count': filtered_count,
        'valid_ratio': valid_count / (valid_count + filtered_count) if (valid_count + filtered_count) > 0 else 0
    }


def analyze_description_quality(input_dir):
    """
    分析描述生成质量
    """
    print("分析描述生成质量...")
    
    total_files = 0
    desc_lengths = []
    desc_keywords = {}
    
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue
            
        json_path = os.path.join(input_dir, filename)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            descriptions = data.get('descriptions', [])
            if not isinstance(descriptions, list):
                continue
            
            total_files += 1
            
            for desc in descriptions:
                if isinstance(desc, dict):
                    desc_text = desc.get('description', '')
                    # 记录描述长度
                    desc_lengths.append(len(desc_text))
                    
                    # 简单关键词统计
                    words = desc_text.lower().split()
                    for word in words[:20]:  # 只统计前20个词
                        if len(word) > 3:  # 只统计长度大于3的词
                            desc_keywords[word] = desc_keywords.get(word, 0) + 1
        
        except:
            continue
    
    if total_files > 0:
        print(f"分析完成:")
        print(f"- 总文件数: {total_files}")
        print(f"- 平均描述长度: {sum(desc_lengths)/len(desc_lengths):.1f} 字符")
        print(f"- 最短描述: {min(desc_lengths)} 字符")
        print(f"- 最长描述: {max(desc_lengths)} 字符")
        
        # 显示出现频率最高的关键词
        sorted_keywords = sorted(desc_keywords.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n前10个高频关键词:")
        for word, count in sorted_keywords:
            print(f"  {word}: {count}次")


if __name__ == "__main__":
    # 使用示例
    input_directory = "/mnt/data/lyf/datasets/1431_part1/descriptions"  # 替换为您的实际路径
    output_directory = "/mnt/data/lyf/datasets/1431_part1/filtered_descriptions"  # 替换为您想要的输出路径
    
    # 运行过滤
    stats = filter_description_json_files(input_directory, output_directory, max_boxes=120)
    
    print(f"\n过滤统计:")
    print(f"- 总计处理文件: {stats['total_processed']}")
    print(f"- 合格文件: {stats['valid_count']}")
    print(f"- 过滤文件: {stats['filtered_count']}")
    print(f"- 合格率: {stats['valid_ratio']*100:.1f}%")
    
    # 分析原始数据质量（可选）
    print(f"\n{'='*50}")
    analyze_description_quality(input_directory)