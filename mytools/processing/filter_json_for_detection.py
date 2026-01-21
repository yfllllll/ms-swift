import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import json
import shutil

def filter_and_copy_valid_files(input_dir, output_dir, max_boxes=135):
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
            
            # 检查是否为空文件或框数量过多
            shapes = data.get('shapes', [])
            if len(shapes) == 0 or len(shapes) > max_boxes:
                print(f"过滤文件: {filename} (框数量: {len(shapes)})")
                filtered_count += 1
                continue
            
            # 检查对应的图像文件是否存在
            if not image_filename or not os.path.exists(os.path.join(input_dir, image_filename)):
                print(f"警告: 找不到 {filename} 对应的图像文件")
                filtered_count += 1
                continue
            
            # 复制JSON文件和对应的图像文件到输出目录
            shutil.copy2(json_path, os.path.join(output_dir, filename))
            shutil.copy2(
                os.path.join(input_dir, image_filename), 
                os.path.join(output_dir, image_filename)
            )
            
            valid_count += 1
            print(f"保留文件: {filename} 和 {image_filename} (框数量: {len(shapes)})")
        
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            filtered_count += 1
    
    print(f"处理完成:")
    print(f"- 输入目录: {input_dir}")
    print(f"- 输出目录: {output_dir}")
    print(f"- 保留文件: {valid_count} 对 (JSON+图像)")
    print(f"- 过滤文件: {filtered_count} 个")

if __name__ == "__main__":
    # 使用示例
    input_directory = "/data/tmp/1431_part1/images"  # 替换为您的实际路径
    output_directory = "/data/tmp/1431_part1/labelme"  # 替换为您想要的输出路径
    
    filter_and_copy_valid_files(input_directory, output_directory, max_boxes=120)