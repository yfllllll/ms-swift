import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class LabelmeDataChecker:
    def __init__(self, image_dir: str, json_dir: str):
        """
        初始化数据检查器
        
        Args:
            image_dir: 图像文件夹路径
            json_dir: JSON标注文件文件夹路径
        """
        self.image_dir = Path(image_dir)
        self.json_dir = Path(json_dir)
        
        # 验证文件夹是否存在
        if not self.image_dir.exists():
            raise ValueError(f"图像文件夹不存在: {image_dir}")
        if not self.json_dir.exists():
            raise ValueError(f"JSON文件夹不存在: {json_dir}")
    
    def check_json_format(self, json_path: Path) -> Tuple[bool, str]:
        """
        检查JSON文件格式是否正确
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查必需的基本字段
            required_fields = [
                "version", "flags", "shapes", "imagePath", 
                "imageHeight", "imageWidth", "vqa_data", "generation_timestamp"
            ]
            
            for field in required_fields:
                if field not in data:
                    return False, f"缺少必需字段: {field}"
            
            # 检查vqa_data结构
            vqa_data = data.get("vqa_data", {})
            vqa_required_fields = ["image_description", "qa_pairs", "model_info"]
            
            for field in vqa_required_fields:
                if field not in vqa_data:
                    return False, f"vqa_data中缺少字段: {field}"
            
            # 检查qa_pairs是否为列表且包含必需字段
            qa_pairs = vqa_data.get("qa_pairs", [])
            if not isinstance(qa_pairs, list):
                return False, "qa_pairs应为列表类型"
            
            for i, qa in enumerate(qa_pairs):
                if not isinstance(qa, dict):
                    return False, f"qa_pairs[{i}]应为字典类型"
                
                qa_required = ["question", "answer", "question_type"]
                for field in qa_required:
                    if field not in qa:
                        return False, f"qa_pairs[{i}]中缺少字段: {field}"
            
            # 检查model_info结构
            model_info = vqa_data.get("model_info", {})
            if not isinstance(model_info, dict):
                return False, "model_info应为字典类型"
            
            # 检查图像文件是否存在
            image_path = self.image_dir / data["imagePath"]
            if not image_path.exists():
                return False, f"对应的图像文件不存在: {data['imagePath']}"
            
            # 检查图像尺寸是否为有效数字
            if not isinstance(data["imageHeight"], (int, float)) or data["imageHeight"] <= 0:
                return False, f"imageHeight应为正数: {data['imageHeight']}"
            if not isinstance(data["imageWidth"], (int, float)) or data["imageWidth"] <= 0:
                return False, f"imageWidth应为正数: {data['imageWidth']}"
            
            return True, "格式正确"
            
        except json.JSONDecodeError as e:
            return False, f"JSON解析错误: {str(e)}"
        except Exception as e:
            return False, f"未知错误: {str(e)}"
    
    def collect_valid_files(self) -> Tuple[List[Path], List[Path], List[Tuple[Path, str]]]:
        """
        收集所有有效的文件
        
        Returns:
            (有效JSON文件列表, 有效图像文件列表, [(无效文件, 错误信息)])
        """
        valid_jsons = []
        valid_images = []
        invalid_files = []
        
        # 遍历JSON文件夹中的所有JSON文件
        json_files = list(self.json_dir.glob("*.json"))
        
        print(f"找到 {len(json_files)} 个JSON文件")
        
        for json_file in json_files:
            is_valid, error_msg = self.check_json_format(json_file)
            
            if is_valid:
                # 读取JSON文件获取图像文件名
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                image_file = self.image_dir / data["imagePath"]
                
                if image_file.exists():
                    valid_jsons.append(json_file)
                    valid_images.append(image_file)
                else:
                    invalid_files.append((json_file, f"图像文件不存在: {data['imagePath']}"))
            else:
                invalid_files.append((json_file, error_msg))
        
        # 去重（同一个图像可能对应多个JSON）
        valid_images = list(set(valid_images))
        
        return valid_jsons, valid_images, invalid_files
    
    def save_valid_files(self, output_dir: str, copy_images: bool = True) -> Dict:
        """
        将有效的文件保存到指定文件夹
        
        Args:
            output_dir: 输出文件夹路径
            copy_images: 是否复制图像文件（True）或只复制JSON文件（False）
            
        Returns:
            处理结果的统计信息
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建子文件夹
        json_output_dir = output_path 
        image_output_dir = output_path 
        
        json_output_dir.mkdir(exist_ok=True)
        if copy_images:
            image_output_dir.mkdir(exist_ok=True)
        
        # 收集有效文件
        valid_jsons, valid_images, invalid_files = self.collect_valid_files()
        
        # 复制有效文件
        copied_jsons = 0
        copied_images = 0
        
        # 复制JSON文件
        for json_file in valid_jsons:
            try:
                shutil.copy2(json_file, json_output_dir / json_file.name)
                copied_jsons += 1
            except Exception as e:
                invalid_files.append((json_file, f"复制失败: {str(e)}"))
        
        # 复制图像文件
        if copy_images:
            for image_file in valid_images:
                try:
                    shutil.copy2(image_file, image_output_dir / image_file.name)
                    copied_images += 1
                except Exception as e:
                    invalid_files.append((image_file, f"复制失败: {str(e)}"))
        
        # 保存错误日志
        if invalid_files:
            error_log_path = output_path / "error_log.txt"
            with open(error_log_path, 'w', encoding='utf-8') as f:
                f.write("无效文件列表:\n")
                f.write("=" * 50 + "\n")
                for file_path, error in invalid_files:
                    f.write(f"{file_path.name}: {error}\n")
        
        # 生成统计信息
        stats = {
            "total_json_files": len(list(self.json_dir.glob("*.json"))),
            "valid_json_files": len(valid_jsons),
            "valid_image_files": len(valid_images),
            "invalid_files": len(invalid_files),
            "copied_json_files": copied_jsons,
            "copied_image_files": copied_images,
            "output_directory": str(output_path),
            "json_output_directory": str(json_output_dir),
            "image_output_directory": str(image_output_dir) if copy_images else None,
            "error_log": str(error_log_path) if invalid_files else None
        }
        
        # 保存统计信息
        stats_path = output_path / "processing_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        return stats


def main():
    """主函数"""
    print("=" * 60)
    print("Labelme VQA 数据检查与整理工具")
    print("=" * 60)
    
    # 获取用户输入路径,增加默认路径
    # 增加默认路径提示
    default_image_dir = "/mnt/data/lyf/datasets/important/needed/images"
    default_json_dir = "/mnt/data/lyf/datasets/important/needed/vqa_data/all"
    default_output_dir = "/mnt/data/lyf/datasets/important/needed/vqa"
    print(f"默认图像文件夹路径: {default_image_dir}")
    print(f"默认JSON文件夹路径: {default_json_dir}")
    print(f"默认输出文件夹路径: {default_output_dir}")

    image_dir = input(f"请输入图像文件夹路径 (默认为 {default_image_dir}): ").strip()
    json_dir = input(f"请输入JSON文件夹路径 (默认为 {default_json_dir}): ").strip()
    output_dir = input(f"请输入输出文件夹路径 (默认为 {default_output_dir}): ").strip()
    if not image_dir:
        image_dir = default_image_dir
    if not json_dir:
        json_dir = default_json_dir
    if not output_dir:
        output_dir = default_output_dir

    copy_images_input = input("是否复制图像文件? (y/n, 默认为 y): ").strip().lower()
    copy_images = copy_images_input != 'n'
    
    try:
        # 创建检查器实例
        checker = LabelmeDataChecker(image_dir, json_dir)
        
        # 执行检查并保存有效文件
        print("\n正在检查数据...")
        stats = checker.save_valid_files(output_dir, copy_images)
        
        # 输出结果
        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)
        print(f"统计信息:")
        print(f"  总JSON文件数: {stats['total_json_files']}")
        print(f"  有效JSON文件数: {stats['valid_json_files']}")
        print(f"  有效图像文件数: {stats['valid_image_files']}")
        print(f"  无效文件数: {stats['invalid_files']}")
        print(f"  已复制JSON文件: {stats['copied_json_files']}")
        if copy_images:
            print(f"  已复制图像文件: {stats['copied_image_files']}")
        print(f"\n输出目录: {stats['output_directory']}")
        print(f"JSON文件保存到: {stats['json_output_directory']}")
        if copy_images:
            print(f"图像文件保存到: {stats['image_output_directory']}")
        
        if stats['invalid_files'] > 0:
            print(f"\n⚠️  发现 {stats['invalid_files']} 个无效文件，详细信息请查看:")
            print(f"   {stats['error_log']}")
        
        print(f"\n统计信息已保存到: {stats['output_directory']}/processing_stats.json")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        sys.exit(1)


def quick_check():
    """快速检查模式，不复制文件，只显示统计信息"""
    print("=" * 60)
    print("Labelme VQA 数据快速检查")
    print("=" * 60)
    
    image_dir = input("请输入图像文件夹路径: ").strip()
    json_dir = input("请输入JSON文件夹路径: ").strip()
    
    try:
        checker = LabelmeDataChecker(image_dir, json_dir)
        valid_jsons, valid_images, invalid_files = checker.collect_valid_files()
        
        print("\n" + "=" * 60)
        print("检查结果:")
        print("=" * 60)
        print(f"总JSON文件数: {len(list(Path(json_dir).glob('*.json')))}")
        print(f"有效JSON文件数: {len(valid_jsons)}")
        print(f"有效图像文件数: {len(valid_images)}")
        print(f"无效文件数: {len(invalid_files)}")
        
        if invalid_files:
            print(f"\n无效文件详情 (前10个):")
            print("-" * 60)
            for i, (file_path, error) in enumerate(invalid_files[:10]):
                print(f"{i+1}. {file_path.name}: {error}")
            
            if len(invalid_files) > 10:
                print(f"... 还有 {len(invalid_files) - 10} 个无效文件")
        
        # 显示一些有效文件示例
        if valid_jsons:
            print(f"\n有效文件示例 (前5个):")
            print("-" * 60)
            for i, json_file in enumerate(valid_jsons[:5]):
                print(f"{i+1}. {json_file.name}")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")


if __name__ == "__main__":
    print("请选择模式:")
    print("1. 完整检查并复制有效文件")
    print("2. 快速检查（只显示统计）")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        quick_check()
    else:
        print("无效选择，请重新运行程序")