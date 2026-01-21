# labelme_preprocessor.py  
import numpy as np  
from swift.llm import RowPreprocessor, DatasetMeta, register_dataset  
from swift.llm.dataset.preprocessor.extra import GroundingMixin  
from swift.llm.template import register_template  
from swift.llm.template.template.qwen import Qwen2VLTemplate, QwenTemplateMeta  
from swift.llm.template.constant import MLLMTemplateType
from swift.llm.template.template_inputs import StdTemplateInputs  
from typing import Dict, Any, Optional, List, Union
from swift.llm.template.base import Context  
import glob  
import json  
import os  
from datasets import Dataset as HfDataset
import pickle
from collections import defaultdict
import random

# 全局变量存储数据集的类别信息
DATASET_CATEGORIES = {}

# 定义负样本的特殊标记
NEGATIVE_BBOX_MARKER = [0, 0, 0, 0]  # 使用[0,0,0,0]作为负样本标记

# 图像描述任务的提示词模板（多种风格）
IMAGE_CAPTION_PROMPTS = [
    "请详细描述这张图片的内容，包括场景、物体、颜色、动作、情感等所有你注意到的细节。",
    "请用中文详细描述这张图像，包括场景、主要物体、颜色、布局和任何有趣的细节。",
    "请全面描述这张图片，包括背景、前景、人物、物体、颜色、光线和整体氛围。",
    "请仔细观察这张图片，然后详细描述你看到的所有内容。",
    "请用生动的语言描述这张图片，让看不到图片的人也能想象出画面。",
    "请描述这张图片的场景和其中包含的物体，注意细节。",
    "请分析这张图片的视觉内容，包括主要元素、颜色搭配和空间布局。",
    "请描述这张图片，包括其中的物体、人物、场景和环境细节。",
    "请详细说明这张图片展示了什么，尽可能描述得具体详细。",
    "请用丰富的细节描述这张图片，包括视觉元素和可能的情境。",
]

# 区域描述任务的提示词模板
REGION_CAPTION_PROMPTS = [
    "请对以下每个区域进行详细描述，输出格式为:\nregion_id: label|brief instance description\n\n要求：\n1. label使用中文名词\n2. description用中文简要描述该物体的特征、状态等\n3. 每个区域单独一行",
    "请描述以下每个指定区域的内容，格式为:\n区域ID: 类别|实例描述\n\n注意：\n1. 类别使用中文\n2. 描述要简洁明了\n3. 每个区域一行",
    "请针对以下每个区域提供描述，按照以下格式：\n编号: 物体类别|简要描述\n\n要求：\n- 类别用中文名词\n- 描述用中文\n- 每行一个区域",
    "请为以下每个图像区域生成描述，格式要求：\n区域编号: 物体名称|特征描述\n\n注意：\n1. 使用中文\n2. 描述要具体\n3. 每行对应一个区域",
]

# 目标检测任务的提示词模板（中文）
GROUNDING_PROMPTS_ZH = [
    "请找到图像中{objects}的位置",
    "请在图像中定位{objects}",
    "请标出{objects}在图像中的位置",
    "请找出图像中的{objects}并给出边界框",
    "请检测图像中的{objects}",
    "请识别并定位图像中的{objects}",
]

# 目标检测任务的提示词模板（英文）
GROUNDING_PROMPTS_EN = [
    "Detect [OBJ].",
    "detect [OBJ].",
    "detect [OBJ].",
    "detect [OBJ].",
    "detect [OBJ].",
    "detect [OBJ].",
    "Please detect [OBJ] in this image.",
    "Detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
    "Please detect [OBJ] in this image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
    "Find [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
    "Detect [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
    "Locate [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
    "Identify [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
    "Please locate [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
    "What is the location of [OBJ]? Return its bounding box as [x0, y0, x1, y1].",
    "Mark the region where [OBJ] appears using [x0, y0, x1, y1] format.",
    "Can you find [OBJ] in this picture? Give the coordinates as [x0, y0, x1, y1].",
    "Highlight [OBJ] in the image and output its bounding box in [x0, y0, x1, y1].",
    "Indicate where [OBJ] is located with bounding box coordinates [x0, y0, x1, y1].",
    "Show me the bounding box for [OBJ] in [x0, y0, x1, y1] format.",
    "Return the bounding box coordinates for [OBJ] in the image.",
    "Give the coordinates of the box around [OBJ] using [x0, y0, x1, y1] format.",
    "Determine the bounding box of [OBJ] and return it as [x0, y0, x1, y1].",
    "Identify the bounding box location of [OBJ] using the format [x0, y0, x1, y1].",
    "detect [OBJ].",
    "please detect [OBJ] in this image.",
    "detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
    "please detect [OBJ] in this image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
    "find [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
    "detect [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
    "locate [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
    "identify [OBJ] in the image. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
    "please locate [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
    "what is the location of [OBJ]? Return its bounding box as [x0, y0, x1, y1].",
    "mark the region where [OBJ] appears using [x0, y0, x1, y1] format.",
    "can you find [OBJ] in this picture? Give the coordinates as [x0, y0, x1, y1].",
    "highlight [OBJ] in the image and output its bounding box in [x0, y0, x1, y1].",
    "indicate where [OBJ] is located with bounding box coordinates [x0, y0, x1, y1].",
    "show me the bounding box for [OBJ] in [x0, y0, x1, y1] format.",
    "return the bounding box coordinates for [OBJ] in the image.",
    "give the coordinates of the box around [OBJ] using [x0, y0, x1, y1] format.",
    "determine the bounding box of [OBJ] and return it as [x0, y0, x1, y1].",
    "identify the bounding box location of [OBJ] using the format [x0, y0, x1, y1].",
]

# BBOX VQA任务的提示词模板
BBOX_VQA_PROMPTS = [
    "请回答以下关于图像中某些区域的问题：",
    "根据图片内容，回答以下关于特定区域的问题：",
    "请仔细观察图像中的指定区域，然后回答问题：",
    "针对以下每个区域，请回答问题：",
    "以下是关于图像中特定区域的问题，请一一回答：",
]

# 合并所有检测提示词
GROUNDING_PROMPTS = GROUNDING_PROMPTS_ZH + GROUNDING_PROMPTS_EN

# 1. 更新自定义模板类，处理负样本标记
class CustomQwen2VLTemplate(Qwen2VLTemplate):  
    """自定义模板,支持 <x0><y0><x1><y1> 格式的坐标输出"""  
    norm_bbox = 'norm1000'  
    
    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:  
        """将 bbox 转换为 <x0><y0><x1><y1> 格式，处理负样本标记"""  
        if bbox == NEGATIVE_BBOX_MARKER:
            return ['None']
        bbox_str = ''.join([f'<{int(coord)}>' for coord in bbox])  
        return [bbox_str]  
      
    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:  
        """保持 legacy 格式的 ref 标签"""  
        return [f'<|object_ref_start|>{ref}<|object_ref_end|>']  
  
  
# 2. Qwen2.5-VL 自定义模板  
class CustomQwen2_5VLTemplate(CustomQwen2VLTemplate):  
    version = 'v2_5'  
    norm_bbox = 'norm1000'  # 使用相对坐标  
  
  
# 3. Qwen3-VL 自定义模板  
class CustomQwen3VLTemplate(CustomQwen2VLTemplate):  
    version = 'v3'  
    norm_bbox = 'norm1000'  # 使用相对坐标

# 4. 注册所有自定义模板  
register_template(  
    QwenTemplateMeta(  
        MLLMTemplateType.qwen2_vl,  
        template_cls=CustomQwen2VLTemplate  
    ),
    exist_ok=True  # 添加这个参数    
)  
  
register_template(  
    QwenTemplateMeta(  
        MLLMTemplateType.qwen2_5_vl,  
        template_cls=CustomQwen2_5VLTemplate  
    ),  
    exist_ok=True  # 添加这个参数  
)  
  
register_template(  
    QwenTemplateMeta(  
        MLLMTemplateType.qwen3_vl,  
        template_cls=CustomQwen3VLTemplate  
    ),
    exist_ok=True  # 添加这个参数   
)  
  
  
# 5. 更新预处理器，支持图像描述、VQA任务和随机提示词
class LabelMeGroundingPreprocessor(RowPreprocessor, GroundingMixin):  
      
    def __init__(self, 
                 task_ratio=(0.3, 0.2, 0.2, 0.2, 0.1),  # 修改为五元组：(grounding, region_caption, image_caption, vqa, bbox_vqa)
                 max_categories=10, 
                 negative_sample_prob=0.3,  # 添加负样本的概率
                 max_negative_categories=3,  # 最大负样本类别数
                 dataset_path=None,  # 数据集路径，用于获取类别信息
                 use_random_prompts=True,  # 是否使用随机提示词
                 prompt_language='mixed',  # 提示词语言：'zh', 'en', 'mixed'
                 vqa_mode='single',  # VQA模式：'single'单问题，'multi'多问题，'mixed'混合
                 max_vqa_questions=3,  # 最大VQA问题数量
                 max_bbox_vqa_boxes=3,  # 最大处理的矩形框数量
                 max_bbox_vqa_qa_per_box=2,  # 每个矩形框最大问答对数量
                 **kwargs):  
        super().__init__(**kwargs)  
        self.task_ratio = task_ratio  
        self.max_categories = max_categories
        self.negative_sample_prob = negative_sample_prob
        self.max_negative_categories = max_negative_categories
        self.dataset_path = dataset_path
        self.all_categories = None
        self.use_random_prompts = use_random_prompts
        self.prompt_language = prompt_language
        self.vqa_mode = vqa_mode
        self.max_vqa_questions = max_vqa_questions
        self.max_bbox_vqa_boxes = max_bbox_vqa_boxes
        self.max_bbox_vqa_qa_per_box = max_bbox_vqa_qa_per_box
        
        # 根据语言选择过滤提示词
        if prompt_language == 'zh':
            self.grounding_prompts = GROUNDING_PROMPTS_ZH
        elif prompt_language == 'en':
            self.grounding_prompts = GROUNDING_PROMPTS_EN
        else:  # 'mixed'
            self.grounding_prompts = GROUNDING_PROMPTS
        
        # 加载全局类别信息
        if dataset_path and dataset_path in DATASET_CATEGORIES:
            self.all_categories = DATASET_CATEGORIES[dataset_path]
            print(f"Loaded {len(self.all_categories)} categories from dataset: {dataset_path}")
      
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:  
        shapes = row.get('shapes', []) or []
        descriptions = row.get('descriptions', []) or []  # 获取图像描述
        vqa_data = row.get('vqa_data', {}) or {}  # 获取VQA数据
        image_path = row.get('imagePath', '')  
        image_height = row.get('imageHeight', 0)
        image_width = row.get('imageWidth', 0)
        
        if image_height == 0 or image_width == 0:
            # 尝试从图像文件获取尺寸，如果获取不到则跳过
            print(f"Warning: image size not found for {image_path}")
            return None
            
        print(f"Processing image: {image_path} with {len(shapes)} shapes, {len(descriptions)} descriptions, vqa_data: {'yes' if vqa_data else 'no'}")  
        
        # 根据可用的数据类型调整任务比例
        available_tasks = []
        task_weights = []
        
        # 检查是否有目标检测数据
        has_grounding_data = len(shapes) > 0
        # 检查是否有图像描述数据
        has_caption_data = isinstance(descriptions, list) and len(descriptions) > 0
        # 检查是否有VQA数据
        has_vqa_data = vqa_data and 'qa_pairs' in vqa_data and len(vqa_data['qa_pairs']) > 0
        # 检查是否有bboxVQA数据
        has_bbox_vqa_data = False
        for shape in shapes:
            if shape.get('shape_type') == 'rectangle' and 'bboxvqa' in shape and shape['bboxvqa']:
                has_bbox_vqa_data = True
                break
        
        if has_grounding_data:
            available_tasks.extend(['grounding', 'region_caption'])
            task_weights.extend([self.task_ratio[0], self.task_ratio[1]])
        
        if has_caption_data:
            available_tasks.append('image_caption')
            task_weights.append(self.task_ratio[2])
        
        if has_vqa_data:
            available_tasks.append('vqa')
            task_weights.append(self.task_ratio[3])
            
        if has_bbox_vqa_data:
            available_tasks.append('bbox_vqa')
            task_weights.append(self.task_ratio[4])
        
        # 如果没有可用的任务数据，则跳过
        if not available_tasks:
            print(f"Warning: No available task data for {image_path}")
            return None
        
        # 归一化权重
        total_weight = sum(task_weights)
        if total_weight == 0:
            # 如果所有权重都为0，则均匀分配
            task_weights = [1.0 / len(available_tasks)] * len(available_tasks)
        else:
            task_weights = [w / total_weight for w in task_weights]
        
        # 随机选择任务类型
        task_type = random.choices(available_tasks, weights=task_weights)[0]
        self.task_type = task_type
        
        # 处理不同任务类型
        if task_type == 'grounding':  
            return self._process_grounding_task(shapes, image_path, image_width, image_height)
        elif task_type == 'region_caption':  
            return self._process_region_caption_task(shapes, image_path, image_width, image_height)
        elif task_type == 'image_caption':  
            return self._process_image_caption_task(descriptions, image_path)
        elif task_type == 'vqa':  
            return self._process_vqa_task(vqa_data, image_path)
        elif task_type == 'bbox_vqa':  
            return self._process_bbox_vqa_task(shapes, image_path, image_width, image_height)
        else:
            return None
    
    def _format_objects_for_prompt(self, categories, prompt_template):
        """根据提示词模板格式化对象列表"""
        # 将类别列表转换为适当的字符串表示
        if len(categories) == 1:
            objects_str = categories[0]
        elif len(categories) == 2:
            objects_str = f"{categories[0]} and {categories[1]}"
        else:
            objects_str = ', '.join(categories[:-1]) + f' and {categories[-1]}'
        
        # 替换占位符
        if '{objects}' in prompt_template:
            return prompt_template.format(objects=objects_str)
        elif '[OBJ]' in prompt_template:
            return prompt_template.replace('[OBJ]', objects_str)
        else:
            # 如果没有占位符，直接返回
            return prompt_template + f" {objects_str}"
    
    def _process_grounding_task(self, shapes, image_path, image_width, image_height):
        """处理目标检测任务"""
        # 按类别分组  
        category_groups = {}  
        for shape in shapes:  
            if shape['shape_type'] == 'rectangle':  
                label_parts = shape['label'].split('/', 1)  
                category = label_parts[0]  
                description = label_parts[1] if len(label_parts) > 1 else category  
                
                if category not in category_groups:  
                    category_groups[category] = []  
                
                # 存储原始坐标，在grounding任务中让模板处理归一化
                points = shape['points']
                x0 = float(points[0][0])
                y0 = float(points[0][1])
                x1 = float(points[1][0])
                y1 = float(points[1][1])
                
                category_groups[category].append({  
                    'bbox': [x0, y0, x1, y1],  # 原始坐标，让模板处理归一化
                    'bbox_norm': None,  # 归一化坐标，在caption任务中使用
                    'category': category,  
                    'description': description  
                })  
        
        # 获取当前样本中出现的类别
        current_categories = list(category_groups.keys())
        
        # 随机选择类别（只从当前样本中存在的类别中选择）
        num_categories = min(len(current_categories),   
                            np.random.randint(1, self.max_categories + 1))  
        selected_categories = np.random.choice(current_categories, num_categories, replace=False)  
        
        # 构造数据  
        response_parts = []  
        ref_list = []  
        bbox_list = []  
        
        # 为每个选中的正样本类别添加实例
        for category in selected_categories:  
            instances = category_groups[category]  
            response_parts.append('<ref-object>')  
            response_parts.append('<|box_start|>')  
            
            # 将这个类别添加到ref_list中（每个类别只添加一次）
            ref_list.append(category)
            
            for i, instance in enumerate(instances):  
                response_parts.append('<bbox>')  
                bbox_list.append(instance['bbox'])  # 原始坐标
                if i < len(instances) - 1:  
                    response_parts.append(', ')  
            
            response_parts.append('<|box_end|>')  
        
        # 添加负样本（不存在的类别）
        print("current_categories:", current_categories)
        if (self.all_categories is not None and 
            len(self.all_categories) > 0 and
            np.random.random() < self.negative_sample_prob):
            
            # 获取数据集中存在但当前样本中不存在的类别
            negative_candidates = list(set(self.all_categories) - set(current_categories))
            print("negative_candidates:", negative_candidates)
            if negative_candidates:
                # 随机选择一些负样本类别
                num_negative = min(len(negative_candidates), 
                                np.random.randint(1, self.max_negative_categories + 1))
                negative_categories = np.random.choice(negative_candidates, 
                                                    num_negative, 
                                                    replace=False)
                
                for category in negative_categories:
                    # 添加负样本到响应中
                    response_parts.append('<ref-object>')  
                    response_parts.append('<|box_start|>None<|box_end|>')
                    
                    # 添加到ref和bbox列表中
                    ref_list.append(category)
                    bbox_list.append(NEGATIVE_BBOX_MARKER)  # 使用特殊标记表示负样本
        
        # 随机选择提示词
        if self.use_random_prompts:
            prompt_template = random.choice(self.grounding_prompts)
            query = f"<image>" + self._format_objects_for_prompt(list(selected_categories), prompt_template)
        else:
            query = f"<image>找到{'、'.join(list(selected_categories))}的位置"
        
        # 如果有负样本，更新查询
        if len(ref_list) > len(selected_categories):
            # 构建包含负样本类别的完整列表
            all_categories_in_query = list(selected_categories)
            negative_categories_in_query = [ref for i, ref in enumerate(ref_list) if i >= len(selected_categories)]
            print("negative_categories_in_query:", negative_categories_in_query)
            all_categories_in_query.extend(negative_categories_in_query)
            
            if self.use_random_prompts:
                query = f"<image>" + self._format_objects_for_prompt(all_categories_in_query, prompt_template)
            else:
                query = f"<image>找到{'、'.join(all_categories_in_query)}的位置（有些可能不存在）"
        
        response = ''.join(response_parts)
        
        return {  
            'messages': [  
                {'role': 'user', 'content': query},  
                {'role': 'assistant', 'content': response}  
            ],  
            'images': [image_path],  
            'objects': {  
                'ref': ref_list,  
                'bbox': bbox_list,  
                'bbox_type': 'norm1000'  
            }  
        }
    
    def _process_region_caption_task(self, shapes, image_path, image_width, image_height):
        """处理区域描述任务"""
        # 按类别分组  
        category_groups = {}  
        for shape in shapes:  
            if shape['shape_type'] == 'rectangle':  
                label_parts = shape['label'].split('/', 1)  
                category = label_parts[0]  
                description = label_parts[1] if len(label_parts) > 1 else category  
                  
                if category not in category_groups:  
                    category_groups[category] = []  
                
                # 存储原始坐标
                points = shape['points']
                x0 = float(points[0][0])
                y0 = float(points[0][1])
                x1 = float(points[1][0])
                y1 = float(points[1][1])
                
                category_groups[category].append({  
                    'bbox': [x0, y0, x1, y1],
                    'category': category,  
                    'description': description  
                })  
        
        all_instances = []  
        for category in category_groups:  
            instances = category_groups[category]  
            all_instances.extend(instances)  
        
        if not all_instances:
            return None
        
        # 随机选择部分实例  
        sample_ratio = np.random.uniform(0.3, 1.0)  
        num_samples = max(1, int(len(all_instances) * sample_ratio))  
        sampled_indices = np.random.choice(len(all_instances), num_samples, replace=False)  
        
        # 构建region_data - 需要自己完成归一化到1000
        region_data = []  
        region_descriptions = []  # 用于存储每个区域的描述信息
        
        for idx in sampled_indices:  
            instance = all_instances[idx]  
            
            # 归一化坐标到[0, 1000]范围
            x0_norm = int(float(instance['bbox'][0]) / image_width * 1000)
            y0_norm = int(float(instance['bbox'][1]) / image_height * 1000)
            x1_norm = int(float(instance['bbox'][2]) / image_width * 1000)
            y1_norm = int(float(instance['bbox'][3]) / image_height * 1000)
            
            # 构建区域数据
            region_index = len(region_data) + 1
            region_data.append({
                'bbox_2d': [x0_norm, y0_norm, x1_norm, y1_norm],
                'label': f'region_{region_index}'
            })  
            
            # 提取类别和描述
            category = instance['category']
            description = instance['description']
            
            # 构建单个区域的描述行
            # 注意: 这里使用'|'而不是'/'作为分隔符
            region_description = f"{region_index}: {category}|{description}"
            region_descriptions.append(region_description)
        
        # 随机选择提示词模板
        if self.use_random_prompts:
            prompt_template = random.choice(REGION_CAPTION_PROMPTS)
        else:
            prompt_template = REGION_CAPTION_PROMPTS[0]
        
        # 构造 query
        query = f"<image>" + prompt_template + f"\n\n以下是需要描述的区域坐标：{json.dumps(region_data, ensure_ascii=False)}"
        
        # 构造 response - 按要求的格式组织
        response_lines = []
        for region_desc in region_descriptions:
            response_lines.append(region_desc)
        
        # 用换行符连接所有行
        response = '\n'.join(response_lines)
        
        return {  
            'messages': [  
                {'role': 'user', 'content': query},  
                {'role': 'assistant', 'content': response}  
            ],  
            'images': [image_path],
            'objects': {  
                'ref': [],  
                'bbox': [],  
                'bbox_type': 'norm1000'  
            } 
        }
    
    def _process_image_caption_task(self, descriptions, image_path):
        """处理图像描述任务 - 随机选择一个描述作为回答"""
        if not isinstance(descriptions, list) or len(descriptions) == 0:
            return None
        
        # 收集所有有效的描述
        valid_descriptions = []
        for desc_item in descriptions:
            if isinstance(desc_item, dict) and 'description' in desc_item:
                desc_text = desc_item['description']
                # 清理描述文本
                cleaned_text = desc_text.replace("```json", "").replace("```", "").strip()
                if cleaned_text and len(cleaned_text) > 10:  # 过滤掉太短的描述
                    valid_descriptions.append(cleaned_text)
        
        # 如果没有有效描述，则跳过
        if not valid_descriptions:
            return None
        
        # 随机选择一个描述作为答案
        answer = random.choice(valid_descriptions)
        
        # 随机选择提示词
        if self.use_random_prompts:
            query = f"<image>" + random.choice(IMAGE_CAPTION_PROMPTS)
        else:
            query = f"<image>请详细描述这张图像的内容。"
        
        return {  
            'messages': [  
                {'role': 'user', 'content': query},  
                {'role': 'assistant', 'content': answer}  
            ],  
            'images': [image_path],
            'objects': {  
                'ref': [],  
                'bbox': [],  
                'bbox_type': 'norm1000'  
            } 
        }
    
    def _process_vqa_task(self, vqa_data: Dict[str, Any], image_path: str) -> Optional[Dict[str, Any]]:
        """处理VQA（视觉问答）任务"""
        if not vqa_data or 'qa_pairs' not in vqa_data or not vqa_data['qa_pairs']:
            return None
        
        qa_pairs = vqa_data['qa_pairs']
        
        # 根据VQA模式处理
        if self.vqa_mode == 'single':
            return self._process_single_vqa(qa_pairs, image_path)
        elif self.vqa_mode == 'multi':
            return self._process_multi_vqa(qa_pairs, image_path)
        else:  # 'mixed'
            # 随机选择单问题或多问题模式
            if random.random() < 0.5:
                return self._process_single_vqa(qa_pairs, image_path)
            else:
                return self._process_multi_vqa(qa_pairs, image_path)
    
    def _process_single_vqa(self, qa_pairs: List[Dict[str, Any]], image_path: str) -> Dict[str, Any]:
        """处理单问题VQA任务"""
        # 随机选择一个QA对
        qa_pair = random.choice(qa_pairs)
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        question_type = qa_pair.get('question_type', '')
        
        if not question or not answer:
            return None
        
        # 随机选择提示词
        if self.use_random_prompts:
            # prompt = random.choice(VQA_PROMPTS)
            query = f"<image>" +f"{question}"
        else:
            query = f"<image>{question}"
        
        return {
            'messages': [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': answer}
            ],
            'images': [image_path],
            'objects': {
                'ref': [],
                'bbox': [],
                'bbox_type': 'norm1000'
            },

        }
    
    def _process_multi_vqa(self, qa_pairs: List[Dict[str, Any]], image_path: str) -> Dict[str, Any]:
        """处理多问题VQA任务"""
        # 随机选择1到max_vqa_questions个问题
        num_questions = min(len(qa_pairs), random.randint(1, self.max_vqa_questions))
        selected_qa_pairs = random.sample(qa_pairs, num_questions)
        
        # 构建问题和答案列表
        questions = []
        answers = []
        question_types = []
        
        for i, qa_pair in enumerate(selected_qa_pairs):
            question = qa_pair.get('question', '')
            answer = qa_pair.get('answer', '')
            question_type = qa_pair.get('question_type', '')
            
            if question and answer:
                questions.append(f"{i+1}. {question}")
                answers.append(f"{i+1}. {answer}")
                question_types.append(question_type)
        
        if not questions:
            return None
        
        # 构建完整的问答文本
        all_questions = "\n".join(questions)
        all_answers = "\n".join(answers)
        

        query = f"<image>请回答以下问题：\n{all_questions}"
        
        return {
            'messages': [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': all_answers}
            ],
            'images': [image_path],
            'objects': {
                'ref': [],
                'bbox': [],
                'bbox_type': 'norm1000'
            },
        }
    
    def _process_bbox_vqa_task(self, shapes, image_path, image_width, image_height):
        """处理矩形框问答对任务"""
        # 收集所有有bboxvqa的矩形框
        bbox_vqa_shapes = []
        for shape in shapes:
            if (shape.get('shape_type') == 'rectangle' and 
                'bboxvqa' in shape and 
                isinstance(shape['bboxvqa'], list) and 
                len(shape['bboxvqa']) > 0):
                
                # 提取矩形框信息
                label_parts = shape['label'].split('/', 1)
                category = label_parts[0]
                
                points = shape['points']
                x0 = float(points[0][0])
                y0 = float(points[0][1])
                x1 = float(points[1][0])
                y1 = float(points[1][1])
                
                # 归一化坐标到[0, 1000]范围
                x0_norm = int(x0 / image_width * 1000)
                y0_norm = int(y0 / image_height * 1000)
                x1_norm = int(x1 / image_width * 1000)
                y1_norm = int(y1 / image_height * 1000)
                
                bbox_vqa_shapes.append({
                    'bbox': [x0_norm, y0_norm, x1_norm, y1_norm],
                    'category': category,
                    'bboxvqa': shape['bboxvqa'],
                    'original_bbox': [x0, y0, x1, y1]
                })
        
        if not bbox_vqa_shapes:
            return None
        
        # 随机选择一些矩形框
        num_boxes = min(len(bbox_vqa_shapes), random.randint(1, self.max_bbox_vqa_boxes))
        selected_shapes = random.sample(bbox_vqa_shapes, num_boxes)
        
        # 构建region_data和问答对
        region_data = []
        questions_list = []
        answers_list = []
        
        for i, shape_data in enumerate(selected_shapes):
            region_index = i + 1
            region_coords = shape_data['bbox']
            category = shape_data['category']
            
            # 构建区域数据
            region_data.append({
                'bbox_2d': region_coords,
                'label': f'region_{region_index}'
            })
            
            # 随机选择这个矩形框的问答对
            available_qa_pairs = shape_data['bboxvqa']
            num_qa_pairs = min(len(available_qa_pairs), 
                              random.randint(1, self.max_bbox_vqa_qa_per_box))
            selected_qa_pairs = random.sample(available_qa_pairs, num_qa_pairs)
            
            for j, qa_pair in enumerate(selected_qa_pairs):
                question = qa_pair.get('question', '')
                answer = qa_pair.get('answer', '')
                
                if question and answer:
                    # 在问题中指明区域
                    questions_list.append(f"区域{region_index}（坐标：{region_coords}）：{question}")
                    answers_list.append(f"区域{region_index}：{answer}")
        
        if not questions_list:
            return None
        
        # 随机选择提示词
        if self.use_random_prompts:
            prompt_template = random.choice(BBOX_VQA_PROMPTS)
        else:
            prompt_template = BBOX_VQA_PROMPTS[0]
        
        # 构建query和response
        query = f"<image>{prompt_template}\n\n" + "\n".join(questions_list)
        response = "\n".join(answers_list)
        
        return {
            'messages': [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': response}
            ],
            'images': [image_path],
            'objects': {
                'ref': [],
                'bbox': [],
                'bbox_type': 'norm1000'
            }
        }


def load_labelme_folder(dataset_syntax, dataset_meta, **kwargs):  
    """自定义加载函数,读取文件夹中的所有 JSON 标注文件，并统计类别"""  
    dataset_path = dataset_syntax.dataset  
    print(f"Loading LabelMe dataset from: {dataset_path}") 
    # 读取所有 JSON 文件  
    json_files = glob.glob(os.path.join(dataset_path, '*.json'))  
      
    data_list = []  
    all_categories = set()  # 使用集合去重
    vqa_count = 0  # 统计VQA样本数量
    bbox_vqa_count = 0  # 统计bboxVQA样本数量
    
    for json_file in json_files:  
        try:
            with open(json_file, 'r', encoding='utf-8') as f:  
                data = json.load(f)  
                # 处理图像路径  
                if 'imagePath' in data and not os.path.isabs(data['imagePath']):  
                    data['imagePath'] = os.path.join(dataset_path, data['imagePath'])  
                    print(f"Resolved image path: {data['imagePath']}")
                
                # 统计VQA数据
                if 'vqa_data' in data and data['vqa_data']:
                    vqa_count += 1
                
                # 统计bboxVQA数据
                has_bbox_vqa = False
                for shape in data.get('shapes', []):
                    if shape['shape_type'] == 'rectangle':
                        label_parts = shape['label'].split('/', 1)
                        category = label_parts[0]
                        all_categories.add(category)
                        
                        # 检查是否有bboxvqa字段
                        if 'bboxvqa' in shape and shape['bboxvqa']:
                            has_bbox_vqa = True
                
                if has_bbox_vqa:
                    bbox_vqa_count += 1
                    
                data_list.append(data)
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
      
    # 保存类别信息到全局变量
    DATASET_CATEGORIES[dataset_path] = list(all_categories)
    
    # 保存类别信息到文件，以便后续使用
    categories_file = os.path.join(dataset_path, 'all_categories.pkl')
    with open(categories_file, 'wb') as f:
        pickle.dump(list(all_categories), f)
    
    print(f"Loaded {len(data_list)} JSON files")
    print(f"Found {len(all_categories)} unique categories from shapes")
    print(f"Found {vqa_count} samples with VQA data")
    print(f"Found {bbox_vqa_count} samples with bboxVQA data")
    print(f"Sample data check - First item has {len(data_list[0].get('shapes', []))} shapes and vqa_data: {'yes' if 'vqa_data' in data_list[0] else 'no'}")
        
    # 创建 HfDataset  
    dataset = HfDataset.from_list(data_list)  
    print(f"Dataset size before preprocessing: {len(dataset)}")  
    
    # 正确调用预处理器  
    preprocessor = dataset_meta.preprocess_func  
    
    # 如果预处理器有dataset_path属性，设置它以便获取类别信息
    if hasattr(preprocessor, 'dataset_path'):
        preprocessor.dataset_path = dataset_path
        # 直接从全局变量设置类别信息
        if dataset_path in DATASET_CATEGORIES:
            preprocessor.all_categories = DATASET_CATEGORIES[dataset_path]
    
    dataset = preprocessor(  
        dataset,  
        num_proc=kwargs.get('num_proc', 1),  
        load_from_cache_file=False,  
        strict=True  
    )  
    
    print(f"Dataset size after preprocessing: {len(dataset)}")  
    
    return dataset


# 6. 注册数据集  
register_dataset(  
    DatasetMeta(  
        dataset_path='/mnt/data/lyf/datasets/1431_part1/labelme',  
        dataset_name = "labelme_dataset1",
        preprocess_func=LabelMeGroundingPreprocessor(
            task_ratio=(0.5, 0.5, 0.0, 0.0, 0.0),  # grounding, region_caption, image_caption, vqa, bbox_vqa
            max_categories=8,
            negative_sample_prob=0.3,  # 30%的概率添加负样本
            max_negative_categories=3,  # 最多3个负样本类别
            dataset_path='/mnt/data/lyf/datasets/1431_part1/labelme',  # 传递数据集路径
            use_random_prompts=True,  # 使用随机提示词
            prompt_language='mixed',  # 使用混合语言提示词
            vqa_mode='mixed',  # 混合VQA模式
            max_vqa_questions=3,  # 最多3个VQA问题
            max_bbox_vqa_boxes=3,  # 最多处理3个矩形框
            max_bbox_vqa_qa_per_box=2  # 每个矩形框最多2个问答对
        ),  
        load_function=load_labelme_folder,  # 使用自定义加载函数  
    ))

register_dataset(  
    DatasetMeta(  
        dataset_path='/mnt/disk/lyf/datasets/important/needed/labelme_train',  
        dataset_name = "labelme_dataset2",
        preprocess_func=LabelMeGroundingPreprocessor(
            task_ratio=(0.5, 0.5, 0.0, 0.0, 0.0),  # grounding, region_caption, image_caption, vqa, bbox_vqa
            max_categories=8,
            negative_sample_prob=0.3,  # 30%的概率添加负样本
            max_negative_categories=3,  # 最多3个负样本类别
            dataset_path='/mnt/disk/lyf/datasets/important/needed/labelme_train',  # 传递数据集路径
            use_random_prompts=True,  # 使用随机提示词
            prompt_language='mixed',  # 使用混合语言提示词
            vqa_mode='mixed',  # 混合VQA模式
            max_vqa_questions=3,  # 最多3个VQA问题
            max_bbox_vqa_boxes=3,  # 最多处理3个矩形框
            max_bbox_vqa_qa_per_box=2  # 每个矩形框最多2个问答对
        ),  
        load_function=load_labelme_folder,  # 使用自定义加载函数  
    ))

# 注册图像描述专用数据集（如果有单独的描述数据集）
register_dataset(  
    DatasetMeta(  
        dataset_path='/mnt/disk/lyf/datasets/1431_part1/filtered_descriptions',  
        dataset_name = "image_caption_dataset",
        preprocess_func=LabelMeGroundingPreprocessor(
            task_ratio=(0.0, 0.0, 1.0, 0.0, 0.0),  # 只使用image_caption任务
            max_categories=10,
            negative_sample_prob=0.0,  # 无负样本
            max_negative_categories=0,
            dataset_path='/mnt/disk/lyf/datasets/1431_part1/filtered_descriptions',
            use_random_prompts=True,
            prompt_language='mixed',
            vqa_mode='single',  # 默认VQA模式
            max_bbox_vqa_boxes=3,
            max_bbox_vqa_qa_per_box=2
        ),  
        load_function=load_labelme_folder,
    ))

# 注册VQA专用数据集（如果有单独的VQA数据集）
register_dataset(  
    DatasetMeta(  
        dataset_path='/mnt/disk/lyf/datasets/1431_part1/vqa_data/all',  # 假设VQA数据集的路径
        dataset_name = "vqa_dataset",
        preprocess_func=LabelMeGroundingPreprocessor(
            task_ratio=(0.0, 0.0, 0.0, 1.0, 0.0),  # 只使用VQA任务
            max_categories=10,
            negative_sample_prob=0.0,  # 无负样本
            max_negative_categories=0,
            dataset_path='/mnt/disk/lyf/datasets/1431_part1/vqa_data/all',
            use_random_prompts=True,
            prompt_language='mixed',
            vqa_mode='mixed',  # 混合VQA模式
            max_vqa_questions=5,  # 最多5个VQA问题
            max_bbox_vqa_boxes=3,
            max_bbox_vqa_qa_per_box=2
        ),  
        load_function=load_labelme_folder,
    ))

register_dataset(  
    DatasetMeta(  
        dataset_path='/mnt/disk/lyf/datasets/needed/vqa',  # 假设VQA数据集的路径
        dataset_name = "vqa_dataset2",
        preprocess_func=LabelMeGroundingPreprocessor(
            task_ratio=(0.0, 0.0, 0.0, 1.0, 0.0),  # 只使用VQA任务
            max_categories=10,
            negative_sample_prob=0.0,  # 无负样本
            max_negative_categories=0,
            dataset_path='/mnt/disk/lyf/datasets/needed/vqa',
            use_random_prompts=True,
            prompt_language='mixed',
            vqa_mode='mixed',  # 混合VQA模式
            max_vqa_questions=5,  # 最多5个VQA问题
            max_bbox_vqa_boxes=3,
            max_bbox_vqa_qa_per_box=2
        ),  
        load_function=load_labelme_folder,
    ))

register_dataset(  
    DatasetMeta(  
        dataset_path='/mnt/disk/lyf/datasets/1431_part1/qwen_objvqa/images',  # 假设VQA数据集的路径
        dataset_name = "objvqa_dataset",
        preprocess_func=LabelMeGroundingPreprocessor(
            task_ratio=(0.5, 0.0, 0.0, 0.0, 0.5),  # 只使用objVQA任务
            max_categories=10,
            negative_sample_prob=0.3,  # 无负样本
            max_negative_categories=3,
            dataset_path='/mnt/disk/lyf/datasets/1431_part1/qwen_objvqa/images',
            use_random_prompts=True,
            prompt_language='mixed',
            vqa_mode='mixed',  # 混合VQA模式
            max_vqa_questions=5,  # 最多5个VQA问题
            max_bbox_vqa_boxes=15,
            max_bbox_vqa_qa_per_box=2
        ),  
        load_function=load_labelme_folder,
    ))

register_dataset(  
    DatasetMeta(  
        dataset_path='/mnt/disk/lyf/datasets/coco/obj_vqa',  # 假设VQA数据集的路径
        dataset_name = "objvqa_dataset1",
        preprocess_func=LabelMeGroundingPreprocessor(
            task_ratio=(0.5, 0.0, 0.0, 0.0, 0.5),  # 只使用objVQA任务
            max_categories=10,
            negative_sample_prob=0.3,  # 无负样本
            max_negative_categories=3,
            dataset_path='/mnt/disk/lyf/datasets/coco/obj_vqa',
            use_random_prompts=True,
            prompt_language='mixed',
            vqa_mode='mixed',  # 混合VQA模式
            max_vqa_questions=5,  # 最多5个VQA问题
            max_bbox_vqa_boxes=15,
            max_bbox_vqa_qa_per_box=2
        ),  
        load_function=load_labelme_folder,
    ))

register_dataset(  
    DatasetMeta(  
        dataset_path='/mnt/disk/lyf/datasets/STAR/train/obj_vqa',  # 假设VQA数据集的路径
        dataset_name = "objvqa_dataset2",
        preprocess_func=LabelMeGroundingPreprocessor(
            task_ratio=(0.5, 0.0, 0.0, 0.0, 0.5),  # 只使用objVQA任务
            max_categories=10,
            negative_sample_prob=0.3,  # 无负样本
            max_negative_categories=3,
            dataset_path='/mnt/disk/lyf/datasets/STAR/train/obj_vqa',
            use_random_prompts=True,
            prompt_language='mixed',
            vqa_mode='mixed',  # 混合VQA模式
            max_vqa_questions=5,  # 最多5个VQA问题
            max_bbox_vqa_boxes=15,
            max_bbox_vqa_qa_per_box=2
        ),  
        load_function=load_labelme_folder,
    ))



########################
# swift sft \  
#     --model Qwen/Qwen2.5-VL-7B-Instruct \  
#     --dataset labelme_dataset1 labelme_dataset2 image_caption_dataset vqa_dataset \  
#     --custom_register_path labelme_preprocessor.py
########################