# test_labelme_debug.py  
import os  

# os.environ["MAX_PIXELS"] = "1003520"  
os.environ["QWENVL_BBOX_FORMAT"] = "legacy"  # ~H~V 'new'  
                                                                                                                                                                                                                                                           
from swift.llm import get_model_tokenizer, get_template
import json
                                                                                                                                                                                                                                                           
# 1. 加载您的自定义注册   
import sys
sys.path.insert(0, '.')
import labelme_preprocessor  # 这会执行注册  
                                                                                                                                                                                                                                                           
# 2. 加载模型和模板(不加载权重)  
_, tokenizer = get_model_tokenizer('Qwen/Qwen2.5-VL-7B-Instruct', load_model=False)
template = get_template(tokenizer.model_meta.template, tokenizer)
                                                                                                                                                                                                                                                           
# 3. 准备测试数据 
json_file = '/mnt/data/lyf/datasets/1431_part1/labelme/202511010300020077-e93f806a07eb4bebb6e6d043232b3461.json'
with open(json_file, 'r') as f:
    labelme_data = json.load(f)
    labelme_data['imagePath'] = os.path.join(os.path.dirname(json_file), labelme_data['imagePath'])

# 4. 测试预处理器  
from labelme_preprocessor import LabelMeGroundingPreprocessor

preprocessor = LabelMeGroundingPreprocessor(task_ratio=(1.0, 0.0, 0.0, 0.0), max_categories=3, negative_sample_prob=0.8, dataset_path='/mnt/data/lyf/datasets/1431_part1/labelme')
preprocessor.all_categories = [cat['label'] for cat in labelme_data['shapes']]
processed_data = preprocessor.preprocess(labelme_data)

print("=== Preprocessed data ===")  
print(f"Query: {processed_data['messages'][0]['content']}")
print(f"Response: {processed_data['messages'][1]['content']}")
print(f"Objects: {processed_data['objects']}")
                                                                                                                                                                                                                                                           
# 5. 测试模板编码  
template.set_mode('train')
encoded = template.encode(processed_data, return_template_inputs=True)
                                                                                                                                                                                                                                                           
print("\n=== Encoded result ===") 
print(f"[INPUT_IDS] {template.safe_decode(encoded['input_ids'])}\n")
print(f"[LABELS] {template.safe_decode(encoded['labels'])}")
print(f"Images: {encoded['template_inputs'].images}")