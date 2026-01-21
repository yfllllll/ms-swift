#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
"""  
验证 ms-swift 加载时是否导致 token ID 偏移  
"""  
  
import torch  
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  
from swift.llm import get_model_tokenizer, MODEL_MAPPING  
from swift.llm.model.register import _get_model_info, get_model_tokenizer_from_local, get_matched_model_meta  
  
model_path = "/mnt/data/lyf/IDEA-Research/Rex-Omni"  
  
print("=" * 60)  
print("方法 1: 使用 ms-swift 的 get_model_tokenizer()")  
print("=" * 60)  
  
model_swift, processor_swift = get_model_tokenizer(  
    model_path,  
    model_type='qwen2_5_vl',  
    torch_dtype=torch.bfloat16,  
    load_model=True,  
    device_map='auto'  
)  
  
tokenizer_swift = processor_swift.tokenizer if hasattr(processor_swift, 'tokenizer') else processor_swift  
print(f"词表大小: {len(tokenizer_swift)}")  
print(f"'<999>' 的 ID: {tokenizer_swift.convert_tokens_to_ids('<999>')}")  
print(f"ID 151642 解码为: '{tokenizer_swift.decode([151642])}'")  
  
print("\n" + "=" * 60)  
print("方法 2: 直接使用 get_model_tokenizer_from_local()")  
print("=" * 60)  
  
# 1. 获取 model_meta  
model_meta = get_matched_model_meta(model_path)  
if model_meta is None:  
    model_meta = MODEL_MAPPING.get('qwen2_5_vl')  
  
# 2. 创建 ModelInfo  
model_info = _get_model_info(  
    model_dir=model_path,  
    model_type='qwen2_5_vl',  
    quantization_config=None  
)  
model_info.torch_dtype = torch.bfloat16  
  
# 3. 加载 processor  
processor_local = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)  
  
# 4. 调用 get_model_tokenizer_from_local,传入正确的 automodel_class  
model_local, _ = get_model_tokenizer_from_local(  
    model_path,  
    model_info,  
    {'device_map': 'auto', 'torch_dtype': torch.bfloat16},  
    load_model=True,  
    tokenizer=processor_local.tokenizer,  
    model_meta=model_meta,  
    automodel_class=Qwen2_5_VLForConditionalGeneration  # 添加这个参数  
)  
  
tokenizer_local = processor_local.tokenizer  
print(f"词表大小: {len(tokenizer_local)}")  
print(f"'<999>' 的 ID: {tokenizer_local.convert_tokens_to_ids('<999>')}")  
print(f"ID 151642 解码为: '{tokenizer_local.decode([151642])}'")  
  
print("\n" + "=" * 60)  
print("方法 3: 只加载 tokenizer,不加载模型")  
print("=" * 60)  
  
processor_only = AutoProcessor.from_pretrained(model_path, trust_remote_code=True,use_fast=False)  
tokenizer_only = processor_only.tokenizer  
print(f"词表大小: {len(tokenizer_only)}")  
print(f"'<999>' 的 ID: {tokenizer_only.convert_tokens_to_ids('<999>')}")  
print(f"ID 151642 解码为: '{tokenizer_only.decode([151642])}'")  
  
print("\n" + "=" * 60)  
print("结论")  
print("=" * 60)  
print("如果方法 1 和方法 2 都显示偏移,而方法 3 正确,")  
print("说明问题出在加载模型时的 resize_token_embeddings() 调用")  
print("解决方案: 修改 config.json 中的 vocab_size 为 152665")
