#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
"""  
测试 Rex-Omni 模型的词表大小和预测头大小  
"""  
  
import torch  
from transformers import AutoProcessor, AutoTokenizer,Qwen2_5_VLForConditionalGeneration  
  
def test_rex_omni_vocab_and_head():  
    """测试 Rex-Omni 的词表大小和预测头大小"""  
      
    model_path = "/mnt/data/lyf/IDEA-Research/Rex-Omni"
      
    print("=" * 60)  
    print("加载 Rex-Omni 模型...")  
    print("=" * 60)  
      
    # 加载 processor (包含 tokenizer)  
    processor = AutoProcessor.from_pretrained(  
        model_path,  
        use_fast=False  
    )  
    tokenizer_hf = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,model_type='qwen2_5_vl')  
    # 加载模型  
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(  
        model_path,  
        torch_dtype=torch.bfloat16,  
        device_map="auto"  
    )  
      
    print("\n" + "=" * 60)  
    print("词表信息")  
    print("=" * 60)  
      
    # 获取词表大小  
    vocab_size = len(processor.tokenizer)  
    print(f"词表大小 (tokenizer): {vocab_size}")  
      
    # 从模型配置获取词表大小  
    config_vocab_size = model.config.vocab_size  
    print(f"词表大小 (model.config): {config_vocab_size}")  
      
    print("\n" + "=" * 60)  
    print("预测头 (lm_head) 信息")  
    print("=" * 60)  
      
    # 获取 lm_head 的参数  
    lm_head = model.lm_head  
    print(f"lm_head 类型: {type(lm_head)}")  
    print(f"lm_head 权重形状: {lm_head.weight.shape}")  
    print(f"lm_head 输出维度 (词表大小): {lm_head.weight.shape[0]}")  
    print(f"lm_head 输入维度 (hidden_size): {lm_head.weight.shape[1]}")  
      
    # 计算参数量  
    lm_head_params = lm_head.weight.numel()  
    print(f"lm_head 参数量: {lm_head_params:,}")  
    print(f"lm_head 参数量 (MB): {lm_head_params * 2 / 1024 / 1024:.2f} MB (bfloat16)")  
      
    print("\n" + "=" * 60)  
    print("模型整体信息")  
    print("=" * 60)  
      
    # 计算总参数量  
    total_params = sum(p.numel() for p in model.parameters())  
    print(f"模型总参数量: {total_params:,}")  
    print(f"模型总参数量 (B): {total_params / 1e9:.2f}B")  
      
    # lm_head 占比  
    lm_head_ratio = (lm_head_params / total_params) * 100  
    print(f"lm_head 参数占比: {lm_head_ratio:.2f}%")  
      
    print("\n" + "=" * 60)  
    print("特殊 Token 测试")  
    print("=" * 60)  
      
    # 测试坐标 bin tokens  
    test_tokens = ["<0>", "<500>", "<999>"]  
    for token in test_tokens:  
        token_id = processor.tokenizer.convert_tokens_to_ids(token)  
        print(f"'{token}' 的 ID by processor: {token_id}")  
        token_id = tokenizer_hf.convert_tokens_to_ids(token)  
        print(f"'{token}' 的 ID by tokenizer: {token_id}")  
      
    # 测试解码  
    test_ids = [151642, 151643]  
    for token_id in test_ids:  
        decoded = processor.tokenizer.decode([token_id])  
        print(f"ID {token_id} by processor 解码为: '{decoded}'")  
        decoded = tokenizer_hf.decode([token_id])  
        print(f"ID {token_id} by tokenizer 解码为: '{decoded}'")  
  
if __name__ == "__main__":  
    test_rex_omni_vocab_and_head()