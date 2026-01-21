#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
"""  
测试 ms-swift 训练阶段的 tokenizer 编码  
"""  
  
import torch  
from swift.llm import get_model_tokenizer, get_template  
  
def test_training_tokenizer_encoding():  
    """测试训练阶段的 tokenizer 编码"""  
      
    model_path = "/mnt/data/lyf/IDEA-Research/Rex-Omni"  
      
    print("=" * 60)  
    print("使用 ms-swift 加载模型和 tokenizer (模拟训练流程)")  
    print("=" * 60)  
      
    # 使用 ms-swift 的方式加载,这会触发训练时的所有逻辑  
    model, processor = get_model_tokenizer(  
        model_path,  
        model_type='qwen2_5_vl',
        torch_dtype=torch.bfloat16,  
        load_model=True,  # 加载模型以触发 resize_token_embeddings  
        device_map='auto'  
    )  
      
    # 获取 tokenizer  
    if hasattr(processor, 'tokenizer'):  
        tokenizer = processor.tokenizer  
    else:  
        tokenizer = processor  
      
    print("\n" + "=" * 60)  
    print("训练阶段的词表信息")  
    print("=" * 60)  
    print(f"tokenizer 词表大小: {len(tokenizer)}")  
    print(f"model.config.vocab_size: {model.config.vocab_size}")  
    if hasattr(model, 'lm_head'):  
        print(f"lm_head 输出维度: {model.lm_head.weight.shape[0]}")  
      
    print("\n" + "=" * 60)  
    print("测试特殊字符编码 (训练阶段)")  
    print("=" * 60)  
      
    # 测试你的特殊 token  
    test_tokens = ["<0>", "<100>", "<500>", "<999>"]  
    expected_ids = {  
        "<0>": 150643,  
        "<100>": 150743,  
        "<500>": 151143,  
        "<999>": 151642  
    }  
      
    all_correct = True  
    for token in test_tokens:  
        actual_id = tokenizer.convert_tokens_to_ids(token)  
        expected_id = expected_ids.get(token, None)  
          
        if expected_id:  
            is_correct = actual_id == expected_id  
            all_correct = all_correct and is_correct  
            status = "✓" if is_correct else "✗"  
            print(f"  {status} '{token}': 期望 ID {expected_id}, 实际 ID {actual_id}")  
        else:  
            print(f"  '{token}': 实际 ID {actual_id}")  
      
    print(f"\n编码结果: {'全部正确 ✓' if all_correct else '存在偏移 ✗'}")  
      
    # 反向测试:从 ID 解码  
    print("\n" + "=" * 60)  
    print("测试 ID 解码 (训练阶段)")  
    print("=" * 60)  
      
    test_ids = [150643, 151143, 151642, 151643]  
    expected_tokens = {  
        150643: "<0>",  
        151143: "<500>",  
        151642: "<999>",  
        151643: "<|endoftext|>"  
    }  
      
    for token_id in test_ids:  
        decoded = tokenizer.decode([token_id])  
        expected = expected_tokens.get(token_id, "?")  
        is_correct = decoded == expected  
        status = "✓" if is_correct else "✗"  
        print(f"  {status} ID {token_id}: 期望 '{expected}', 实际 '{decoded}'")  
      
    # 测试实际的编码和解码流程  
    print("\n" + "=" * 60)  
    print("测试完整的编码-解码流程")  
    print("=" * 60)  
      
    # 加载 template 来测试训练时的编码  
    template = get_template(tokenizer.model_meta.template, tokenizer)  
    template.set_mode('train')  # 设置为训练模式  
      
    # 测试包含特殊 token 的文本  
    test_data = {  
        'messages': [  
            {'role': 'user', 'content': 'Test <0> and <999>'},  
            {'role': 'assistant', 'content': 'Response with <500>'}  
        ]  
    }  
      
    encoded = template.encode(test_data)  
    input_ids = encoded['input_ids']  
      
    print(f"编码后的 input_ids 长度: {len(input_ids)}")  
    print(f"input_ids 中是否包含特殊 token:")  
    for token, expected_id in expected_ids.items():  
        if expected_id in input_ids:  
            print(f"  ✓ 找到 '{token}' (ID: {expected_id})")  
        else:  
            # 检查是否是偏移后的 ID  
            offset_id = expected_id + 1000  
            if offset_id in input_ids:  
                print(f"  ✗ '{token}' 使用了偏移后的 ID {offset_id} (期望 {expected_id})")  
            else:  
                print(f"  ? '{token}' 未找到")  
      
    # 解码查看实际内容  
    decoded_text = template.safe_decode(input_ids)  
    print(f"\n解码后的文本:\n{decoded_text}")  
      
    return all_correct  
  
if __name__ == "__main__":  
    result = test_training_tokenizer_encoding()  
    print("\n" + "=" * 60)  
    if result:  
        print("✓ 训练阶段编码测试通过")  
    else:  
        print("✗ 训练阶段编码存在问题,请检查 config.json 中的 vocab_size")  
    print("=" * 60)