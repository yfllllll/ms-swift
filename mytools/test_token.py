from swift.llm import get_model_tokenizer  
  
# 加载模型和 tokenizer(只加载 tokenizer,不加载模型)  
model_dir = "/mnt/data/lyf/IDEA-Research/Rex-Omni"  # 替换为你的模型路径  
_, tokenizer = get_model_tokenizer(model_dir, load_model=False,model_type='qwen2_5_vl')  
  
# 定义你期望的特殊字符映射  
expected_special_tokens = {  
    150643: "<0>",  
    150644: "<1>",  
    151642: "<999>",  
    151664: "<|file_sep|>",

}  
  
print("=== 测试特殊字符编码 ===\n")  
  
# 测试 1: 检查 token ID 到字符串的映射  
print("1. 检查 token ID -> 字符串映射:")  
all_correct = True  
for token_id, expected_token in expected_special_tokens.items():  
    actual_token = tokenizer.decode([token_id])  
    is_correct = actual_token == expected_token  
    all_correct = all_correct and is_correct  
    status = "✓" if is_correct else "✗"  
    print(f"  {status} ID {token_id}: 期望 '{expected_token}', 实际 '{actual_token}'")  
  
print(f"\n结果: {'全部正确' if all_correct else '存在错误'}\n")  
  
# 测试 2: 检查字符串到 token ID 的映射  
print("2. 检查 字符串 -> token ID 映射:")  
all_correct = True  
for expected_id, token_str in expected_special_tokens.items():  
    actual_ids = tokenizer.encode(token_str, add_special_tokens=False)  
    is_correct = len(actual_ids) == 1 and actual_ids[0] == expected_id  
    all_correct = all_correct and is_correct  
    status = "✓" if is_correct else "✗"  
    print(f"  {status} '{token_str}': 期望 [{expected_id}], 实际 {actual_ids}")  
  
print(f"\n结果: {'全部正确' if all_correct else '存在错误'}\n")  
  
# 测试 3: 检查 tokenizer 的特殊 token 属性  
print("3. 检查 tokenizer 特殊 token 属性:")  
print(f"  eos_token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")  
print(f"  pad_token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")  
print(f"  bos_token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")  
  
# 测试 4: 检查 additional_special_tokens  
# print("\n4. 检查 additional_special_tokens:")  
# if hasattr(tokenizer, 'additional_special_tokens'):  
#     print(f"  数量: {len(tokenizer.additional_special_tokens)}")  
#     for token in tokenizer.additional_special_tokens:  
#         token_id = tokenizer.convert_tokens_to_ids(token)  
#         print(f"  - '{token}' (ID: {token_id})")


  
# 检查特定 token 的实际 ID  
token_str = "<999>"  
actual_id = tokenizer.convert_tokens_to_ids(token_str)  
print(f"'{token_str}' 的实际 ID: {actual_id}")  
  
# 反向验证:从 ID 解码回 token  
decoded_token = tokenizer.decode([151642])  
print(f"ID 151642 解码为: '{decoded_token}'")  
  
# 检查 added_tokens_decoder 是否被正确加载  
if hasattr(tokenizer, 'added_tokens_decoder'):  
    print(f"added_tokens_decoder 中的 151642: {tokenizer.added_tokens_decoder.get(151642)}")