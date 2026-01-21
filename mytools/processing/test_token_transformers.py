from transformers import AutoTokenizer  
from swift.llm import get_model_tokenizer  
  
model_dir = "/mnt/data/lyf/IDEA-Research/Rex-Omni"  # 替换为你的模型路径
  
# 1. 使用 transformers 直接加载  
print("=== 使用 transformers 加载 ===")  
tokenizer_hf = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True,model_type='qwen2_5_vl')  
print(f"词表大小: {len(tokenizer_hf)}")  
print(f"'<999>' 的 ID: {tokenizer_hf.convert_tokens_to_ids('<999>')}")  
print(f"ID 151642 解码为: '{tokenizer_hf.decode([151642])}'")  
  
# 2. 使用 ms-swift 加载  
print("\n=== 使用 ms-swift 加载 ===")  
_, tokenizer_swift = get_model_tokenizer(model_dir, load_model=False,model_type='qwen2_5_vl')  
print(f"词表大小: {len(tokenizer_swift.tokenizer)}")  
print(f"'<999>' 的 ID: {tokenizer_swift.tokenizer.convert_tokens_to_ids('<999>')}")  
print(f"ID 151642 解码为: '{tokenizer_swift.tokenizer.decode([151642])}'")  
  
# 3. 检查差异  
print(f"\n词表大小差异: {len(tokenizer_swift.tokenizer) - len(tokenizer_hf)}")
