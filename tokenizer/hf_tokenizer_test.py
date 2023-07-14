import torch
import os.path as osp
from transformers import LlamaTokenizer, BertTokenizer, BertTokenizerFast, GPT2Tokenizer, AlbertTokenizer

llama_ckpt_dir = "/home/wenlong.cao/models/llama-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(llama_ckpt_dir)
token_ids = tokenizer("You are an AI assistant named SearchGPT, developed by Shopee Search Team. You are designed to be helpful, honest, and harmless. <Human>: What happens when you mix oil and water? <AI>: ")["input_ids"]
print(token_ids)
output_text = tokenizer.decode(token_ids)
print(output_text)
# 1,   887,   526,   385,   319, 29902, 20255,  4257, 11856, 29954,
#          7982, 29892,  8906,   491, 17550,   412, 29872, 11856,  8583, 29889,
#           887,   526,  8688,   304,   367,  8444, 29892, 15993, 29892,   322,
#          4023,   828,   404, 29889,   529, 29950,  7889, 23917,  1724,  5930,
#           746,   366,  6837, 17182,   322,  4094, 29973,   529, 23869, 23917
          
# result = "29896 29897 2448 2121 17182 3643 4094 526 899 431 280 297 1269 916 29892 577 896 674 451 6837 29889 13 29906 29897 960 366 6837 963 4208 29892 896 674 883 5004 15359 29892 411 278 17182 373 2246 1363 372 338 3109 20619 1135 4094 29889 13 29941 29897 960 366 528 1296 278 29544 29892 278 1023 15359 674 14405 322 1653 385 953 25381 29889 29871 2"
# result = "1,887,526,385,319,29902,20255,4257,11856,29954,7982,29892,8906,491,17550,412,29872,11856,8583,29889,887,526,8688,304,367,8444,29892,15993,29892,322,4023,828,404,29889,529,29950,7889,23917,1724,5930,746,366,6837,17182,322,4094,29973,529,23869,23917,29871"
# result = "529,23869,23917,29871"
# result = list(map(int, result.split(",")))
# # print(" ".join(map(str, result)))
# output_text = tokenizer.decode(result, skip_special_tokens=True)
# print(output_text)

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2", is_split_into_words=True)
# token_ids = tokenizer("Hello, This is Shopee MLP AIP Group, ahaha! I'm a students")["input_ids"]
# # token_ids = [15496, 11, 770, 318, 13705, 1453, 10373, 47, 317, 4061, 4912, 11, 29042, 12236, 0]
# print(token_ids)
# output_text = tokenizer.decode(token_ids)
# print(output_text)

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# token_ids = tokenizer("Hello, This is Shopee MLP AIP Group, ahaha! I'm a students")["input_ids"]
# print(token_ids)
# output_text = tokenizer.decode(token_ids)
# print(output_text)
