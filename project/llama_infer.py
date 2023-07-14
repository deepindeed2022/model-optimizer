import torch
import torch.nn as nn
import os.path as osp
from transformers import AutoModelForCausalLM, AutoTokenizer

device = None
tokenizer = None
model = None
BOS_TOKEN = 0
PAD_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
DEBUG = False

def load_model(ckpt_dir = f"{osp.expanduser('~')}/models/llama-7b-hf"):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map="auto", torch_dtype=torch.float16)
    return model, tokenizer


@torch.inference_mode()
def generate_infer(model, text, batch_size=1, max_length=512):
    token_ids = tokenizer.encode(text)
    inputs = [token_ids] * batch_size
    inputs = [torch.tensor(toks, dtype=torch.int32, device=device) for toks in inputs]
    inputs = nn.utils.rnn.pad_sequence(inputs, True, padding_value=PAD_TOKEN)
    generated_ids = model.generate(inputs, max_length=max_length, 
                                   no_repeat_ngram_size=5, num_beams=5, do_sample=True)
    torch.cuda.synchronize()
    res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(res[0])
    
    # more sampling: https://huggingface.co/docs/transformers/generation_strategies
    # generated_ids = model.generate(inputs, max_length=max_length, 
    #                                no_repeat_ngram_size=5, num_beams=5, do_sample=False)
    # torch.cuda.synchronize()
    # res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # print(res[0])
    
def main():
    global model, tokenizer, device
    model, tokenizer = load_model()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    generate_infer(model, text="Hello, How are you")

if __name__ == "__main__":
    main()
    
    