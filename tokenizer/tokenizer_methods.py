from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "hello world"
token_ids = tokenizer.encode(text)
print(token_ids)
token_ids = tokenizer.encode_plus(text)
print(token_ids)
token_ids = tokenizer.batch_encode_plus([text])
print(token_ids)
token_ids = tokenizer([text])
print(token_ids)
token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
print(token_ids)