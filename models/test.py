
from transformers import AutoTokenizer, LlamaForCausalLM

# model_id = "meta-llama/Llama-2-7b-chat-hf"
model_path = "llm/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    # token=hf_auth
)

model = LlamaForCausalLM.from_pretrained(
    model_path,
    # token=hf_auth
)

model = model.eval()

prompt = "今天你吃了吗，用中文回答我"

input = tokenizer(prompt, return_tensors="pt")

res = model.generate(input_ids=input.input_ids, max_length=30)

# 将生成的输出转换为文本
generated_text = tokenizer.decode(res[0], skip_special_tokens=True)

print('input', input)

print('res', res)

print('generated_text', generated_text)


# input {'input_ids': tensor([[    1, 29871, 31482, 30408, 30919,   232,   147,   134, 30743,   232,
#            147,   154, 30214, 30406, 30275, 30333, 30742,   234,   176,   151,
#          30672]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
# res tensor([[    1, 29871, 31482, 30408, 30919,   232,   147,   134, 30743,   232,
#            147,   154, 30214, 30406, 30275, 30333, 30742,   234,   176,   151,
#          30672, 30267,    13,    13, 29902,   263,   371, 17803,  1833,  4646]])
# generated_text 今天你吃了吗，用中文回答我。

# I ate dinner last night

# from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
# import torch

# # Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# # Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained('llm/text2vec-base-chinese')
# model = AutoModel.from_pretrained('llm/text2vec-base-chinese')
# sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
# # Tokenize sentences
# encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)
# # Perform pooling. In this case, mean pooling.
# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
# print("Sentence embeddings:")
# print(sentence_embeddings)

# Sentence embeddings:
# tensor([[-4.4386e-04, -2.9735e-01,  8.5790e-01,  ..., -5.2770e-01,
#          -1.4316e-01, -1.0008e-01],
#         [ 6.5362e-01, -7.6667e-02,  9.5962e-01,  ..., -6.0123e-01,
#          -1.6791e-03,  2.1458e-01]])