import json
from transformers import AutoModel, AutoConfig, DebertaV2Tokenizer

questions = json.load(open('../json/questions.json'))
mat_questions = json.load(open('../json/mat_questions.json'))

q_embeds = {}

config = AutoConfig.from_pretrained("microsoft/deberta-v3-large")
deberta = AutoModel.from_pretrained('microsoft/deberta-v3-large', config=config)

tokenizer_deberta = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')

for q in questions:

    deberta_tokens = tokenizer_deberta(questions[q]['question'], return_tensors='pt')
    deberta_embeddings = deberta(**deberta_tokens).last_hidden_state.squeeze()
    deberta_embeddings = deberta_embeddings[0, :]
    q_embeds[q] = deberta_embeddings.detach().numpy().tolist()

json.dump(q_embeds, open("../json/deberta_embeddings.json", 'w'))

matq_embeds = {}

for q in mat_questions:
    deberta_tokens = tokenizer_deberta(mat_questions[q], return_tensors='pt')
    deberta_embeddings = deberta(**deberta_tokens).last_hidden_state.squeeze()
    deberta_embeddings = deberta_embeddings[0, :]
    matq_embeds[q] = deberta_embeddings.detach().numpy().tolist()

json.dump(matq_embeds, open("../json/deberta_mat_embeddings.json", 'w'))