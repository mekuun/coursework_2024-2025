import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Устройство:', device)

df = pd.read_csv('files2.csv')
df = df[df['year'] <= 2024]

vect = CountVectorizer(lowercase=True,
                       stop_words='english',
                       token_pattern=r'\b[a-zA-Z]{3,}\b',
                       min_df=5, max_df=0.5)
X = vect.fit_transform(df['title'])
vocab = vect.get_feature_names_out()
print('Словарь:', len(vocab))

def load_glove(path, vocab, dim=300):
    glove, hits = {}, 0
    with open(path, encoding='utf-8') as f:
        for ln in f:
            w, *vec = ln.split()
            glove[w] = np.asarray(vec, np.float32)
    mat = np.random.normal(0, .01, (len(vocab), dim))
    for i, w in enumerate(vocab):
        if w in glove:
            mat[i] = glove[w]
            hits += 1
    print(f'GloVe найдено {hits}/{len(vocab)} слов')
    return torch.tensor(mat, dtype=torch.float32), set(glove.keys())

emb, glove_vocab = load_glove('glove.6B.300d.txt', vocab)
emb = emb.to(device)
X_tt = torch.tensor(X.toarray(), dtype=torch.float32, device=device)

class ETM(nn.Module):
    def __init__(self, n_topics, emb_dim, n_docs, pretrained):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(n_docs, n_topics) * 0.01)
        self.beta = nn.Parameter(torch.randn(n_topics, emb_dim) * 0.01)
        self.emb = pretrained

    def forward(self, _):
        theta = torch.softmax(self.theta, 1)
        beta = torch.softmax(self.beta @ self.emb.T, 1)
        return torch.log(theta @ beta + 1e-12)

n_topics = 25
model = ETM(n_topics, 300, X.shape[0], emb).to(device)
opt = optim.Adam(model.parameters(), 1e-2)

for ep in range(300):
    opt.zero_grad()
    loss = -(X_tt * model(X_tt)).sum()
    loss.backward(); opt.step()
    if (ep + 1) % 50 == 0:
        print(f'Эпоха {ep+1}: {loss.item():.1f}')

with torch.no_grad():
    doc_topics = torch.softmax(model.theta, 1).cpu().numpy()
df['topic'] = doc_topics.argmax(1)

glove_words = set(vocab) & glove_vocab
doc_words = []
for row in X:
    idxs = row.nonzero()[1]
    words_in_doc = [vocab[i] for i in idxs if vocab[i] in glove_words]
    doc_words.append(words_in_doc)
df['used_words'] = doc_words

df.to_csv('file2_with_topics.csv', index=False)

with torch.no_grad():
    beta = torch.softmax(model.beta @ emb.T, 1).detach().cpu().numpy()
words = np.array(vocab)
top_words = []
for i in range(n_topics):
    top = words[beta[i].argsort()[::-1][:10]]
    top_words.append({'topic_id': i, 'words': list(top)})
pd.DataFrame(top_words).to_csv('all_topics.csv', index=False)
