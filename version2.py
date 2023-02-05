import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple, Dict
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
eval_iters = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-3
train_percentage = 0.8

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

char_set = sorted(list(set(text)))

chtoi = {ch: i for i, ch in enumerate(char_set)}
itoch = {i: ch for i, ch in enumerate(char_set)}

encode = lambda s: [chtoi[c] for c in s] # encoding function which maps chars to index
decode = lambda n: "".join([itoch[e] for e in n]) #decoding function which maps indexes to chars

data = torch.tensor(encode(text), dtype=torch.long)

dividing_index = int(data.shape[0] * train_percentage)
train_data = data[:dividing_index]
val_data = data[dividing_index:]

def get_data_batch(part: str) -> Tuple[torch.Tensor, torch.Tensor]:
    data = train_data if part == "train" else val_data
    indices = torch.randint(data.shape[0] - block_size, (batch_size,))
    x = torch.stack([data[i: (i + block_size)] for i in indices])
    x = x.to(device)
    y = torch.stack([data[(i+1): (i + block_size + 1)] for i in indices])
    y = y.to(device)

    return x, y

@torch.no_grad()
def estimate_loss(model: nn.Module) -> Dict:
    model.eval()
    out = {}

    for mode in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_data_batch(mode)
            logits, loss = model(X, Y)
            losses[i] = loss.item()

        out[mode] = losses.mean()

    model.train()
    return out

class Head(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int):
        super(Head, self).__init__()
        self.key = nn.Linear(embed_dim, head_dim)
        self.query = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))


    def forward(self, X):
        B, S, E = X.shape
        key = self.key(X) # (B, S, H)
        value = self.value(X) # (B, S, H)
        query = self.query(X) # (B, S, H)

        weight = query @ key.transpose(-2, -1) * E ** - 0.5 # (B, S, H) matmul (B, H, S) divided by sqrt of embedding dim
        weight = weight.masked_fill(self.tril[:S, :S] == 0, -float("inf"))
        # weight is (B, S, S)
        weight = F.softmax(weight, dim=-1)
        output = weight @ value # (B, S, S) @ (B, S, H) -> (B, S, H)

        return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([Head(embed_dim, head_dim) for _ in range(num_heads)])
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, X):
        output = torch.cat([layer(X) for layer in self.heads], -1)
        return self.projection(output)

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim,4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, X):
        return self.layers(X)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        assert embed_dim % num_heads == 0
        self.multi_head = MultiHeadSelfAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, X):
        X = X + self.multi_head(X)
        X = X + self.feed_forward(X)
        return X


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int):
        super(BigramLanguageModel, self).__init__()
        self.lookup = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(
            Block(embed_dim, num_heads),
            Block(embed_dim, num_heads),
            Block(embed_dim, num_heads)
        )
        self.projection = nn.Linear(embed_dim, vocab_size)


    def forward(self, input: torch.Tensor, target: torch.Tensor = None) -> Tuple[torch.Tensor, int]:
        B, S = input.shape
        logits = self.lookup(input)
        positions = self.position(torch.arange(S, device=device))
        x = logits + positions
        x = self.blocks(x)
        logits = self.projection(x)

        if target == None:
            loss = None
        else:
            B, S, E = logits.shape
            logits = logits.view(B * S, E)
            target = target.view(-1)

            loss = F.cross_entropy(logits, target)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_length: int = 200) -> torch.Tensor:

        for _ in range(max_length):
            # restriction on only last block_size inputs
            idx_crop = idx[:, -block_size:]
            #losses not used because it's none
            logits, loss = self(idx_crop)

            logits = logits[:,-1, :]
            probabilities = F.softmax(logits, dim=-1)

            next_input = torch.multinomial(probabilities, num_samples=1)

            idx = torch.cat((idx, next_input), dim=1)

        return idx


model = BigramLanguageModel(len(char_set), 32, 4)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"Step {i} train loss {losses['train']:.4f} val loss {losses['val']:.4f}")

    X, Y = get_data_batch("train")
    logits, loss = model(X, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#generation
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, 400)[0].tolist()))
