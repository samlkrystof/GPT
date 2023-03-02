import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple, Dict

batch_size = 64
block_size = 256

max_iters = 5000
eval_interval = 300
eval_iters = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-3
train_percentage = 0.8
dropout = 0.2
num_blocks = 6
num_heads = 8
embed_dim = 512

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

char_set = sorted(list(set(text)))

chtoi = {ch: i for i, ch in enumerate(char_set)}
itoch = {i: ch for i, ch in enumerate(char_set)}

encode = lambda s: [chtoi[c] for c in s]  # encoding function which maps chars to index
decode = lambda n: "".join([itoch[e] for e in n])  # decoding function which maps indexes to chars

data = torch.tensor(encode(text), dtype=torch.long)

dividing_index = int(data.shape[0] * train_percentage)
train_data = data[:dividing_index]
val_data = data[dividing_index:]


def get_data_batch(part: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Get a batch of data from the training or validation set
        :param part: "train" or "val" to indicate which set to get data from
        :return: a tuple of (input, target) tensors
        """
    data = train_data if part == "train" else val_data
    indices = torch.randint(data.shape[0] - block_size, (batch_size,))
    x = torch.stack([data[i: (i + block_size)] for i in indices])
    x = x.to(device)
    y = torch.stack([data[(i + 1): (i + block_size + 1)] for i in indices])
    y = y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(model: nn.Module) -> Dict:
    """
        Estimate the loss on the training and validation sets
        :param model: the model to evaluate
        :return: a dictionary with keys "train" and "val" containing the estimated loss on each set
        """
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


class MultiHeadSelfAttention(nn.Module):

    """
    Multi-head self-attention layer
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super(MultiHeadSelfAttention, self).__init__()
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: input tensor of shape (batch_size, block_size, embed_dim)
        :return: output tensor of shape (batch_size, block_size, embed_dim)
        """
        B, S, E = X.shape

        key, query, value = [layer(X).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
                             for layer in [self.key, self.query, self.value]]

        res = (query @ key.transpose(-1, -2)) / E ** -0.5
        res = torch.masked_fill(res, self.tril[:S, :S] == 0, -float("inf"))
        res = F.softmax(res, -1)
        res = self.dropout(res)
        res = res @ value

        res = res.transpose(1, 2).contiguous().view(B, S, E)
        output = self.projection(res)

        return output


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.layers(X))


class Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super(Block, self).__init__()
        assert embed_dim % num_heads == 0
        self.multi_head = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, dropout)
        self.lnorm1 = nn.LayerNorm(embed_dim)
        self.lnorm2 = nn.LayerNorm(embed_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X + self.multi_head(self.lnorm1(X))
        X = X + self.feed_forward(self.lnorm2(X))
        return X


class GPT(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, dropout: float, num_blocks: int):
        super(GPT, self).__init__()
        self.lookup = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, dropout) for _ in range(num_blocks)])

        self.lnorm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, vocab_size)

    def forward(self, input: torch.Tensor, target: torch.Tensor = None) -> Tuple[torch.Tensor, int]:
        B, S = input.shape
        logits = self.lookup(input)
        positions = self.position(torch.arange(S, device=device))
        x = logits + positions
        x = self.blocks(x)
        x = self.lnorm(x)
        logits = self.projection(x)

        if target == None:
            loss = None
        else:
            B, S, E = logits.shape
            logits = logits.view(B * S, E)
            target = target.view(-1)

            loss = F.cross_entropy(logits, target)

        return logits, loss

    @torch.inference_mode()
    def generate(self, idx: torch.Tensor, max_length: int = 200) -> torch.Tensor:
        model.eval()

        for _ in range(max_length):
            # restriction on only last block_size inputs
            idx_crop = idx[:, -block_size:]
            # losses not used because it's none
            logits, loss = self(idx_crop)

            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)

            next_input = torch.multinomial(probabilities, num_samples=1)

            idx = torch.cat((idx, next_input), dim=1)
        model.train()
        return idx


model = GPT(len(char_set), embed_dim, num_heads, dropout, num_blocks)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training
for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"Step {i} train loss {losses['train']:.4f} val loss {losses['val']:.4f}")

    X, Y = get_data_batch("train")
    logits, loss = model(X, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# generation
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, 400)[0].tolist()))

# write output to file
with open("output.txt", "w") as f:
    f.write(decode(model.generate(context, 10000)[0].tolist()))
