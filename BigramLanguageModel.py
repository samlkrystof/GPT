import torch
import torch.nn.functional as F
import torch.nn as nn

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
eval_iters = 100
device = "gpu" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-2
train_percentage = 0.8

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

char_set = sorted(list(set(text)))

chtoi = {ch: i for ch, i in enumerate(char_set)}
itoch = {i: ch for ch, i in enumerate(char_set)}

encode = lambda s: [chtoi[c] for c in s] # encoding function which maps chars to index
decode = lambda n: "".join([itoch[e] for e in n]) #decoding function which maps indexes to chars

data = torch.tensor(encode(text), dtype=torch.long)

dividing_index = int(data.shape[0] * train_percentage)
train_data = data[:dividing_index]
val_data = data[dividing_index:]

def get_data_batch(part: str) -> Tuple[torch.Tensor, torch.Tensor]:
    data = train_data if part == "train" else val_data
    indices = torch.randint(data.shape[0] - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in indices])
    y = torch.stack([data[i+1: i + block_size + 1] for i in indices])

    return x, y


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        self.lookup = nn.Embedding(vocab_size, vocab_size)


    def forward(self, input: torch.Tensor, target: torch.Tensor = None) -> Tuple[torch.Tensor, int]:
        logits = self.lookup(input)

        if target == None:
            loss = None
        else:
            B, S, E = logits.shape
            logits = logits.view(B * S, E)
            target = target.view(-1)

            loss = F.cross_entropy(logits, target)

        return logits, loss

    @torch.no_grad()
    def generate(self, input: torch.Tensor, max_length: int = 200) -> torch.Tensor:

        for _ in range(max_length):
            #loss not used because it's none
            logits, loss = self(input)

            logits = logits[:,-1, :]
            probabilities = logits.softmax(logits, dim=1)

            next_input = torch.multinomial(probabilities, num_samples=1)

            input = torch.cat((input, next_input), dim=1)

        return input



