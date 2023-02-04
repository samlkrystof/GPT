import torch

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

char_set = sorted(list(set(text)))

chtoi = {ch: i for ch, i in enumerate(char_set)}
itoch = {i: ch for ch, i in enumerate(char_set)}

encode = lambda s: [chtoi[c] for c in s] # encoding function which maps chars to index
decode = lambda n: "".join([itoch[e] for e in n]) #decoding function which maps indexes to chars

data = torch.tensor(encode(text), dtype=torch.long)

train_percentage = 0.8
dividing_index = int(data.shape[0] * train_percentage)
train_data = data[:dividing_index]
val_data = data[dividing_index:]

block_size = 8
batch_size = 4

def get_data_batch(part: str) -> Tuple[torch.Tensor, torch.Tensor]:
    data = train_data if part == "train" else val_data
    indices = torch.randint(data.shape[0] - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in indices])
    y = torch.stack([data[i+1: i + block_size + 1] for i in indices])

    return x, y



