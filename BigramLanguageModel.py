import torch
import torch.nn.functional as F
import torch.nn as nn

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



