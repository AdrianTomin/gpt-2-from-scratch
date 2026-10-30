from concurrent.futures import ThreadPoolExecutor
from time import sleep
import torch
import tiktoken
from torch.nn import functional as F

from train_gpt2 import GPT
from utils import get_device

def main() -> None:
    num_return_sequences = 5
    max_length = 30

    with ThreadPoolExecutor(max_workers=1) as executor:
        # Start loading the model in another thread
        future = executor.submit(GPT.from_pretrained, 'gpt2')

        while not future.done():
            print('Model is loading...')
            sleep(1)

        # Retrieve the model
        model: GPT = future.result()

    print('Loaded successfully')

    model.eval()
    device = get_device()
    model.to(device)

    # prefix tokens
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode('Hello, I\'m a language model,')
    tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
    x = tokens.to(device)

    # generate; right now x is (B, T) where B = 5, T = 8
    # set the seed to 42
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            # (B, T, vocab_size)
            logits = model(x)

            # take the logits at the last position
            # (B, vocab_size)
            logits = logits[:, -1, :]

            #get the probabilities
            probs = F.softmax(logits, dim=-1)

            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

            # select a token from the top-k probabilities
            # (B, 1)
            ix = torch.multinomial(topk_probs, 1)

            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)

            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print('>', decoded)


if __name__ == '__main__':
    main()
