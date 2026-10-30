from concurrent.futures import Future, ThreadPoolExecutor
from time import sleep

import tiktoken
import torch
from torch.nn import functional as F

from gpt_2_from_scratch.train_gpt2 import GPTConfig
from train_gpt2 import GPT
from utils import get_device


def main() -> None:
    """Load a GPT-2 model and generate sample text from a fixed prompt."""
    num_return_sequences: int = 5
    max_length: int = 30

    # Start loading the model in another thread
    with ThreadPoolExecutor(max_workers=1) as executor:
        future: Future[GPT] = executor.submit(lambda: GPT(GPTConfig()))

        while not future.done():
            print('Model is loading...')
            sleep(1)

        # Retrieve the model
        model: GPT = future.result()

    print('Loaded successfully')

    model.eval()
    device: str = get_device()
    model.to(device)

    # prefix tokens
    enc: tiktoken.Encoding = tiktoken.get_encoding('gpt2')
    tokens: torch.Tensor = torch.tensor(
        enc.encode("Hello, I'm a language model,"), dtype=torch.long
    )  # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
    x: torch.Tensor = tokens.to(device)

    # generate; right now x is (B, T) where B = 5, T = 8
    # set the seed to 42
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            # (B, T, vocab_size)
            logits: torch.Tensor = model(x)

            # take the logits at the last position
            # (B, vocab_size)
            logits = logits[:, -1, :]

            # get the probabilities
            probs: torch.Tensor = F.softmax(logits, dim=-1)

            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs: torch.Tensor
            topk_indices: torch.Tensor
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

            # select a token from the top-k probabilities
            # (B, 1)
            ix: torch.Tensor = torch.multinomial(topk_probs, 1)

            # gather the corresponding indices
            xcol: torch.Tensor = torch.gather(topk_indices, -1, ix)

            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        token_ids: list[int] = x[i, :max_length].tolist()
        decoded: str = enc.decode(token_ids)
        print('>', decoded)


if __name__ == '__main__':
    main()
