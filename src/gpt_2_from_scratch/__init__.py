from concurrent.futures import ThreadPoolExecutor
from time import sleep

from train_gpt2 import GPT

def main() -> None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Start loading the model in another thread
        future = executor.submit(GPT.from_pretrained, 'gpt2')

        while not future.done():
            print('Model is loading...')
            sleep(1)

        # Retrieve the model
        _model = future.result()

    print('Loaded successfully')


if __name__ == '__main__':
    main()
