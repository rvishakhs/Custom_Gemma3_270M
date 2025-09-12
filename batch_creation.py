import os 
import numpy as np
from tqdm.auto import tqdm
import torch


# we need to check if the cuda available or not 

cuda_available = torch.cuda.is_available()
if cuda_available:
    print("✅ CUDA is available.")
    # You can also get more details
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("❌ CUDA is not available.")


def get_batch(split, batch_size=32, block_size=128):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    else: 
        data = np.memmap('val.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size))
    #x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    #y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    x = torch.stack([torch.from_numpy((data[i.item():i.item()+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i.item()+1:i.item()+1+block_size]).astype(np.int64)) for i in ix])



    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y

def get_batch_optimized(split, batch_size=32, block_size=128):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = np.memmap(f'{split}.bin', dtype=np.uint16, mode='r')
    
    # 1. Generate random starting indices
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # 2. Create the input (x) and target (y) batches without a Python loop
    # This is much faster as it avoids list comprehension
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    
    # 3. Move to the target device
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        
    return x, y


if __name__ == "__main__":
    get_batch("train")