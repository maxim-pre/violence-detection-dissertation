import torch

def get_available_device(min_memory_gb=4):
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_gpu = None
    most_free_gb = 0

    for gpu_id in range(torch.cuda.device_count()):
        free_memory, _ = torch.cuda.mem_get_info(gpu_id)
        free_gb = free_memory / (1024 ** 3)

        if free_gb > most_free_gb:
            most_free_gb = free_gb
            best_gpu = gpu_id

    if most_free_gb < min_memory_gb:
        raise RuntimeError(f"No GPU has enough free memory. Best has {most_free_gb:.2f} GB")

    print(f"Using cuda:{best_gpu} with {most_free_gb:.2f} GB free")
    return torch.device(f"cuda:{best_gpu}")