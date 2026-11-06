import torch


def get_device() -> str:
    """Return the best available PyTorch device name.

    Prefers CUDA, then Apple MPS, and falls back to CPU.

    Returns:
        One of ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    device: str

    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {device}")
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print(f"Using device: {device}")
        return device
    device = "cpu"
    print(f"Using device: {device}")
    return device
