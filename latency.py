import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'  
import torch
import time

# =========================
PT_PATH = "./lsrw.pt"   
H, W = 256, 256         
EPOCHS = 100             
WARMUP = 10              
# =========================

@torch.no_grad()
def measure_speed(model, shape, epoch=100, warmup=10, device="cuda:0"):
    model = model.to(device).eval()
    dummy = torch.randn(shape, device=device)

    # warmup
    for _ in range(warmup):
        _ = model(dummy)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(epoch):
        _ = model(dummy)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.time()

    avg = (t1 - t0) / epoch
    fps = 1.0 / avg
    return avg, fps


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading TorchScript model from {PT_PATH} ...")
    model = torch.jit.load(PT_PATH, map_location=device)
    model.eval()

    shape = (1, 3, H, W)
    avg_time, fps = measure_speed(model, shape, epoch=EPOCHS, warmup=WARMUP, device=device)
    print(f"[SPEED] Input: {H}x{W}, avg time: {avg_time*1000:.2f} ms, FPS: {fps:.2f}")

    if device.startswith("cuda"):
        mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"[MEM] Peak memory usage: {mem:.1f} MB")


if __name__ == "__main__":
    main()
