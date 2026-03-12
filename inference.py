import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

# ----------------- IO -----------------
def load_image(path):
    """Load image as torch tensor [1, 3, H, W] normalized to [0,1]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor

# ----------------- Metrics (SSIM / PSNR) -----------------
def _ssim_single_channel(img1, img2):
    # img1, img2: [H, W], float64, range [0,255]
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()

def calculate_ssim(pred, gt):
    """Calculate SSIM between two RGB or grayscale images (range [0,255])."""
    img1 = np.array(pred, dtype=np.float64)
    img2 = np.array(gt, dtype=np.float64)
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return _ssim_single_channel(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            return np.mean([_ssim_single_channel(img1[:, :, i], img2[:, :, i]) for i in range(3)])
        elif img1.shape[2] == 1:
            return _ssim_single_channel(img1[:, :, 0], img2[:, :, 0])
        else:
            raise ValueError("Unsupported channel count.")
    else:
        raise ValueError("Wrong input image dimensions.")

def calculate_psnr(pred, gt):
    """Calculate PSNR between two images (range [0,255])."""
    img1 = np.array(pred, dtype=np.float32)
    img2 = np.array(gt, dtype=np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 * 255.0 / (mse + 1e-8))

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="TorchScript Inference + PSNR/SSIM (no saving)")
    ap.add_argument("--model", type=str, default="lsrw_huawei.pt", help="TorchScript model path (.pt)")
    ap.add_argument("--input", type=str, required=True, help="Input folder containing images to process")
    ap.add_argument("--gt", type=str, required=True, help="Ground truth folder with matching filenames")
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                    help="Device, e.g. 'cuda:0', 'cuda:1', or 'cpu'")
    ap.add_argument("--exts", type=str, default=".jpg,.jpeg,.png,.bmp,.tif,.tiff",
                    help="File extensions to include (comma separated)")
    ap.add_argument("--fp16", action="store_true", help="Use autocast mixed precision on CUDA")
    args = ap.parse_args()

    # ===== Device check =====
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available, but a CUDA device was requested.")
    if args.device.startswith("cuda:"):
        gpu_id = int(args.device.split(":")[1])
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid GPU id {gpu_id}, only {torch.cuda.device_count()} GPUs available.")

    # ===== Paths =====
    input_dir = Path(args.input)
    gt_dir = Path(args.gt)

    # ===== Load model =====
    print(f"[INFO] Loading model: {args.model}")
    model = torch.jit.load(args.model, map_location=args.device).to(args.device).eval()
    print(f"[OK] Model loaded successfully on {args.device}")

    # ===== Collect images =====
    exts = {e.strip().lower() for e in args.exts.split(",")}
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    if not files:
        print(f"[WARN] No images found in: {input_dir}")
        return
    print(f"[INFO] Found {len(files)} image(s), starting inference and metric evaluation...")

    # ===== Inference + Metrics (no saving) =====
    avg_psnr, avg_ssim, n, skipped = 0.0, 0.0, 0, 0

    dtype_autocast = (args.fp16 and args.device.startswith("cuda"))
    autocast_cm = torch.cuda.amp.autocast if dtype_autocast else torch.cpu.amp.autocast

    with torch.inference_mode():
        try:
            cm = autocast_cm()
        except Exception:
            class _NullCM:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            cm = _NullCM()

        for i, img_path in enumerate(sorted(files), start=1):
            gt_path = gt_dir / img_path.name
            if not gt_path.exists():
                print(f"[SKIP] Ground truth not found for: {img_path.name}")
                skipped += 1
                continue

            # Load input and GT
            x = load_image(img_path).to(args.device)  # [1,3,H,W], 0~1
            gt_img = Image.open(gt_path).convert("RGB")
            gt_np = np.array(gt_img).astype(np.float32)  # [H,W,3], 0~255

            # Inference
            with cm:
                y = model(x)  # expected [1,3,H,W], 0~1
            if isinstance(y, (list, tuple)):
                y = y[0]
            y = y.clamp(0, 1)

            # Resize to match GT size if needed
            _, _, h, w = y.shape
            gt_w, gt_h = gt_img.size
            if (w != gt_w) or (h != gt_h):
                y = F.interpolate(y, size=(gt_h, gt_w), mode="bilinear", align_corners=False)

            # Convert to numpy [0,255]
            y_np = (y.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0)

            # Compute metrics
            psnr = calculate_psnr(y_np, gt_np)
            ssim = calculate_ssim(y_np, gt_np)
            avg_psnr += psnr
            avg_ssim += ssim
            n += 1

            print(f"[{i}/{len(files)}] {img_path.name}  PSNR: {psnr:.4f} dB  SSIM: {ssim:.4f}")

    if n == 0:
        print("[WARN] No valid image pairs for evaluation.")
        return

    avg_psnr /= n
    avg_ssim /= n
    print("\n========== Evaluation Results (Inference + PSNR / SSIM) ==========")
    print(f"Valid images: {n}  |  Skipped: {skipped}")
    print(f"Avg.PSNR: {avg_psnr:.4f} dB")
    print(f"Avg.SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    main()
