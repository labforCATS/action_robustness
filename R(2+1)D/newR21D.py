import torch
import decord
import numpy as np
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

def load_frames_pytorch(video_path, num_frames=16, fps=30):
    decord.bridge.set_bridge("torch")  # Return tensors instead of numpy arrays
    vr = decord.VideoReader(video_path)

    total_frames = len(vr)
    if total_frames < num_frames:
        raise ValueError(f"Only {total_frames} frames available; {num_frames} required.")

    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    clip = vr.get_batch(indices)  # [T, H, W, C], dtype=torch.uint8

    # Convert to [T, C, H, W] and float32
    clip = clip.permute(0, 3, 1, 2).float() / 255.0
    return clip  # [T, C, H, W]


def R2PLUS1D(video_path):
    weights = R2Plus1D_18_Weights.KINETICS400_V1
    model = r2plus1d_18(weights=weights).eval()
    transform = weights.transforms()
    class_names = weights.meta["categories"]

    clip = load_frames_pytorch(video_path)
    clip = transform(clip)
    clip = clip.unsqueeze(0)

    with torch.no_grad():
        out = model(clip)
        probs = torch.nn.functional.softmax(out, dim=1)

    top5 = torch.topk(probs, 5).indices.squeeze().tolist()
    results= [class_names[i] for i in top5]
    return ", ".join(results)
