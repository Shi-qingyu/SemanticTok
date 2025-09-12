import numpy as np

from PIL import Image

import torch
import torch.nn.functional as F

from sklearn.decomposition import PCA


def pca_reduce_to_rgb(latents: torch.Tensor) -> torch.Tensor:
    """
    latents: [b, c, h, w] → PCA on channel dimension at each spatial location → [b, 3, h, w]
    Returns tensor in [0, 1], shape (b, 3, h, w).
    """
    b, c, h, w = latents.shape

    x_rgb = []
    for latent in latents:
        x = latent.permute(1, 2, 0).contiguous()  # (h, w, c)
        x_np = x.view(-1, c).cpu().numpy()       # (N, c)
        pca = PCA(n_components=3, svd_solver="full", random_state=0)
        x_np = pca.fit_transform(x_np)           # (N, 3)
        x_rgb.append(x_np)
        
    x_rgb = np.stack(x_rgb, axis=0)              # (b, N, 3)
    x_rgb = x_rgb.reshape(b, h, w, 3)
    x_rgb = x_rgb.transpose(0, 3, 1, 2)          # (b, 3, h, w)
    
    for x_np in x_rgb:
        for i in range(3):
            ch = x_np[i]
            ch = (ch - ch.min()) / (ch.max() - ch.min())
            x_np[i] = ch
    
    return torch.from_numpy(x_rgb).to(latents.device)


def to_image(arr_chw_01: torch.Tensor) -> list[Image.Image]:
    """
    arr_chw_01: (b, 3, h, w) in [0, 1]
    """
    imgs = []
    for img in arr_chw_01:
        img = (torch.clip(img, 0.0, 1.0) * 255.0).to(torch.uint8).permute(1, 2, 0)  # (h, w, 3)
        img = img.cpu().numpy()
        pil = Image.fromarray(img, mode="RGB")
        imgs.append(pil)
    return imgs