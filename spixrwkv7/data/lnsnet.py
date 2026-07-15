import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn


class LNSNetEmbedder(nn.Module):
    def __init__(self, is_dilation=True):
        super().__init__()
        self.is_dilation = is_dilation
        self.rpad_1 = nn.ReflectionPad2d(1)
        if self.is_dilation:
            self.c1_1 = nn.Conv2d(5, 10, 3, padding=0)
            self.c1_2 = nn.Conv2d(5, 10, 3, padding=0, dilation=1)
            self.c1_3 = nn.Conv2d(5, 10, 3, padding=1, dilation=2)
            self.c1_4 = nn.Sequential(
                nn.InstanceNorm2d(35, affine=True),
                nn.ReLU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(35, 10, 3, padding=0)
            )
        else:
            self.c1 = nn.Conv2d(5, 10, 3, padding=0)
        self.inorm_1 = nn.InstanceNorm2d(10, affine=True)
        self.rpad_2 = nn.ReflectionPad2d(1)
        self.c2 = nn.Conv2d(15, 20, 3, padding=0)
        self.inorm_2 = nn.InstanceNorm2d(20, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        spix = self.rpad_1(x)
        if self.is_dilation:
            spix_1 = self.c1_1(spix)
            spix_2 = self.c1_2(spix)
            spix_3 = self.c1_3(spix)
            spix = torch.cat([x, spix_1, spix_2, spix_3], dim=1)
            spix = self.c1_4(spix)
        else:
            spix = self.c1(spix)
        spix = self.inorm_1(spix)
        spix = self.relu(spix)
        spix = torch.cat((spix, x), dim=1)
        spix = self.rpad_2(spix)
        spix = self.c2(spix)
        spix = self.inorm_2(spix)
        spix = self.relu(spix)
        return spix

class LNSNetSeedGenerater(nn.Module):
    def __init__(self, n_spix, seed_strategy='network'):
        super().__init__()
        self.c3_inorm_1 = nn.InstanceNorm2d(20, affine=True)
        self.c3_seeds_1 = nn.Conv2d(20, 20, 3, padding=1)
        self.c3_seeds_2 = nn.Conv2d(20, 3, 1)
        self.relu = nn.ReLU()
        self.sp_num = n_spix
        self.seed_strategy = seed_strategy

    def _compute_grid_indices(self, h: int, w: int, device):
        """Compute grid-based superpixel seed indices and base coordinates."""
        S = h * w / self.sp_num
        sp_h = max(1, int(np.floor(np.sqrt(S) / (w / float(h)))))
        sp_w = max(1, int(np.floor(S / np.floor(sp_h))))
        sp_c = []
        for i in range(0, h, sp_h):
            for j in range(0, w, sp_w):
                end_x = min(i + sp_h, h) - 1
                end_y = min(j + sp_w, w) - 1
                x = (end_x + i) / 2.0
                y = (end_y + j) / 2.0
                ind = int(x) * w + int(y)
                sp_c.append(ind)
        sp_c = torch.tensor(sp_c, dtype=torch.long, device=device)
        o_cx = torch.floor(sp_c.float() / float(w))
        o_cy = sp_c.float() - o_cx * w
        return sp_h, sp_w, sp_c, o_cx, o_cy

    def seed_generate(self, spix):
        b, _, h, w = spix.size()
        sp_h, sp_w, sp_c, o_cx, o_cy = self._compute_grid_indices(h, w, spix.device)

        pooled = nn.AdaptiveAvgPool2d((int(np.ceil(h / sp_h)), int(np.ceil(w / sp_w))))(spix)
        pooled = self.c3_seeds_1(pooled)
        pooled = self.c3_inorm_1(pooled)
        pooled = self.relu(pooled)
        pooled = self.c3_seeds_2(pooled)

        prob = torch.sigmoid(pooled[:, 0].view(b, -1))
        dx = torch.sigmoid(pooled[:, 1].view(b, -1)) - 0.5
        dy = torch.sigmoid(pooled[:, 2].view(b, -1)) - 0.5

        dx_subset = dx[:, :len(sp_c)]
        dy_subset = dy[:, :len(sp_c)]
        cx = torch.floor(o_cx.unsqueeze(0) + dx_subset * sp_h * 2.0)
        cy = torch.floor(o_cy.unsqueeze(0) + dy_subset * sp_w * 2.0)

        cx = cx.clamp(0, h - 1)
        cy = cy.clamp(0, w - 1)
        return cx, cy, prob

    def grid_seed(self, spix):
        b, _, h, w = spix.size()
        _, _, sp_c, o_cx, o_cy = self._compute_grid_indices(h, w, spix.device)

        cx = o_cx.unsqueeze(0).expand(b, -1).clamp(0, h - 1)
        cy = o_cy.unsqueeze(0).expand(b, -1).clamp(0, w - 1)
        return cx, cy, torch.ones(b, h * w, device=spix.device, dtype=spix.dtype)

    def forward(self, x):
        if self.seed_strategy == 'network':
            cx, cy, probs = self.seed_generate(x)
        else:
            cx, cy, probs = self.grid_seed(x)
        return cx, cy, probs

class LNSNetGRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.recons = nn.Conv2d(20, 5, 1)

    def forward(self, f):
        return f, self.recons(f)

class LNSNet(nn.Module):
    def __init__(self, n_spix, seed_strategy='network', is_dilation=True):
        super().__init__()
        self.n_spix = n_spix
        self.embedder = LNSNetEmbedder(is_dilation)
        self.generater = LNSNetSeedGenerater(n_spix, seed_strategy=seed_strategy)
        self.grm = LNSNetGRM()

    def forward(self, x):
        f = self.embedder(x)
        cx, cy, probs = self.generater(f)
        b, c, h, w = f.shape
        f = f.view(b, c, h * w)
        return cx, cy, f, probs

def download_lnsnet_weights(check_path):
    url = "https://github.com/zh460045050/LNSNet/raw/main/lnsnet_BSDS_checkpoint.pth"
    os.makedirs(os.path.dirname(check_path), exist_ok=True)
    if not os.path.exists(check_path):
        print(f"Downloading LNSNet weights from {url} to {check_path}...")
        try:
            urllib.request.urlretrieve(url, check_path)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download LNSNet weights: {e}")

def lnsnet_assignment(f, input_5ch, cx, cy, alpha=1.0):
    b, _, h, w = input_5ch.size()
    cind = (cx * w + cy).long()

    c_f = torch.gather(f, 2, cind.unsqueeze(1).expand(-1, f.shape[1], -1))

    c_f_norm = torch.sum(c_f ** 2, dim=1, keepdim=True)
    f_norm = torch.sum(f ** 2, dim=1, keepdim=True)
    xy = torch.bmm(c_f.transpose(1, 2), f)
    dis = c_f_norm.transpose(1, 2) + f_norm - 2.0 * xy
    dis = torch.clamp(dis, min=0.0)

    dis = dis / alpha
    dis = torch.pow((1 + dis), -(alpha + 1) / 2)
    dis = dis.permute(0, 2, 1).contiguous()
    dis = dis / (torch.sum(dis, dim=2, keepdim=True) + 1e-8)

    dis = dis.permute(0, 2, 1).contiguous().view(b, -1, h, w)
    return dis
