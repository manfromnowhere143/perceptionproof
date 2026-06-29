"""Jointly-trained vision planner ensemble (P2h) — the GPU test the stability study motivated.

Fine-tunes K end-to-end planners (DINOv2-S backbone + ego, trained jointly — NOT frozen)
on WOD-E2E, each on a bootstrap of the training frames with a distinct seed, then predicts
trajectories for the 479 rater frames. Output: per-member rater trajectories, for RFS
scoring + member-bootstrap stability analysis in the `wod` env. Runs in the DLVM torch env.
"""
import os

import numpy as np
import timm
import torch
import torch.nn as nn

PIX = os.environ.get("PIX", os.path.expanduser("~/work/pixels.npz"))
OUTPRED = os.environ.get("OUTPRED", os.path.expanduser("~/work/vision_preds.npz"))
K = int(os.environ.get("K", "6"))
EPOCHS = int(os.environ.get("EPOCHS", "4"))
BS = int(os.environ.get("BS", "96"))
SMOKE = os.environ.get("SMOKE", "")
T = 20
DEV = "cuda"
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_t(imgs_u8):
    return ((torch.from_numpy(imgs_u8).float() / 255).permute(0, 3, 1, 2) - _MEAN) / _STD


class Planner(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                                     num_classes=0, img_size=224, dynamic_img_size=True)
        self.head = nn.Sequential(nn.Linear(384 + 12, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2 * T))

    def forward(self, img, ego):
        return self.head(torch.cat([self.bb(img), ego], 1))


def main():
    if SMOKE:
        print("SMOKE: build + 1 step on random data", flush=True)
        m = Planner().to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        img = torch.randn(8, 3, 224, 224, device=DEV)
        ego = torch.randn(8, 12, device=DEV)
        y = torch.randn(8, 2 * T, device=DEV)
        out = m(img, ego)
        loss = nn.MSELoss()(out, y)
        loss.backward()
        opt.step()
        print(f"SMOKE OK out={tuple(out.shape)} loss={loss.item():.3f}", flush=True)
        return

    d = np.load(PIX)
    train_img, train_ego, train_Y = d["train_img"], d["train_ego"], d["train_Y"]
    rater_img, rater_ego = d["rater_img"], d["rater_ego"]
    ego_mean, ego_sd = train_ego.mean(0), train_ego.std(0) + 1e-6
    egn_tr = torch.from_numpy((train_ego - ego_mean) / ego_sd).float()
    egn_r = torch.from_numpy((rater_ego - ego_mean) / ego_sd).float()
    Ytr = torch.from_numpy(train_Y).float()
    Rimg = to_t(rater_img)
    n = len(train_img)
    print(f"train={n} rater={len(rater_img)} K={K} epochs={EPOCHS} bs={BS}", flush=True)

    preds = np.zeros((K, len(rater_img), T, 2), np.float32)
    lossf = nn.MSELoss()
    for k in range(K):
        torch.manual_seed(1000 + k)
        rs = np.random.RandomState(1000 + k)
        idx = rs.randint(0, n, n)  # bootstrap
        model = Planner().to(DEV)
        opt = torch.optim.AdamW([{"params": model.bb.parameters(), "lr": 1e-5},
                                 {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
        model.train()
        for ep in range(EPOCHS):
            perm = rs.permutation(n)
            tot = 0.0
            for s in range(0, n, BS):
                b = idx[perm[s:s + BS]]
                img = to_t(train_img[b]).to(DEV)
                eg = egn_tr[b].to(DEV)
                y = Ytr[b].to(DEV)
                opt.zero_grad()
                loss = lossf(model(img, eg), y)
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            print(f"  member {k} epoch {ep} loss {tot / n:.4f}", flush=True)
        model.eval()
        with torch.no_grad():
            out = []
            for s in range(0, len(Rimg), 128):
                out.append(model(Rimg[s:s + 128].to(DEV), egn_r[s:s + 128].to(DEV)).cpu().numpy())
        preds[k] = np.concatenate(out, 0).reshape(-1, T, 2)
        print(f"  member {k} predicted", flush=True)

    np.savez(OUTPRED, preds=preds, rater_ego=rater_ego)
    print(f"WROTE {OUTPRED} preds={preds.shape}", flush=True)


if __name__ == "__main__":
    main()
