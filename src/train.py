"""Training loops for the image and audio denoisers, with resumable
checkpointing: pass `checkpoint_path` to save model+optimizer+scheduler
state after every epoch, and to automatically resume from it if it
already exists (e.g. after a Colab disconnect and re-run)."""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.checkpoint import save_checkpoint, load_checkpoint


def train_img_model(model, name, train_loader, diff, timesteps, epochs, lr, device,
                     checkpoint_path=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    start_epoch = 0
    if checkpoint_path is not None:
        result = load_checkpoint(checkpoint_path, model, opt, sched, device=device)
        if result is not None:
            last_epoch, _ = result
            start_epoch = last_epoch + 1
            print(f"[Resume] {name}: found checkpoint at epoch {last_epoch + 1}/{epochs}, "
                  f"resuming from epoch {start_epoch + 1}.")

    if start_epoch >= epochs:
        print(f"[Resume] {name}: already fully trained ({epochs} epochs) -- skipping training.")
        return model

    print(f"\nTraining {name}...")
    for ep in range(start_epoch, epochs):
        model.train()
        total = 0.0
        for x, _ in tqdm(train_loader, leave=False, desc=f"{name} Ep {ep + 1}"):
            x = x.to(device)
            t = torch.randint(0, timesteps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            xt = diff.q_sample(x, t, noise)
            pred = model(xt, t)
            loss = F.mse_loss(pred, noise)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        print(f"Epoch {ep + 1}/{epochs}  Loss: {total / len(train_loader):.4f}")

        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, model, opt, sched, epoch=ep)
    return model


def train_audio_model(model, name, train_loader, corruption, T_audio, epochs, lr, device,
                       checkpoint_path=None):
    """
    Args:
        corruption: any object exposing `.corrupt(x0, t, device) -> (noisy, target)`,
            e.g. a PoissonDiffusion / GaussianAudioDiffusion / BernoulliAudioDiffusion
            instance from src/corruption_registry.py.
        checkpoint_path: if given, model+optimizer+scheduler state is saved
            after every epoch, and training auto-resumes from it on restart.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    start_epoch = 0
    if checkpoint_path is not None:
        result = load_checkpoint(checkpoint_path, model, opt, sched, device=device)
        if result is not None:
            last_epoch, _ = result
            start_epoch = last_epoch + 1
            print(f"[Resume] {name}: found checkpoint at epoch {last_epoch + 1}/{epochs}, "
                  f"resuming from epoch {start_epoch + 1}.")

    if start_epoch >= epochs:
        print(f"[Resume] {name}: already fully trained ({epochs} epochs) -- skipping training.")
        return model

    print(f"\nTraining {name}...")
    for ep in range(start_epoch, epochs):
        model.train()
        total = 0.0
        for x0 in tqdm(train_loader, leave=False, desc=f"{name} Ep {ep + 1}"):
            x0 = x0.to(device)
            t = torch.randint(0, T_audio, (x0.size(0),), device=device)
            noisy, target = corruption.corrupt(x0, t, device)
            pred = model(noisy, t)
            loss = F.mse_loss(pred, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        print(f"Epoch {ep + 1}/{epochs}  Loss: {total / len(train_loader):.5f}")

        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, model, opt, sched, epoch=ep)
    return model
