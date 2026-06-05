"""Pretraining loop for JEPA (spatial or temporal).

This orchestrates the pieces you implement elsewhere; the loop structure is given so you can
see the order of operations (esp. WHERE the EMA step happens — after the optimizer step, using
the just-updated student). The per-step math (forward/loss/backward) routes through your modules.

Loop (one step)
    1. build mask_spec for the batch (multiblock for spatial / past-future split for temporal)
    2. with autocast: pred, target = model(batch, mask_spec); loss = jepa_latent_loss(pred, target)
    3. scaler.scale(loss).backward(); (grad accumulation) optimizer step; zero grad
    4. m = momentum_schedule(step, total); ema_update(student=context_encoder, teacher=target_encoder, m)
    5. every log.diagnostics_every: log loss + collapse_metrics(...)  (and periodically a quick probe)

This file may be left mostly as a TODO skeleton; the *learning-critical* parts are the modules
it calls. Implement those first, then wire the loop.
"""
from __future__ import annotations


def train_one_epoch(model, loader, optimizer, scaler, scheduler, ema_cfg, step0, total_steps, logger):
    """Run one epoch. Returns the updated global step.

    TODO
        - iterate batches; build mask_spec; forward; loss; AMP backward with grad accumulation.
        - EMA update AFTER the optimizer step.
        - log diagnostics on schedule; NEVER judge by loss alone.
    """
    raise NotImplementedError("M1/M2")


def main(config_path):
    """Entry point: load config, build dataset/loader, model (JEPA), optimizer/schedule, run.
    TODO: assemble from the modules; keep everything seed-controlled and checkpointed."""
    raise NotImplementedError("M1/M2")
