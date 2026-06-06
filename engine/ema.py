"""EMA (exponential moving average) update for the target encoder + momentum schedule.

The target encoder is a slow-moving average of the context encoder. It receives NO gradient.
A *lagging* target is essential: it makes the trivial constant solution non-stationary, which
(together with the predictor bottleneck and stop-grad) prevents collapse.

    teacher_param <- m * teacher_param + (1 - m) * student_param   (for every param AND buffer)
    m ramps 0.996 -> 1.0 (linear) over training.
"""
from __future__ import annotations

import torch


def momentum_schedule(step, total_steps, base=0.996, final=1.0):
    """Linear ramp from `base` to `final` across `total_steps`. Clamped to `final` at the end."""
    if total_steps <= 0:
        return final
    frac = min(1.0, max(0.0, step / total_steps))
    return base + (final - base) * frac


@torch.no_grad()
def ema_update(student, teacher, momentum):
    """In-place EMA update of teacher from student (params AND float buffers).

        teacher <- momentum * teacher + (1 - momentum) * student

    With momentum=0 the teacher becomes the student; with momentum=1 it is unchanged.
    No gradient history is created (torch.no_grad). Float buffers (e.g. BN running stats) are
    EMA-blended; integer buffers (num_batches_tracked) are copied so the teacher stays valid.
    """
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.mul_(momentum).add_(ps.detach(), alpha=1.0 - momentum)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        if bt.dtype.is_floating_point:
            bt.mul_(momentum).add_(bs.detach(), alpha=1.0 - momentum)
        else:
            bt.copy_(bs)
