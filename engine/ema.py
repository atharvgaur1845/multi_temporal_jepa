"""EMA (exponential moving average) update for the target encoder + momentum schedule.

The target encoder is a slow-moving average of the context encoder. It receives NO gradient.
A *lagging* target is essential: it makes the trivial constant solution non-stationary, which
(together with the predictor bottleneck and stop-grad) prevents collapse.

    teacher_param <- m * teacher_param + (1 - m) * student_param   (for every param AND buffer)
    m ramps 0.996 -> 1.0 (linear) over training.
"""
from __future__ import annotations


def momentum_schedule(step, total_steps, base=0.996, final=1.0):
    """Linear ramp from `base` to `final` across `total_steps`.
    Returns the scalar momentum for this step. TODO: implement (clamp at final)."""
    raise NotImplementedError("M1")


def ema_update(student, teacher, momentum):
    """In-place EMA update of teacher from student.

    TODO
        - under torch.no_grad(), iterate paired params (and buffers) of student/teacher:
            teacher.data = momentum * teacher.data + (1 - momentum) * student.data
        - do NOT create gradient history.

    Invariants (tests/test_ema.py)
        - teacher params have requires_grad == False.
        - with momentum=0, teacher becomes student; with momentum=1, teacher is unchanged.
    """
    raise NotImplementedError("M1")
