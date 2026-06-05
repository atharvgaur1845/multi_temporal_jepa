"""EMA wiring invariants — the anti-collapse mechanism must be exactly right."""
import copy

import torch
import torch.nn as nn

from engine.ema import ema_update, momentum_schedule


def _pair():
    student = nn.Linear(8, 8)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    # perturb student so they differ
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn_like(p))
    return student, teacher


def test_teacher_has_no_grad():
    _, teacher = _pair()
    assert all(not p.requires_grad for p in teacher.parameters())


def test_momentum_zero_copies_student():
    student, teacher = _pair()
    ema_update(student, teacher, momentum=0.0)
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        assert torch.allclose(ps, pt, atol=1e-6), "m=0 should set teacher := student"


def test_momentum_one_freezes_teacher():
    student, teacher = _pair()
    before = [p.clone() for p in teacher.parameters()]
    ema_update(student, teacher, momentum=1.0)
    for b, pt in zip(before, teacher.parameters()):
        assert torch.allclose(b, pt, atol=1e-6), "m=1 should leave teacher unchanged"


def test_schedule_ramps_up():
    m0 = momentum_schedule(0, 1000, base=0.996, final=1.0)
    m1 = momentum_schedule(1000, 1000, base=0.996, final=1.0)
    assert abs(m0 - 0.996) < 1e-6 and abs(m1 - 1.0) < 1e-6 and m1 >= m0
