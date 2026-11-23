"""Unit tests for Knowledge Distillation (NightKD) pipeline."""

import sys
from pathlib import Path

# Ensure repo root is always in Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import torch
import torch.nn as nn
from src.losses import CombinedLoss
from src.distillation.loss import KnowledgeDistillationLoss
from src.distillation.teacher import DummyTeacher, build_teacher
from src.models.model import build_model


class TestDistillation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.num_classes = 19
        self.ignore_index = 255
        self.b, self.h, self.w = 2, 64, 64

        self.student_logits = torch.randn(self.b, self.num_classes, self.h, self.w, requires_grad=True)
        self.teacher_logits = torch.randn(self.b, self.num_classes, self.h, self.w)
        self.targets = torch.randint(0, self.num_classes, (self.b, self.h, self.w), dtype=torch.long)
        self.targets[0, :8, :8] = self.ignore_index

        self.base_loss = CombinedLoss(num_classes=self.num_classes, ignore_index=self.ignore_index)
        self.kd_loss = KnowledgeDistillationLoss(
            base_criterion=self.base_loss,
            temperature=4.0,
            alpha=0.5,
            conf_threshold=0.5,
            use_confidence_masking=True,
            ignore_index=self.ignore_index,
        )

    def test_kd_loss_without_teacher(self):
        """Verify KD loss gracefully falls back to task loss if teacher_logits is None."""
        loss = self.kd_loss(self.student_logits, self.targets, teacher_logits=None)
        expected = self.base_loss(self.student_logits, self.targets)
        self.assertAlmostEqual(loss.item(), expected.item(), places=4)

    def test_kd_loss_with_teacher_and_gradients(self):
        """Verify KD loss computation and gradient backpropagation to student."""
        loss = self.kd_loss(self.student_logits, self.targets, teacher_logits=self.teacher_logits)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(self.student_logits.grad)
        self.assertTrue(torch.isfinite(self.student_logits.grad).all())

    def test_confidence_masking(self):
        """Verify confidence masking filters low-confidence teacher regions."""
        # Create teacher logits with very low confidence (< 0.1) across all classes
        uniform_teacher = torch.zeros(self.b, self.num_classes, self.h, self.w)
        loss = self.kd_loss(self.student_logits, self.targets, teacher_logits=uniform_teacher)
        # Should fall back cleanly without exploding
        self.assertTrue(torch.isfinite(loss))

    def test_dummy_teacher_resolution(self):
        """Verify teacher forward outputs matching spatial resolution."""
        teacher = DummyTeacher(num_classes=19)
        images = torch.randn(2, 3, 128, 128)
        logits = teacher(images)
        self.assertEqual(logits.shape, (2, 19, 128, 128))

    def test_student_model_parameters_untouched(self):
        """Verify student model parameters remain strictly 26,975."""
        student = build_model()
        params = sum(p.numel() for p in student.parameters())
        self.assertEqual(params, 26_975)


if __name__ == "__main__":
    unittest.main()
