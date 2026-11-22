"""Unit tests for segmentation losses (baseline CombinedLoss and novel BOLTLoss)."""

import os
import sys
from pathlib import Path
import unittest

# Ensure repo root is always in Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from src.losses import (
    SoftDiceLoss,
    CombinedLoss,
    OHEMCrossEntropyLoss,
    ClassWeightedBoundaryLoss,
    BOLTLoss,
    build_loss,
    get_acdc_class_weights,
)


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.num_classes = 19
        self.ignore_index = 255
        self.batch_size = 2
        self.h = 64
        self.w = 64

        # Dummy logits and targets
        self.logits = torch.randn(self.batch_size, self.num_classes, self.h, self.w, requires_grad=True)
        self.targets = torch.randint(0, self.num_classes, (self.batch_size, self.h, self.w), dtype=torch.long)
        # Add some ignore pixels
        self.targets[0, 0:10, 0:10] = self.ignore_index

    def test_acdc_class_weights(self):
        weights = get_acdc_class_weights(c_smooth=1.02, num_classes=self.num_classes)
        self.assertEqual(weights.shape, (self.num_classes,))
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=4)
        # Road (class 0, dominant) should have lowest weight
        self.assertTrue(weights[0] < weights[11])  # road < person
        self.assertTrue(weights[0] < weights[13])  # road < car

    def test_soft_dice_loss(self):
        criterion = SoftDiceLoss(num_classes=self.num_classes, ignore_index=self.ignore_index)
        loss = criterion(self.logits, self.targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertLessEqual(loss.item(), 1.0)

        # Gradient check
        loss.backward()
        self.assertIsNotNone(self.logits.grad)
        self.assertTrue(torch.isfinite(self.logits.grad).all())

    def test_baseline_combined_loss(self):
        logits = self.logits.clone().detach().requires_grad_(True)
        criterion = CombinedLoss(num_classes=self.num_classes, ignore_index=self.ignore_index)
        loss = criterion(logits, self.targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_ohem_loss(self):
        logits = self.logits.clone().detach().requires_grad_(True)
        weights = get_acdc_class_weights()
        criterion = OHEMCrossEntropyLoss(
            ignore_index=self.ignore_index,
            class_weights=weights,
            ohem_ratio=0.5,
            min_kept=100,
        )
        loss = criterion(logits, self.targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_boundary_loss(self):
        logits = self.logits.clone().detach().requires_grad_(True)
        criterion = ClassWeightedBoundaryLoss(num_classes=self.num_classes, ignore_index=self.ignore_index)
        loss = criterion(logits, self.targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_bolt_loss(self):
        logits = self.logits.clone().detach().requires_grad_(True)
        criterion = BOLTLoss(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            use_class_weights=True,
            use_ohem=True,
            use_boundary_loss=True,
        )
        loss = criterion(logits, self.targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_build_loss_factory(self):
        # Baseline config
        cfg_baseline = {"training": {"loss_type": "baseline"}}
        loss_fn = build_loss(cfg_baseline)
        self.assertIsInstance(loss_fn, CombinedLoss)

        # BOLT config
        cfg_bolt = {"training": {"loss_type": "bolt"}}
        loss_fn_bolt = build_loss(cfg_bolt)
        self.assertIsInstance(loss_fn_bolt, BOLTLoss)

    def test_all_ignore_pixels_edge_case(self):
        """Ensure model does not crash if all pixels in batch are ignore_index."""
        all_ignore_targets = torch.full((1, 32, 32), self.ignore_index, dtype=torch.long)
        dummy_logits = torch.randn(1, self.num_classes, 32, 32, requires_grad=True)

        criterion = BOLTLoss(num_classes=self.num_classes, ignore_index=self.ignore_index)
        loss = criterion(dummy_logits, all_ignore_targets)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
