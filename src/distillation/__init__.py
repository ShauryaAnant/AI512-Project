"""Knowledge Distillation package for ACDC Night semantic segmentation."""

from src.distillation.loss import KnowledgeDistillationLoss
from src.distillation.teacher import SegFormerTeacher, build_teacher, DummyTeacher

__all__ = ["KnowledgeDistillationLoss", "SegFormerTeacher", "build_teacher", "DummyTeacher"]
