"""Verify project-starter-cnn-mini: inject reference impls, run all tests.

Usage (dari root repo):
    python AI-Engineering-Curriculum/05-Computer-Vision/_verify_project.py

Student stubs di nn_mini/{conv,augment,model}.py di-patch dengan
solusi/cnn_mini_ref.py supaya seluruh test suite bisa dijalankan end-to-end.
"""

import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent / "project-starter-cnn-mini"

sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "solusi"))

import cnn_mini_ref as ref  # noqa: E402

# ---- patch stub module namespaces dengan implementasi referensi ----
import nn_mini.conv as conv_mod  # noqa: E402
import nn_mini.augment as aug_mod  # noqa: E402
import nn_mini.model as model_mod  # noqa: E402

conv_mod.pad_zero = ref.pad_zero
conv_mod.conv2d_forward = ref.conv2d_forward
conv_mod.conv2d_backward = ref.conv2d_backward
conv_mod.maxpool2x2_forward = ref.maxpool2x2_forward
conv_mod.maxpool2x2_backward = ref.maxpool2x2_backward
conv_mod.Flatten = ref.Flatten
conv_mod.Conv2D = ref.Conv2D

aug_mod.flip_horizontal = ref.flip_horizontal
aug_mod.flip_vertical = ref.flip_vertical
aug_mod.random_noise = ref.random_noise
aug_mod.random_shift = ref.random_shift
aug_mod.augment_batch = ref.augment_batch

model_mod.ReLU = ref.ReLU
model_mod.MaxPool2x2 = ref.MaxPool2x2
model_mod.CNNMini = ref.CNNMini
model_mod.Conv2D = ref.Conv2D
model_mod.Flatten = ref.Flatten
model_mod.conv2d_forward = ref.conv2d_forward
model_mod.maxpool2x2_forward = ref.maxpool2x2_forward
model_mod.maxpool2x2_backward = ref.maxpool2x2_backward

# ---- jalankan test suite ----
loader = unittest.TestLoader()
suite = loader.discover(str(BASE / "tests"))
runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)

print("\n" + "=" * 60)
total = result.testsRun
failed = len(result.failures) + len(result.errors)
if failed:
    print(f"FAILED: {failed} test gagal dari {total}")
    sys.exit(1)
print(f"ALL {total} TESTS OK")
