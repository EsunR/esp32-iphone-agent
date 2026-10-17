import shutil
import subprocess
from pathlib import Path

import pytest


def test_firmware_interpolation_is_straight_constant_magnitude_and_interruptible(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("a C++ compiler is needed for firmware motion tests")
    source = tmp_path / "motion_test.cpp"
    source.write_text(r'''
#include <cassert>
#include <cmath>
#include "motion.h"
int main() {
  for (int dx = -45; dx <= 45; ++dx) {
    for (int dy = -45; dy <= 45; ++dy) {
      int x = 0, y = 0, reports = 0;
      assert(walkLine(dx, dy, [&](int sx, int sy) {
        assert(abs(sx) + abs(sy) == 1);
        x += sx; y += sy; ++reports;
        // Every emitted point stays within 1.5 units of the ideal segment.
        assert(std::abs(double(x) * dy - double(y) * dx) <= 1.5 * std::hypot(dx, dy));
        return true;
      }));
      assert(x == dx && y == dy);
      assert(reports == abs(dx) + abs(dy));
      x = y = reports = 0;
      assert(walkFast(dx, dy, [&](int sx, int sy) {
        assert((sx == 0) != (sy == 0));
        assert(abs(sx) + abs(sy) <= 10);
        x += sx; y += sy; ++reports;
        return true;
      }));
      assert(x == dx && y == dy);
      assert(reports == (abs(dx) + 9) / 10 + (abs(dy) + 9) / 10);
    }
  }
  int calls = 0;
  assert(!walkLine(100, 40, [&](int, int) { return ++calls < 5; }));
  assert(calls == 5);
  calls = 0;
  assert(!walkFast(1000, 400, [&](int, int) { return ++calls < 5; }));
  assert(calls == 5);
}
''')
    include = Path(__file__).resolve().parents[2] / "hardware" / "hid_bridge"
    binary = tmp_path / "motion_test"
    subprocess.run([compiler, "-std=c++11", "-I", str(include), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
