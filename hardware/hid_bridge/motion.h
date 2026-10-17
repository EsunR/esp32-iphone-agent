#pragma once

#include <stdint.h>
#include <stdlib.h>

// Interpolate the whole segment, with at most one mouse unit per report.
// Axis-only reports keep the report magnitude identical for every direction,
// including tiny circles and calibration. emit returns false to abort a move.
template <typename Emit>
bool walkLine(long dx, long dy, Emit emit) {
  const long ax = labs(dx), ay = labs(dy);
  const long steps = ax > ay ? ax : ay;
  long x = 0, y = 0;
  for (long i = 1; i <= steps; ++i) {
    const long tx = (int64_t(ax) * i + steps / 2) / steps;
    const long ty = (int64_t(ay) * i + steps / 2) / steps;
    if (tx != x && !emit(dx < 0 ? -1 : 1, 0)) return false;
    if (ty != y && !emit(0, dy < 0 ? -1 : 1)) return false;
    x = tx;
    y = ty;
  }
  return true;
}

// Button-up transit uses the same interpolation on a coarser grid. Full
// reports have a fixed magnitude of 10; at most two short tail reports remain.
template <typename Emit>
bool walkFast(long dx, long dy, Emit emit) {
  if (!walkLine(dx / 10, dy / 10, [&](int x, int y) { return emit(x * 10, y * 10); })) return false;
  if (dx % 10 && !emit(dx % 10, 0)) return false;
  if (dy % 10 && !emit(0, dy % 10)) return false;
  return true;
}
