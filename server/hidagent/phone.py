"""Touch-like gestures on the phone, built from relative mouse moves.

The pointer is relative, so the server tracks its position and recovers it visually
when needed. Button-up transit can use a separately calibrated fast mode; drawing
keeps fixed unit reports. Coordinates are screen fractions (0..1, origin top-left).
"""

import asyncio
import json
import logging
import math
import time
from collections import deque
from pathlib import Path

from . import vision
from .device import Device, DeviceError

log = logging.getLogger(__name__)

CALIBRATION_MOVE = 200  # mouse units moved diagonally for a manual (one point) calibration; stays on screen
CALIBRATION_STOPS = (150, 350)  # automatic calibration finds the pointer at these distances from the corner
FAST_THRESHOLD = 240    # use fast transit only when unit motion would take at least ~2 s
FAST_PROBE = 100        # independent X/Y probes; each full fast report is 10 units
HOLD = 0.25             # seconds to rest before lifting, so iOS adds no momentum (the swipe stops where it ends)
PEN_DOWN = 0.15         # let iOS register contact before streaming the first drawing segment
TAP_GAP = 0.15          # seconds to stay still after a click, so iOS sees a finished tap, not the start of a drag
NUDGE = 12              # mouse units the pointer is wiggled by to find it on screen
AIM_TOLERANCE = 0.006   # fraction of screen width, on both axes (~3 px at 498 px wide)
AIM_ATTEMPTS = 4


def drawing_park(x: float, y: float) -> tuple[float, float]:
    """Move the lifted pointer clear of the ink and the feedback mask."""
    return (x + 0.07 if x <= 0.85 else x - 0.07), y


class Phone:
    def __init__(self, device: Device, calibration_file: Path, screen=None):
        self.device, self._file, self.screen = device, calibration_file, screen
        # mouse units needed to cross the full screen width / height
        saved = json.loads(calibration_file.read_text()) if calibration_file.exists() else None
        self.span = {"w": saved["w"], "h": saved["h"]} if saved else None
        self._profile = (saved or {}).get("motion_profile", "legacy")
        self._calibration_pending = False
        self._pos = None  # pointer position in mouse units; None = unknown
        self._link = 0  # connection associated with the tracked position
        self._fast_span = (saved or {}).get("fast_span") if (saved or {}).get("fast_profile") == "fast-v1" else None
        # Calibration observations define the coordinate mapping. Do not fit
        # accumulated path drift as if it were a new absolute calibration.
        self._seen = {axis: deque(map(tuple, (saved or {}).get("seen", {}).get(axis, [])), maxlen=12)
                      for axis in ("x", "y")}

    @property
    def calibrated(self) -> bool:
        return self.span is not None and self._profile == getattr(self.device, "motion_profile", "legacy")

    async def start_calibration(self) -> bool:
        """Find the pointer on screen at two known distances from the corner and fit the mapping from both.

        Two points give the scale and the offset, which one point cannot (iOS pointer motion is not exactly
        proportional). Returns True when it calibrated itself; otherwise the user clicks the pointer
        (finish_calibration).
        """
        self._pos, self.span = None, None
        self._fast_span = None
        self._calibration_pending = False
        await self.device.send("release")
        await self.device.send("home")
        seen = []
        stops = (250, 500) if getattr(self.device, "motion_profile", "legacy") == "unit-v1" else CALIBRATION_STOPS
        for units in stops:
            step = units - (seen[-1][0] if seen else 0)
            await self.device.send(f"move {step} {step}")
            found = await self._find_pointer() if self.screen is not None and self.screen.live else None
            if not found:
                break
            seen.append((units, *found))
        if len(seen) < 2 or seen[1][1] - seen[0][1] < 0.01 or seen[1][2] - seen[0][2] < 0.01:
            log.info("calibration: pointer not found on screen, waiting for a click")
            # A failed second observation has already moved to the second stop.
            # Always establish the exact position that finish_calibration uses.
            await self.device.send("home")
            await self.device.send(f"move {CALIBRATION_MOVE} {CALIBRATION_MOVE}")
            self._calibration_pending = True
            return False
        (u1, x1, y1), (u2, x2, y2) = seen
        self.span = {"w": round((u2 - u1) / (x2 - x1)), "h": round((u2 - u1) / (y2 - y1))}
        self._profile = getattr(self.device, "motion_profile", "legacy")
        for samples in self._seen.values():
            samples.clear()
        for units, x, y in seen:  # where each stop landed, so moves are corrected for the offset too
            self._seen["x"].append((units / self.span["w"], x))
            self._seen["y"].append((units / self.span["h"], y))
        self._save()
        self._pos, self._link = (u2, u2), self.device.links
        log.info("calibration: pointer at %s, span %s", [(round(x, 3), round(y, 3)) for _, x, y in seen], self.span)
        return True

    async def _find_pointer(self, nudge: int = NUDGE) -> tuple[float, float] | None:
        """Wiggle the pointer and find it anywhere on screen (the span may still be unknown)."""
        await self.screen.settled(quiet=0.2, least=0.3, most=1.5)
        before, sent = self.screen.frame, time.time()
        await self.device.send(f"move {nudge} 0")
        after = await self.screen.fresh(sent, 1.0)
        if after:
            await self.screen.settled(quiet=0.15, least=0.1, most=0.6)
            after = self.screen.frame
        await self.device.send(f"move {-nudge} 0")
        found = before and after and vision.pointer(before, after, (0.5, 0.5), None,
                                                   reach=(0.5, 0.5), direction=nudge)
        log.debug("pointer nudge %d: %s", nudge, found)
        return found if found and 0.03 < found[0] < 0.97 and 0.03 < found[1] < 0.97 else None

    def finish_calibration(self, x: float, y: float) -> None:
        """x, y: where the pointer ended up after start_calibration() (one point, picked by the user)."""
        if not self._calibration_pending:
            raise RuntimeError("start calibration before selecting the pointer")
        if not all(math.isfinite(v) and 0.03 < v < 0.97 for v in (x, y)):
            raise RuntimeError("calibration pointer must be away from screen edges")
        self.span = {"w": round(CALIBRATION_MOVE / x), "h": round(CALIBRATION_MOVE / y)}
        self._profile = getattr(self.device, "motion_profile", "legacy")
        self._calibration_pending = False
        self._pos, self._link = (CALIBRATION_MOVE, CALIBRATION_MOVE), self.device.links
        for samples in self._seen.values():
            samples.clear()
        self._save()

    def _save(self) -> None:
        """Calibration plus the observed landings, so the correction survives a restart."""
        self._file.write_text(json.dumps({**self.span, "motion_profile": self._profile,
                                         "fast_profile": "fast-v1", "fast_span": self._fast_span,
                                         "seen": {k: list(v) for k, v in self._seen.items()}}))

    async def _ensure_position(self) -> None:
        """Recover without visiting a corner unless the pointer cannot be seen."""
        if self._link != self.device.links:
            await self.device.send("release")
            self._pos, self._link = None, self.device.links
        if self._pos is not None:
            return
        if self.screen is not None and self.screen.live:
            for nudge in (NUDGE, -NUDGE):
                found = await self._find_pointer(nudge)
                if found and all(0.03 < v < 0.97 for v in found):
                    self._pos = (self._dx(found[0]), self._dy(found[1]))
                    log.debug("pointer recovered on screen at %s", found)
                    return
        log.info("pointer not visible; recovering position at the corner")
        await self.device.send("home")
        self._pos = (0, 0)

    async def _calibrate_fast(self) -> bool:
        """Measure fast-mode gain independently; never reuse the fine-motion scale."""
        await self._ensure_position()
        before = await self._find_pointer()
        if not before and self._pos == (0, 0):
            # The status-bar corner is excluded from image matching. A short
            # button-up move exposes the pointer for the initial observation.
            self._pos = None
            await self.device.send("fast 80 80")
            before = await self._find_pointer()
        if not before:
            log.warning("fast calibration: initial pointer observation unavailable")
            self._pos = None
            return False
        spans = {}
        for axis, key in enumerate(("w", "h")):
            step = FAST_PROBE if before[axis] < 0.5 else -FAST_PROBE
            self._pos = None
            await self.device.send(f"fast {step if axis == 0 else 0} {step if axis == 1 else 0}")
            after = await self._find_pointer()
            log.info("fast calibration %s: %s -> %s (%d units)", key, before, after, step)
            if not after:
                return False
            self._pos = (self._dx(after[0]), self._dy(after[1]))
            distance = (after[axis] - before[axis]) * (1 if step > 0 else -1)
            if not 0.015 < distance < 0.6:
                return False
            spans[key] = round(FAST_PROBE / distance)
            before = after
        self._fast_span = spans
        self._save()
        log.info("fast transit calibrated: %s", spans)
        return True

    async def point(self, x: float, y: float, aim: bool = False) -> bool:
        if not self.calibrated:
            raise RuntimeError("pointer not calibrated for this firmware; run calibration")
        fast = ("fast-v1" in getattr(self.device, "capabilities", ())
                and self.screen is not None and self.screen.live)
        if fast:
            if self._fast_span is None:
                fast = await self._calibrate_fast()
            await self._ensure_position()
            dx, dy = self._dx(x) - self._pos[0], self._dy(y) - self._pos[1]
            if fast and abs(dx) + abs(dy) >= FAST_THRESHOLD:
                # Edge gestures start where the pointer is clipped or hidden
                # by the status bar. Verify an interior waypoint, then use a
                # short fine move to reach the actual edge before pressing.
                fx, fy = min(max(x, .08), .92), min(max(y, .08), .92)
                dx = round((self._dx(fx) - self._pos[0]) * self._fast_span["w"] / self.span["w"])
                dy = round((self._dy(fy) - self._pos[1]) * self._fast_span["h"] / self.span["h"])
                self._pos = None
                await self.device.send(f"fast {dx} {dy}")
                # Fast transit is always verified before any click or press,
                # including ordinary taps which did not request precise mode.
                if not await self._aim(fx, fy):
                    raise RuntimeError("cannot verify fast pointer movement; check screen mirroring")
                if (fx, fy) != (x, y):
                    await self._move_to(self._dx(x), self._dy(y), rehome=False)
                    if aim:
                        return await self._aim(x, y)
                return True
        await self._move_to(self._dx(x), self._dy(y))
        if aim:
            return bool(self.screen is not None and self.screen.live and await self._aim(x, y))
        return True

    async def _aim(self, x: float, y: float) -> bool:
        """Observe and correct until the target is verified, rebasing the tracked position."""
        nudge = NUDGE if x < 0.9 else -NUDGE  # stay clear of the screen edge
        for attempt in range(AIM_ATTEMPTS):
            found = await self._find_pointer(nudge * (1 + attempt // 2))
            if not found:
                self._pos = None
                continue
            fx, fy = found
            log.debug("aim (%.3f, %.3f): pointer at (%.3f, %.3f)", x, y, fx, fy)
            self._pos = (self._dx(fx), self._dy(fy))
            aspect = getattr(self.screen, "aspect", 0.46)
            if abs(fx - x) <= AIM_TOLERANCE and abs(fy - y) <= AIM_TOLERANCE * aspect:
                self._pos = (self._dx(x), self._dy(y))
                return True
            if attempt < AIM_ATTEMPTS - 1:
                await self._move_to(self._dx(x), self._dy(y), rehome=False)
        self._pos = None
        return False

    def _fix(self, t: float, axis: str) -> float:
        """The fraction to aim at so the pointer lands on t, from the line through the observed landings."""
        pts = self._seen[axis]
        if len(pts) < 2:
            return t
        n = len(pts)
        mc, mf = sum(c for c, _ in pts) / n, sum(f for _, f in pts) / n
        var = sum((c - mc) ** 2 for c, _ in pts)
        a = sum((c - mc) * (f - mf) for c, f in pts) / var if var > 1e-4 else 1.0  # landed = a * aimed + b
        a = a if 0.85 < a < 1.15 else 1.0
        b = mf - a * mc
        return (t - b) / a if abs(b) < 0.1 else t

    async def _move_to(self, tx: int, ty: int, rehome: bool = True) -> None:
        if not rehome and (self._link != self.device.links or self._pos is None):
            self._pos = None
            raise DeviceError("pointer position lost during gesture")
        if rehome:
            await self._ensure_position()
        x, y = self._pos
        self._pos = None  # unknown until the move succeeds
        await self.device.send(f"move {tx - x} {ty - y}")
        self._pos = (tx, ty)

    async def reset(self) -> None:
        """Emergency stop releases at the current location; recover visually next time."""
        self._pos = None
        await self.device.send("release")

    async def tap(self, x: float, y: float, aim: bool = False) -> None:
        """aim=True finds the pointer on screen first (slower); worth it when a tap just missed."""
        if not await self.point(x, y, aim):
            raise RuntimeError("cannot verify pointer before tapping")
        await self.device.send("click")
        await asyncio.sleep(TAP_GAP)

    async def swipe(self, x1: float, y1: float, x2: float, y2: float, hold: bool = True) -> None:
        await self.drag([(x1, y1), (x2, y2)], hold)

    async def drag(self, points: list[tuple[float, float]], hold: bool = True, precise: bool = False) -> None:
        """Press at the first point, move through the rest, release: one continuous stroke.

        hold=True rests at the end first, so lists and pickers stop exactly there instead of flinging on.
        System gestures (home, back) need the flick, so they pass hold=False.
        """
        if not 2 <= len(points) <= 65:
            raise ValueError("a stroke needs 2-65 points")
        if not all(math.isfinite(v) and 0 <= v <= 1 for point in points for v in point):
            raise ValueError("stroke points must be inside the screen")
        if not await self.point(*points[0], aim=precise) and precise:
            raise RuntimeError("cannot verify drawing start; check screen mirroring and recalibrate")
        failed = False
        try:
            await self.device.send("press")
            if precise:
                await asyncio.sleep(PEN_DOWN)
            if self._link != self.device.links:
                raise DeviceError("device reconnected before gesture")
            if getattr(self.device, "motion_profile", "legacy") == "unit-v1":
                # One command prevents network round trips from changing the
                # speed at each vertex, particularly on small circles.
                targets = [(self._dx(x), self._dy(y)) for x, y in points]
                deltas = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(targets, targets[1:], strict=False)]
                self._pos = None
                await self.device.send("path " + " ".join(str(v) for delta in deltas for v in delta))
                if self._link != self.device.links:
                    raise DeviceError("device reconnected during gesture")
                self._pos = targets[-1]
            else:
                for x, y in points[1:]:
                    await self._move_to(self._dx(x), self._dy(y), rehome=False)
            if hold:
                await asyncio.sleep(HOLD)
        except BaseException:
            failed = True
            self._pos = None
            raise
        finally:
            try:
                await self.device.send("release")
            except DeviceError:
                self._pos = None
                if not failed:
                    raise
        if precise:
            x, y = drawing_park(*points[-1])
            await self._move_to(self._dx(x), self._dy(y), rehome=False)

    async def scroll(self, x: float, y: float, lines: int) -> None:
        await self.point(x, y)
        await self.device.send(f"scroll {lines} 0")

    async def type(self, text: str) -> None:
        await self.device.send(f"type {text}")

    async def key(self, combo: str) -> None:
        await self.device.send(f"key {combo}")

    async def go_home(self) -> None:
        await self.swipe(0.5, 0.995, 0.5, 0.6, hold=False)  # resting here would open the app switcher

    async def go_back(self) -> None:
        await self.swipe(0.005, 0.45, 0.6, 0.45, hold=False)

    def _dx(self, x: float) -> int:
        return round(min(max(self._fix(x, "x"), 0), 1) * self.span["w"])

    def _dy(self, y: float) -> int:
        return round(min(max(self._fix(y, "y"), 0), 1) * self.span["h"])
