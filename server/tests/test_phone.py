import asyncio
import json
from types import SimpleNamespace

import pytest

from hidagent.agent import circle
from hidagent.device import DeviceError
from hidagent.phone import Phone


class Board:
    links = 1
    motion_profile = "unit-v1"

    def __init__(self):
        self.commands = []
        self.position = (0, 0)
        self.ink = []
        self.pressed = False
        self.initial_error = (0, 0)

    async def send(self, command):
        self.commands.append(command)
        name, *args = command.split()
        if name == "home":
            self.position = (0, 0)
        elif name == "press":
            self.pressed = True
            self.ink.append(self.position)
        elif name == "release":
            self.pressed = False
        elif name in ("move", "path"):
            values = list(map(int, args))
            for dx, dy in zip(values[::2], values[1::2], strict=True):
                x, y = self.position
                ex, ey = self.initial_error
                self.initial_error = (0, 0)
                self.position = (x + dx + ex, y + dy + ey)
                if self.pressed:
                    self.ink.append(self.position)


def phone(tmp_path, board=None):
    board = board or Board()
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"w": 1000, "h": 1000, "motion_profile": board.motion_profile}))
    p = Phone(board, calibration, SimpleNamespace(live=True))

    async def find(nudge=12):
        return tuple(v / 1000 for v in board.position)

    p._find_pointer = find
    return p, board


def test_drawing_rebases_verified_start_and_preserves_entire_small_circle(tmp_path):
    p, board = phone(tmp_path)
    board.initial_error = (30, -40)
    points = circle(.39, .43, .017, .46)
    asyncio.run(p.drag(points, hold=False, precise=True))
    expected = [(round(x * 1000), round(y * 1000)) for x, y in points]
    assert board.ink == expected  # the correction must not be undone at the next vertex
    assert board.ink[0] == board.ink[-1]
    assert sum(c.startswith("path ") for c in board.commands) == 1
    assert not board.pressed


def test_drawing_does_not_press_when_pointer_cannot_be_verified(tmp_path):
    p, board = phone(tmp_path)

    async def missing(nudge=12):
        return None

    p._find_pointer = missing
    with pytest.raises(RuntimeError, match="cannot verify"):
        asyncio.run(p.drag([(.5, .5), (.6, .6)], precise=True))
    assert "press" not in board.commands
    assert p._pos is None


def test_drawing_retries_a_missing_frame_before_pressing(tmp_path):
    p, board = phone(tmp_path)
    p._pos, p._link = (0, 0), board.links
    calls = 0

    async def find(nudge=12):
        nonlocal calls
        calls += 1
        return None if calls == 1 else tuple(v / 1000 for v in board.position)

    p._find_pointer = find
    asyncio.run(p.drag([(.5, .5), (.6, .6)], hold=False, precise=True))
    assert calls == 2
    assert board.ink == [(500, 500), (600, 600)]
    assert board.position == (670, 600) and not board.pressed


def test_drawing_stops_when_corrections_do_not_converge(tmp_path):
    p, board = phone(tmp_path)

    async def stuck(nudge=12):
        return .4, .4

    p._find_pointer = stuck
    with pytest.raises(RuntimeError, match="cannot verify"):
        asyncio.run(p.drag([(.5, .5), (.6, .6)], precise=True))
    assert "press" not in board.commands
    assert len(board.commands) < 10


def test_firmware_change_invalidates_saved_calibration(tmp_path):
    p, board = phone(tmp_path)
    board.motion_profile = "legacy"
    assert not p.calibrated
    with pytest.raises(RuntimeError, match="calibrat"):
        asyncio.run(p.point(.5, .5))
    assert not board.commands


@pytest.mark.parametrize("observations", [[(.3, .2), None], [(.3, .2), (.3, .2)], [(.3, .2), (.2, .1)]])
def test_failed_calibration_always_resets_to_manual_distance(tmp_path, observations):
    p, board = phone(tmp_path)
    observations = iter(observations)

    async def find(nudge=12):
        return next(observations)

    p._find_pointer = find
    assert asyncio.run(p.start_calibration()) is False
    assert board.commands[-2:] == ["home", "move 200 200"]
    assert not p.calibrated
    p.finish_calibration(.2, .2)
    assert p.span == {"w": 1000, "h": 1000}
    assert Phone(board, p._file).calibrated


def test_successful_calibration_uses_observed_scale_and_offset(tmp_path):
    p, board = phone(tmp_path)
    observations = iter([(.27, .135), (.52, .26)])

    async def find(nudge=12):
        return next(observations)

    p._find_pointer = find
    assert asyncio.run(p.start_calibration()) is True
    assert p.span == {"w": 1000, "h": 2000}
    assert p._dx(.52) == 500 and p._dy(.26) == 500
    assert json.loads(p._file.read_text())["motion_profile"] == "unit-v1"


def test_legacy_stroke_does_not_home_after_reconnect(tmp_path):
    board = Board()
    board.motion_profile = "legacy"
    p, _ = phone(tmp_path, board)
    send = board.send

    async def reconnect(command):
        await send(command)
        if command == "move 100 100":
            board.links += 1

    board.send = reconnect
    with pytest.raises(DeviceError, match="position lost"):
        asyncio.run(p.drag([(.5, .5), (.6, .6), (.7, .7)], hold=False))
    assert board.commands.count("home") == 1
    assert board.commands[-1] == "release"
    assert p._pos is None


def test_repeated_positioning_and_stop_do_not_rehome(tmp_path):
    p, board = phone(tmp_path)
    board.position = (400, 400)

    async def scenario():
        for i in range(25):
            await p.point(.4 + i / 1000, .4)
        await p.reset()
        await p.point(.5, .5)

    asyncio.run(scenario())
    assert "home" not in board.commands
    assert board.position == (500, 500)


def test_fast_transit_uses_its_own_scale_and_fine_correction(tmp_path):
    p, board = phone(tmp_path)
    board.capabilities = {"fast-v1"}
    board.position = (200, 200)
    p._fast_span = {"w": 250, "h": 250}  # fast reports move four times as far
    send = board.send
    observed = []

    async def fast(command):
        if command.startswith("fast "):
            board.commands.append(command)
            _, dx, dy = command.split()
            board.position = (board.position[0] + int(dx) * 4 + 10, board.position[1] + int(dy) * 4 - 10)
        else:
            await send(command)

    async def find(nudge=12):
        observed.append(board.position)
        return tuple(v / 1000 for v in board.position)

    board.send, p._find_pointer = fast, find
    asyncio.run(p.tap(.8, .7))
    assert "fast 150 125" in board.commands
    assert "move -10 10" in board.commands
    assert observed[-1] == (800, 700)
    assert board.commands[-1] == "click"
    assert "home" not in board.commands


def test_failed_fast_verification_never_clicks(tmp_path):
    p, board = phone(tmp_path)
    board.capabilities = {"fast-v1"}
    p._fast_span = {"w": 250, "h": 250}
    p._pos, p._link = (200, 200), board.links

    async def missing(nudge=12):
        return None

    p._find_pointer = missing
    with pytest.raises(RuntimeError, match="cannot verify fast"):
        asyncio.run(p.tap(.8, .7))
    assert "click" not in board.commands and "press" not in board.commands


@pytest.mark.parametrize("start,end", [((.5, .995), (.5, .6)), ((.005, .45), (.6, .45))])
def test_fast_transit_preserves_gestures_starting_at_screen_edges(tmp_path, start, end):
    p, board = phone(tmp_path)
    board.capabilities = {"fast-v1"}
    board.position = (700, 300)
    p._fast_span = {"w": 1000, "h": 1000}
    send = board.send

    async def fast(command):
        await send(command.replace("fast ", "move ", 1))

    async def find(nudge=12):
        x, y = (v / 1000 for v in board.position)
        return (x, y) if .03 < x < .97 and .05 < y < .97 else None

    board.send, p._find_pointer = fast, find
    asyncio.run(p.swipe(*start, *end, hold=False))
    assert board.ink[0] == tuple(round(v * 1000) for v in start)
    assert board.ink[-1] == tuple(round(v * 1000) for v in end)
    assert "home" not in board.commands and not board.pressed


def test_fast_calibration_is_independent_and_saved(tmp_path):
    p, board = phone(tmp_path)
    board.capabilities = {"fast-v1"}
    board.position = (300, 300)
    send = board.send

    async def fast(command):
        if command.startswith("fast "):
            board.commands.append(command)
            _, dx, dy = command.split()
            board.position = (board.position[0] + int(dx) * 3, board.position[1] + int(dy) * 2)
        else:
            await send(command)

    board.send = fast
    assert asyncio.run(p._calibrate_fast())
    assert p._fast_span == {"w": 333, "h": 500}
    assert p.span == {"w": 1000, "h": 1000}
    assert Phone(board, p._file)._fast_span == p._fast_span
    assert "home" not in board.commands


def test_cancelled_path_releases_button_and_invalidates_position(tmp_path):
    p, board = phone(tmp_path)
    send = board.send

    async def cancelled(command):
        if command.startswith("path "):
            raise asyncio.CancelledError
        await send(command)

    board.send = cancelled
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(p.drag([(.5, .5), (.6, .6)], hold=False))
    assert board.commands[-1] == "release"
    assert not board.pressed and p._pos is None


def test_link_loss_during_cancel_does_not_mask_cancellation(tmp_path):
    p, board = phone(tmp_path)
    send = board.send
    cancelled = False

    async def disconnected(command):
        nonlocal cancelled
        if command.startswith("path "):
            cancelled = True
            raise asyncio.CancelledError
        if cancelled:
            raise DeviceError("board not connected")
        await send(command)

    board.send = disconnected
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(p.drag([(.5, .5), (.6, .6)], hold=False))
    assert p._pos is None
