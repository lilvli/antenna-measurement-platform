from __future__ import annotations

import asyncio
import re
import time

import numpy as np
import pytest

from antenna_service.devices.real import RealBeamController, RealTurntable, RealVna
from antenna_service.devices.simulated import SimulatedTurntable
from antenna_service.devices.manager import DeviceManager
from antenna_service.errors import ServiceError
from antenna_service.events import EventBus
from antenna_service.models import DeviceSource, DeviceState
from antenna_service.protocol.crc import append_crc_be
from antenna_service.protocol.profile import ResponseRule


class FakeImac:
    """Small vendor-DLL stand-in used only to verify our engineering-unit boundary."""

    def __init__(self) -> None:
        self.positions = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 7: 0.0}
        self.last_move: tuple[int, int, float, float, bool] | None = None
        self.i_values = {500: 1}
        self.m_values: dict[int, int] = {}

    def MoveDeviceToPos(self, device: int, axis: int, target: float, speed: float, absolute: bool) -> None:
        self.last_move = (device, axis, target, speed, absolute)
        self.positions[axis] = target

    def GetPosStr(self, _device: int, axis: int) -> str:
        return str(self.positions[axis])

    def GetVelStr(self, _device: int, _axis: int) -> str:
        return "0"

    def GetI(self, _device: int, variable: int) -> int:
        return self.i_values.get(variable, 0)

    def GetM(self, _device: int, variable: int) -> int:
        return self.m_values.get(variable, 0)


class MovingFakeImac(FakeImac):
    """Time-based stand-in that remains in motion long enough for concurrent readback."""

    def __init__(self, duration: float = 0.5) -> None:
        super().__init__()
        self.duration = duration
        self.started_at: float | None = None
        self.moving_axis: int | None = None
        self.start_position = 0.0
        self.target_position = 0.0
        self.commanded_speed = 0.0

    def MoveDeviceToPos(self, device: int, axis: int, target: float, speed: float, absolute: bool) -> None:
        self.last_move = (device, axis, target, speed, absolute)
        self.started_at = time.monotonic()
        self.moving_axis = axis
        self.start_position = self.positions[axis]
        self.target_position = target
        self.commanded_speed = speed

    def _progress(self) -> float:
        if self.started_at is None:
            return 1.0
        return min((time.monotonic() - self.started_at) / self.duration, 1.0)

    def GetPosStr(self, _device: int, axis: int) -> str:
        if axis != self.moving_axis:
            return str(self.positions[axis])
        progress = self._progress()
        value = self.start_position + (self.target_position - self.start_position) * progress
        if progress >= 1.0:
            self.positions[axis] = self.target_position
        return str(value)

    def GetVelStr(self, _device: int, axis: int) -> str:
        if axis == self.moving_axis and self._progress() < 1.0:
            return str(self.commanded_speed)
        return "0"

    def Stop(self, _device: int, axis: int) -> None:
        if axis != self.moving_axis:
            return
        self.positions[axis] = float(self.GetPosStr(_device, axis))
        self.started_at = None
        self.moving_axis = None
        self.commanded_speed = 0.0


class FakeSerial:
    """Thread-callable serial endpoint for testing the real adapter's RX ownership."""

    def __init__(self) -> None:
        self.incoming = bytearray()
        self.written = bytearray()
        self.closed = False

    @property
    def in_waiting(self) -> int:
        return len(self.incoming)

    def read(self, size: int) -> bytes:
        if not self.incoming:
            time.sleep(0.01)
            return b""
        data = bytes(self.incoming[:size])
        del self.incoming[:size]
        return data

    def write(self, payload: bytes) -> int:
        self.written.extend(payload)
        return len(payload)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def feed(self, payload: bytes) -> None:
        self.incoming.extend(payload)


class FakeVnaInstrument:
    """E5080B-shaped SCPI endpoint for verifying the structured VNA contract."""

    def __init__(self, *, setup_error: str | None = None) -> None:
        self.writes: list[str] = []
        self.setup_error = setup_error
        self.error_reads = 0
        self.timeout = 10_000
        self.display_traces: dict[int, dict[int, str]] = {1: {1: "my_trace1", 2: "my_trace2"}}
        self.active_measurement = "my_trace1"
        self.measurement_number = 7
        self.measurement_name = "ANTENNA_MEAS"
        self.queries: list[str] = []

    def write(self, command: str) -> None:
        self.writes.append(command)
        match = re.fullmatch(r"DISP:WIND(\d+):TRAC(\d+):FEED '([^']+)'", command)
        if match:
            window, trace, measurement = match.groups()
            self.display_traces.setdefault(int(window), {})[int(trace)] = measurement
            return
        match = re.fullmatch(r"DISP:WIND(\d+):TRAC(\d+):SEL", command)
        if match:
            window, trace = (int(value) for value in match.groups())
            self.active_measurement = self.display_traces[window][trace]
            return
        match = re.fullmatch(r"DISP:WIND(\d+):STAT ON", command)
        if match:
            self.display_traces.setdefault(int(match.group(1)), {})

    def query(self, command: str) -> str:
        self.queries.append(command)
        average_count_command = next((item for item in reversed(self.writes) if item.startswith("SENS1:AVER:COUN ")), None)
        average_count = average_count_command.split()[-1] if average_count_command else "1"
        power_command = next((item for item in reversed(self.writes) if item.startswith("SOUR1:POW") and " " in item), None)
        power = power_command.split()[-1] if power_command else "-10"
        modified_measurement = next(
            (item for item in reversed(self.writes) if item.startswith("CALC1:PAR:MOD:EXT ")),
            None,
        )
        selected_s_parameter = modified_measurement.rsplit("'", 2)[1] if modified_measurement else "S11"
        replies = {
            "*OPC?": "+1",
            "CALC1:PAR:CAT:EXT? DEF": f'"ANTENNA_MEAS,{selected_s_parameter}"',
            "DISP:CAT?": '"' + ",".join(str(value) for value in self.display_traces) + '"',
            "SYST:ACT:MEAS?": f'"{self.active_measurement}"',
            "CALC1:PAR:MNUM?": str(self.measurement_number),
            f"SYST:MEAS{self.measurement_number}:NAME?": f'"{self.measurement_name}"',
            "SENS1:FREQ:STAR?": "+8.00000000000E+009",
            "SENS1:FREQ:STOP?": "+8.01000000000E+009",
            "SENS1:SWE:POIN?": "+3",
            "SENS1:BWID?": "+1.00000000000E+003",
            "SOUR1:POW1?": power,
            "SOUR1:POW2?": power,
            "SENS1:AVER?": "1" if "SENS1:AVER ON" in self.writes else "0",
            "SENS1:AVER:COUN?": average_count,
            "SENS1:AVER:MODE?": "SWEEP",
            "SENS1:SWE:TIME?": "0.01",
            "SENS1:SWE:TYPE?": "LIN",
            "TRIG:SOUR?": "IMM",
            "SENS1:SWE:TRIG:MODE?": "CHAN",
            "FORM:DATA?": "ASC,0",
        }
        if command == "SYST:ERR?":
            self.error_reads += 1
            if self.setup_error and self.error_reads == 2:
                return self.setup_error
            return '+0,"No error"'
        match = re.fullmatch(r"DISP:WIND(\d+):CAT\?", command)
        if match:
            traces = self.display_traces[int(match.group(1))]
            return '"' + (",".join(str(value) for value in traces) if traces else "EMPTY") + '"'
        match = re.fullmatch(r"DISP:WIND(\d+):TRAC:NEXT\?", command)
        if match:
            traces = self.display_traces[int(match.group(1))]
            return str(max(traces, default=0) + 1)
        return replies[command]

    def query_ascii_values(self, command: str, *, container=np.array):
        self.queries.append(command)
        values = {
            f"CALC1:MEAS{self.measurement_number}:X?": [8.0e9, 8.005e9, 8.01e9],
            f"CALC1:MEAS{self.measurement_number}:DATA:SDATA?": [1.0, 0.0, 0.5, -0.25, 0.0, 1.0],
        }[command]
        return container(values)


@pytest.mark.asyncio
async def test_real_turntable_uses_confirmed_scale_10000_for_move_and_readback(tmp_path):
    turntable = RealTurntable(str(tmp_path / "ImacFxDll.dll"))
    fake = FakeImac()
    turntable._imac = fake

    result = await turntable.move_to(1, 12.345, 1.23456)

    assert RealTurntable.SCALE == 10000.0
    assert fake.last_move == (0, 1, 123450.0, 12346.0, True)
    assert result["position"] == pytest.approx(12.345)
    assert result["velocity"] == 0.0
    assert set(result["positions"]) == {"1", "2", "3", "4", "7"}
    readback = await turntable.read_all_axes()
    assert set(readback["positions"]) == {"1", "2", "3", "4", "7"}

    negative = await turntable.move_to(2, -3.25, 0.5)
    assert fake.last_move == (0, 2, -32500.0, 5000.0, True)
    assert negative["position"] == pytest.approx(-3.25)

    with pytest.raises(ServiceError, match="必须大于 0"):
        await turntable.move_to(1, 0, -0.0001)
    with pytest.raises(ServiceError, match="必须大于 0"):
        await turntable.move_to(1, 0, 0)


@pytest.mark.asyncio
async def test_real_turntable_allows_readback_while_motion_wait_is_active(tmp_path):
    turntable = RealTurntable(str(tmp_path / "ImacFxDll.dll"))
    fake = MovingFakeImac(duration=0.5)
    turntable._imac = fake

    move_task = asyncio.create_task(turntable.move_to(3, 2.0, 1.0, timeout_seconds=2.0))
    await asyncio.sleep(0.05)

    readback = await asyncio.wait_for(turntable.read_all_axes(), timeout=0.3)

    assert not move_task.done()
    assert 0.0 < readback["positions"]["3"] < 2.0
    assert readback["velocities"]["3"] == pytest.approx(1.0)
    result = await asyncio.wait_for(move_task, timeout=2.0)
    assert result["position"] == pytest.approx(2.0)
    assert result["velocity"] == 0.0


@pytest.mark.asyncio
async def test_real_turntable_rejects_inactive_or_faulted_translation_motor_before_move(tmp_path):
    turntable = RealTurntable(str(tmp_path / "ImacFxDll.dll"))
    fake = FakeImac()
    turntable._imac = fake

    fake.i_values[500] = 0
    with pytest.raises(ServiceError) as inactive_error:
        await turntable.move_to(7, 0.1, 0.1)
    assert inactive_error.value.code == "NOT_RUNNABLE"
    assert inactive_error.value.details["physical_motor"] == 5
    assert inactive_error.value.details["activation_i"] == 0
    assert fake.last_move is None

    fake.i_values[500] = 1
    fake.m_values[543] = 1
    with pytest.raises(ServiceError) as fault_error:
        await turntable.move_to(7, 0.1, 0.1)
    assert fault_error.value.details["amplifier_fault"] is True
    assert fake.last_move is None

    fake.m_values[543] = 0
    result = await turntable.move_to(7, 0.1, 0.1)
    assert fake.last_move == (0, 7, 1000.0, 1000.0, True)
    assert result["position"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_device_manager_allows_turntable_readback_and_stop_during_manual_motion(tmp_path):
    manager = DeviceManager(EventBus())
    turntable = RealTurntable(str(tmp_path / "ImacFxDll.dll"))
    fake = MovingFakeImac(duration=1.0)
    turntable._imac = fake
    turntable.snapshot.update(state=DeviceState.READY)
    manager.devices["turntable"] = turntable

    move_task = asyncio.create_task(
        manager.command("turntable", "move_to", {"axis": 4, "target": 2.0, "speed": 1.0})
    )
    for _ in range(50):
        if fake.started_at is not None:
            break
        await asyncio.sleep(0.005)

    readback = await asyncio.wait_for(manager.command("turntable", "read_axes", {}), timeout=0.3)
    assert not move_task.done()
    assert 0.0 <= readback["positions"]["4"] < 2.0
    assert readback["velocities"]["4"] == pytest.approx(1.0)

    with pytest.raises(ServiceError) as duplicate_error:
        await manager.command("turntable", "move_to", {"axis": 4, "target": 3.0, "speed": 1.0})
    assert duplicate_error.value.code == "CONTROL_LOCKED"

    stopped = await asyncio.wait_for(manager.command("turntable", "stop", {"axis": "all"}), timeout=0.5)
    assert stopped["velocities"]["4"] == 0.0
    with pytest.raises(ServiceError, match="已被软件停止"):
        await move_task


@pytest.mark.asyncio
async def test_real_vna_uses_e5080b_single_sweep_and_verifies_actual_axis():
    vna = RealVna("TCPIP0::192.168.1.100::hislip0::INSTR")
    instrument = FakeVnaInstrument()
    vna._instrument = instrument
    frequencies = np.array([8.0e9, 8.005e9, 8.01e9])

    readback = await vna.configure(
        s_parameter="S21",
        frequencies_hz=frequencies,
        if_bandwidth_hz=1000,
        source_power_dbm=-12.5,
        averaging_enabled=False,
        averaging_count=1,
    )
    values = await vna.acquire(frequencies)

    assert "CALC1:PAR:DEL:ALL" not in instrument.writes
    assert "SENS1:SWE:MODE HOLD" in instrument.writes
    assert "CALC1:PAR:MOD:EXT 'S21'" in instrument.writes
    assert "DISP:WIND1:TRAC3:FEED 'ANTENNA_MEAS'" in instrument.writes
    assert "DISP:WIND1:TRAC3:SEL" in instrument.writes
    assert "DISP:WIND1:TRAC3:TITL OFF" in instrument.writes
    assert not any(":TITL:DATA " in command for command in instrument.writes)
    assert "DISP:WIND1:TRAC3:Y:AUTO" in instrument.writes
    assert instrument.display_traces[1] == {1: "my_trace1", 2: "my_trace2", 3: "ANTENNA_MEAS"}
    assert "SENS1:SWE:TYPE LIN" in instrument.writes
    assert "SENS1:SWE:TRIG:MODE CHAN" in instrument.writes
    assert "SOUR1:POW1 -12.5" in instrument.writes
    assert "SENS1:AVER:MODE SWEEP" in instrument.writes
    assert "SENS1:AVER OFF" in instrument.writes
    assert "FORM:DATA ASC,0" in instrument.writes
    assert "SENS1:SWE:MODE SING" in instrument.writes
    assert readback["s_parameter"] == "S21"
    assert readback["display_window"] == 1
    assert readback["display_trace"] == 3
    assert readback["display_measurement"] == "ANTENNA_MEAS"
    assert readback["source_power_dbm"] == -12.5
    assert readback["averaging_enabled"] is False
    assert readback["error_queue"] == ['+0,"No error"']
    assert values.tolist() == [complex(1, 0), complex(0.5, -0.25), complex(0, 1)]
    assert vna.snapshot.details["last_frequency_axis_hz"] == frequencies.tolist()


@pytest.mark.asyncio
async def test_real_vna_reuses_existing_visible_measurement_trace():
    instrument = FakeVnaInstrument()
    instrument.display_traces[1][3] = "ANTENNA_MEAS"
    vna = RealVna("TCPIP0::192.168.1.100::hislip0::INSTR")
    vna._instrument = instrument

    readback = await vna.configure(
        s_parameter="S21",
        frequencies_hz=np.array([8.0e9, 8.005e9, 8.01e9]),
        if_bandwidth_hz=1000,
        source_power_dbm=-12.5,
        averaging_enabled=False,
        averaging_count=1,
    )

    assert not any(":FEED 'ANTENNA_MEAS'" in command for command in instrument.writes)
    assert instrument.display_traces[1] == {1: "my_trace1", 2: "my_trace2", 3: "ANTENNA_MEAS"}
    assert readback["display_trace"] == 3


@pytest.mark.asyncio
async def test_real_vna_rejects_nonzero_scpi_error_queue():
    vna = RealVna("TCPIP0::192.168.1.100::hislip0::INSTR")
    vna._instrument = FakeVnaInstrument(setup_error='-113,"Undefined header"')

    with pytest.raises(ServiceError, match="设置或读回验证失败") as caught:
        await vna.configure(
            s_parameter="S21",
            frequencies_hz=np.array([8.0e9, 8.005e9, 8.01e9]),
            if_bandwidth_hz=1000,
            source_power_dbm=-10,
            averaging_enabled=False,
            averaging_count=1,
        )
    assert caught.value.code == "DEVICE_FAULT"


@pytest.mark.asyncio
async def test_real_flash_requires_connected_serial_transport():
    beam = RealBeamController("COM_TEST")
    with pytest.raises(ServiceError, match="未连接") as caught:
        await beam.flash_write(1, 0x1000, bytes(256))
    assert caught.value.code == "NOT_RUNNABLE"


@pytest.mark.asyncio
async def test_real_flash_writes_and_reads_correlated_page_frames():
    events = EventBus()
    beam = RealBeamController("COM_TEST", events=events)
    serial = FakeSerial()
    beam._serial = serial
    beam._reader_task = asyncio.create_task(beam._receive_loop())
    tile_id = 3
    address = 0x1200
    page = bytes(index % 256 for index in range(256))

    try:
        write_task = asyncio.create_task(beam.flash_write(tile_id, address, page))
        for _ in range(50):
            if len(serial.written) == 286:
                break
            await asyncio.sleep(0.005)
        assert bytes(serial.written[:11]) == bytes.fromhex("AA 55 01 1E 0A 03 00 32 00 12 00")
        serial.feed(append_crc_be(bytes.fromhex("AA 55 00 16 0A 03 00 32 00 12 00") + bytes(9)))
        await asyncio.wait_for(write_task, timeout=0.5)

        serial.written.clear()
        read_task = asyncio.create_task(beam.flash_read(tile_id, address, 256))
        for _ in range(50):
            if len(serial.written) == 22:
                break
            await asyncio.sleep(0.005)
        assert bytes(serial.written[:11]) == bytes.fromhex("AA 55 00 16 0A 03 00 6B 00 12 00")
        response = append_crc_be(bytes.fromhex("AA 55 01 1E 0A 03 00 6B 00 12 00") + page + bytes(17))
        serial.feed(response)
        assert await asyncio.wait_for(read_task, timeout=0.5) == page
    finally:
        await beam.disconnect()


@pytest.mark.asyncio
async def test_simulated_turntable_has_matching_readback_and_rejects_negative_speed():
    turntable = SimulatedTurntable()
    await turntable.connect()
    await turntable.move_to(2, 3.25, 0.12345)
    await turntable.move_to(1, -2.5, 0.25)
    readback = await turntable.read_all_axes()
    assert readback["positions"]["2"] == pytest.approx(3.25)
    assert readback["positions"]["1"] == pytest.approx(-2.5)
    assert readback["velocities"]["2"] == 0.0
    with pytest.raises(ServiceError, match="不能为负数"):
        await turntable.move_to(2, 0, -1)


@pytest.mark.asyncio
async def test_real_vna_uses_internal_sweep_averaging_on_the_s_parameter_source_port():
    vna = RealVna("TCPIP0::192.168.1.100::hislip0::INSTR")
    instrument = FakeVnaInstrument()
    vna._instrument = instrument
    frequencies = np.array([8.0e9, 8.005e9, 8.01e9])

    readback = await vna.configure(
        s_parameter="S12",
        frequencies_hz=frequencies,
        if_bandwidth_hz=1000,
        source_power_dbm=-18,
        averaging_enabled=True,
        averaging_count=4,
    )
    await vna.acquire(frequencies)

    assert "SOUR1:POW2 -18" in instrument.writes
    assert "SENS1:AVER ON" in instrument.writes
    assert "SENS1:AVER:CLE" in instrument.writes
    assert "SENS1:SWE:GRO:COUN 4" in instrument.writes
    assert "SENS1:SWE:MODE GRO" in instrument.writes
    assert "DISP:WIND1:TRAC3:Y:AUTO" in instrument.writes
    assert readback["source_port"] == 2
    assert readback["averaging_count"] == 4
    assert readback["averaging_mode"] == "SWEEP"


class FeedHomeCompleteFlagStuckFake(FakeImac):
    def SHome(self, _device: int, axis: int) -> None:
        self.positions[axis] = 0.0

    def GetHomeComplete(self, _device: int, _axis: int) -> bool:
        return False


@pytest.mark.asyncio
async def test_feed_home_accepts_stable_zero_when_vendor_complete_flag_stays_false(tmp_path):
    turntable = RealTurntable(str(tmp_path / "ImacFxDll.dll"))
    fake = FeedHomeCompleteFlagStuckFake()
    fake.positions[4] = 12.0
    turntable._imac = fake

    result = await turntable.home(4, timeout_seconds=2)

    assert result["axis"] == 4
    assert result["position"] == 0.0
    assert result["velocity"] == 0.0
    assert result["home_complete"] is False
    assert result["completion_basis"] == "POSITION_AND_VELOCITY_STABLE"


@pytest.mark.asyncio
async def test_real_beam_manual_send_and_background_receive_are_independent(loaded_assets):
    registry, _, _ = loaded_assets
    events = EventBus()
    queue = events.subscribe()
    beam = RealBeamController(
        "COM_TEST",
        events=events,
        frame_decoder=registry.decode_matching_frame,
    )
    serial = FakeSerial()
    beam._serial = serial
    beam._reader_task = asyncio.create_task(beam._receive_loop())
    known = bytes.fromhex("AA 55 00 16 22 00 00 00 11 94 0B B8 00 00 00 04 00 00 04 00 E4 DF")

    try:
        await asyncio.wait_for(beam.send_only(known), timeout=0.1)
        assert bytes(serial.written) == known

        serial.feed(bytes.fromhex("DE AD BE EF") + known)
        received = []
        for _ in range(10):
            event = await asyncio.wait_for(queue.get(), timeout=0.2)
            received.append(event)
            if event.get("parsed"):
                break
        assert "DE AD BE EF" in " ".join(event["raw_hex"] for event in received)
        matched = next(event for event in received if event.get("parsed"))
        assert matched["parsed"]["fields"]
        assert "decoded" not in matched
    finally:
        await beam.disconnect()

    assert serial.closed


@pytest.mark.asyncio
async def test_real_beam_automatic_response_matches_crc_opcode_and_array_id():
    beam = RealBeamController("COM_TEST")
    serial = FakeSerial()
    beam._serial = serial
    beam._reader_task = asyncio.create_task(beam._receive_loop())
    request = append_crc_be(bytes.fromhex("AA 55 00 16 22 00") + bytes(14))
    unrelated = append_crc_be(bytes.fromhex("AA 55 00 16 22 01") + bytes(14))

    try:
        exchange = asyncio.create_task(beam.send_frame(request, timeout_ms=500, response_opcode=0x22))
        for _ in range(50):
            if len(serial.written) == len(request):
                break
            await asyncio.sleep(0.005)
        serial.feed(unrelated)
        await asyncio.sleep(0.05)
        assert not exchange.done()
        serial.feed(request)
        assert await asyncio.wait_for(exchange, timeout=0.5) == request
    finally:
        await beam.disconnect()


@pytest.mark.asyncio
async def test_automatic_run_control_blocks_manual_device_reconnect_and_disconnect():
    manager = DeviceManager(EventBus())
    await manager.connect("vna", DeviceSource.SIMULATED, {})
    await manager.acquire_run_control("RUN-1")
    try:
        with pytest.raises(ServiceError) as disconnect_error:
            await manager.disconnect("vna")
        assert disconnect_error.value.code == "CONTROL_LOCKED"
        with pytest.raises(ServiceError) as reconnect_error:
            await manager.connect("vna", DeviceSource.SIMULATED, {})
        assert reconnect_error.value.code == "CONTROL_LOCKED"
    finally:
        manager.release_run_control("RUN-1")
    assert (await manager.disconnect("vna"))["state"] == "DISCONNECTED"


class LowSpeedNearTargetFake(FakeImac):
    def __init__(self) -> None:
        super().__init__()
        self.stationary = False
        self.target = 0.0
        self.read_count = 0

    def MoveDeviceToPos(self, _device, _axis, target, _speed, _absolute):
        self.target = target

    def SHome(self, _device, _axis):
        self.target = 0.0

    def GetHomeComplete(self, _device, _axis):
        return True

    def Stop(self, _device, _axis):
        pass  # Deceleration is not complete until the test releases stationary.

    def GetPosStr(self, _device, axis):
        return str(self.target if self.stationary else self.target - 50) if axis == 1 else "0"

    def GetVelStr(self, _device, axis):
        self.read_count += 1
        return "50" if axis == 1 and not self.stationary else "0"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["move", "home", "stop"])
async def test_real_turntable_waits_for_actual_stillness_at_low_speed(action):
    turntable = RealTurntable("FAKE.dll")
    fake = LowSpeedNearTargetFake()
    turntable._imac = fake
    operation = (
        turntable.move_to(1, 1.0, 0.005, timeout_seconds=2)
        if action == "move" else turntable.home(1, timeout_seconds=2)
        if action == "home" else turntable.stop(1)
    )
    task = asyncio.create_task(operation)
    try:
        for _ in range(50):
            if fake.read_count:
                break
            await asyncio.sleep(0.005)
        # Both position error and actual velocity are .005: the old .01 threshold
        # wrongly accepted a motor still moving at the full requested low speed.
        await asyncio.sleep(0.25)
        assert not task.done()
        fake.stationary = True
        result = await asyncio.wait_for(task, timeout=1)
        assert result["velocities"]["1"] == 0
        assert fake.read_count >= 4
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_translation_home_uses_the_same_interlocks_as_absolute_motion():
    turntable = RealTurntable("FAKE.dll")
    fake = LowSpeedNearTargetFake()
    fake.i_values[500] = 0
    turntable._imac = fake
    with pytest.raises(ServiceError, match="未激活"):
        await turntable.home(7, timeout_seconds=1)
    assert fake.read_count == 0
    fake.i_values[500] = 1
    fake.m_values[543] = 1
    with pytest.raises(ServiceError, match="驱动故障"):
        await turntable.home(7, timeout_seconds=1)


@pytest.mark.asyncio
async def test_stopping_another_axis_does_not_cancel_current_motion():
    turntable = RealTurntable("FAKE.dll")
    fake = MovingFakeImac(duration=0.6)
    turntable._imac = fake
    task = asyncio.create_task(turntable.move_to(4, 1, 1, timeout_seconds=2))
    await asyncio.sleep(0.05)
    await turntable.stop(1)
    assert not task.done()
    assert turntable._motion_lock.locked()
    assert (await asyncio.wait_for(task, timeout=2))["position"] == 1


@pytest.mark.asyncio
async def test_vna_reads_platform_measurement_even_if_front_panel_selection_changes():
    vna = RealVna("FAKE")
    fake = FakeVnaInstrument()
    vna._instrument = fake
    frequencies = np.array([8e9, 8.005e9, 8.01e9])
    await vna.configure(
        s_parameter="S21", frequencies_hz=frequencies, if_bandwidth_hz=1000,
        source_power_dbm=-10, averaging_enabled=False, averaging_count=1,
    )
    fake.active_measurement = "my_trace1"
    values = await vna.acquire(frequencies)
    assert values.tolist() == [1 + 0j, 0.5 - 0.25j, 1j]
    assert "CALC1:MEAS7:DATA:SDATA?" in fake.queries
    assert "CALC1:MEAS7:X?" in fake.queries
    assert "CALC1:DATA? SDATA" not in fake.queries
    fake.measurement_name = "REPLACED_MEASUREMENT"
    with pytest.raises(ServiceError, match="采集失败"):
        await vna.acquire(frequencies)


@pytest.mark.asyncio
async def test_vna_extends_timeout_for_a_slow_single_sweep_without_averaging():
    class SlowSweepFake(FakeVnaInstrument):
        observed_scan_timeout = None

        def query(self, command):
            if command == "SENS1:SWE:TIME?":
                return "15.0"
            if command == "*OPC?" and "SENS1:SWE:MODE SING" in self.writes:
                self.observed_scan_timeout = self.timeout
                if self.timeout < 15000:
                    raise TimeoutError("15-second scan exceeds VISA timeout")
            return super().query(command)

    vna = RealVna("FAKE")
    fake = SlowSweepFake()
    vna._instrument = fake
    frequencies = np.array([8e9, 8.005e9, 8.01e9])
    await vna.configure(
        s_parameter="S21", frequencies_hz=frequencies, if_bandwidth_hz=1000,
        source_power_dbm=-10, averaging_enabled=False, averaging_count=1,
    )
    await vna.acquire(frequencies)
    assert fake.observed_scan_timeout >= 20000
    assert fake.timeout == 10000


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt_prefix", [b"\xAA\x55\x10\x00", b"\xAA\x55\x01\x1e", b"\xAA\x55\x00\x16\xff"])
async def test_serial_recovers_valid_reply_after_bad_length_and_logs_bytes_once(corrupt_prefix):
    events = EventBus()
    queue = events.subscribe()
    beam = RealBeamController("FAKE", events=events)
    serial = FakeSerial()
    beam._serial = serial
    beam._reader_task = asyncio.create_task(beam._receive_loop())
    request = append_crc_be(bytes.fromhex("AA 55 00 16 22 00") + bytes(14))
    try:
        task = asyncio.create_task(beam.send_frame(request, timeout_ms=1000))
        for _ in range(50):
            if serial.written:
                break
            await asyncio.sleep(0.005)
        serial.feed(corrupt_prefix + request)
        event = await asyncio.wait_for(queue.get(), timeout=0.2)
        assert bytes.fromhex(event["raw_hex"]) == corrupt_prefix + request
        assert await asyncio.wait_for(task, timeout=1) == request
        await asyncio.sleep(0.02)
        assert queue.empty()  # parsing never duplicates the physical RX byte log
    finally:
        await beam.disconnect()


def _multi_response(index: int) -> bytes:
    return append_crc_be(bytes.fromhex("AA 55 00 16 1A 02") + bytes([0, 3, index]) + bytes(11))


@pytest.mark.asyncio
async def test_real_serial_collects_all_response_frames_after_one_request():
    beam = RealBeamController("FAKE")
    serial = FakeSerial()
    beam._serial = serial
    beam._reader_task = asyncio.create_task(beam._receive_loop())
    rule = ResponseRule("POWER", "POWER_QUERY", 0x1A, "MULTI", 0, 2, 3, 4, 3, True, 500)
    request = append_crc_be(bytes.fromhex("AA 55 00 16 1A 02") + bytes(14))
    task = asyncio.create_task(beam.send_frame(request, timeout_ms=500, response_rule=rule))
    try:
        for _ in range(50):
            if serial.written:
                break
            await asyncio.sleep(0.005)
        serial.feed(_multi_response(1))
        await asyncio.sleep(0.03)
        assert not task.done()
        serial.feed(_multi_response(2) + _multi_response(3))
        assert await asyncio.wait_for(task, timeout=0.5) == [_multi_response(index) for index in (1, 2, 3)]
        assert bytes(serial.written) == request
    finally:
        await beam.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
async def test_real_serial_rejects_out_of_order_or_incomplete_group_without_retry(missing):
    beam = RealBeamController("FAKE")
    serial = FakeSerial()
    beam._serial = serial
    beam._reader_task = asyncio.create_task(beam._receive_loop())
    rule = ResponseRule("POWER", "POWER_QUERY", 0x1A, "MULTI", 0, 2, 3, 4, 3, True, 100)
    request = append_crc_be(bytes.fromhex("AA 55 00 16 1A 02") + bytes(14))
    task = asyncio.create_task(beam.send_frame(request, timeout_ms=500, response_rule=rule))
    try:
        for _ in range(50):
            if serial.written:
                break
            await asyncio.sleep(0.005)
        serial.feed(_multi_response(1) if missing else _multi_response(3))
        with pytest.raises(ServiceError) as error:
            await asyncio.wait_for(task, timeout=0.5)
        assert error.value.code == ("TIMEOUT" if missing else "DATA_INTEGRITY")
        assert bytes(serial.written) == request
        assert not beam._reader_task.done()
    finally:
        await beam.disconnect()
