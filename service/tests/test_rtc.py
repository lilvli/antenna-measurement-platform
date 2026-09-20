from __future__ import annotations

import asyncio
import time

import pytest

from antenna_service.errors import ServiceError
from antenna_service.devices.base import RtcDeviceAdapter
from antenna_service.models import DeviceSource, DeviceState
from antenna_service.protocol.profile import ResponseRule
from antenna_service.protocol.crc import append_crc_be
from antenna_service.protocol.rtc import (
    OPCODES, RtcClient, RtcEndpoint, RtcFrameDecoder, SimulatedRtcClient,
    _SimulatedSerial, build_fixed, parse_frame,
)

WAVE = bytes.fromhex("AA 55 00 16 22 00 00 00 11 94 0B B8 00 00 00 04 00 00 04 00 E4 DF")


async def configure(client, *, mode="software", waves=1, points=2, host_timeout_ms=1000, vna_timeout_ms=100):
    await client.set_timing(host_timeout_ms=host_timeout_ms, vna_ready_timeout_ms=vna_timeout_ms,
                            antenna_tx_timeout_ms=50, ack_timeout_ms=50, sync_io_timeout_ms=50)
    await client.set_counts(waves, points)
    await client.set_trigger_mode(mode)
    await client.set_tr_config("TX", 100, 20, 1)
    for address in range(1, waves + 1):
        await client.write_wave_entry(address, WAVE)
        assert await client.read_wave_entry(address) == WAVE


async def wait_state(client, state, *, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = await client.get_status()
        if result["state"] == state:
            return result
        await asyncio.sleep(0.002)
    pytest.fail(f"RTC 未到达 {state}: {result}")


def test_fixed_v1_vectors_and_reserved_long_opcodes():
    assert build_fixed(0x30, data=WAVE)[-2:] == bytes.fromhex("76 D1")
    assert build_fixed(0x15)[-2:] == bytes.fromhex("1A 76")
    assert build_fixed(0x16)[-2:] == bytes.fromhex("FE F4")
    assert build_fixed(0xE2, control=22, aux=1, data=WAVE)[-2:] == bytes.fromhex("2D C4")
    assert build_fixed(0x0D)[-2:] == bytes.fromhex("4E 81")
    assert 0x31 not in OPCODES.values() and 0x33 not in OPCODES.values()
    with pytest.raises(ServiceError):
        parse_frame(build_fixed(0xE2, control=0x96))
    short = append_crc_be(bytes.fromhex("A5 5A 00 0A 83 00 00 00"))
    with pytest.raises(ServiceError):
        parse_frame(short)


def test_stream_recovery_fragmented_coalesced_embedded_header_and_bad_crc():
    decoder = RtcFrameDecoder(0.1)
    inner = b"\xA5\x5A\x00\x20" + bytes(18)
    good = build_fixed(0xE2, control=22, data=inner)
    bad = good[:-1] + bytes([good[-1] ^ 1])
    assert decoder.feed(b"noise\xa5", now=0) == []
    assert decoder.feed(b"\x5a\x00\x20" + good[4:13], now=0.01) == []
    assert decoder.feed(good[13:] + bad + good + good[:5], now=0.02) == [good, good]
    assert decoder.feed(good[5:], now=0.03) == [good]
    assert decoder.bad_frames >= 1
    assert len(decoder.buffer) < 32


def test_half_frame_expiry_searches_cached_bytes_for_next_header():
    decoder = RtcFrameDecoder(0.1)
    good = build_fixed(0x83)
    assert decoder.feed(good[:8], now=1) == []
    assert decoder.feed(good, now=1.2) == [good]
    assert decoder.bad_frames == 1


@pytest.mark.asyncio
async def test_software_group_rdy_counts_and_single_done_then_rearm():
    client = SimulatedRtcClient()
    try:
        assert (await client.connect())["protocol"] == "1.0"
        await configure(client, waves=2, points=3)
        await client.arm()
        await client.software_trigger()
        status = await wait_state(client, "COMPLETE")
        assert not status["tr_running"] and not status["io_busy"]
        p = await client.get_progress()
        assert (p["accepted_groups"], p["completed_groups"], p["valid_triggers"], p["completed_points"]) == (1, 1, 6, 6)
        done = [e for e in client.received_events if e["event"] == "DONE"]
        assert len(done) == 1 and done[0]["reason"] == 0
        assert client.firmware.rdy_trace == [(4, 1), (7, 0), (8, 1)] * 6
        with pytest.raises(ServiceError, match="INVALID_STATE"):
            await client.software_trigger()
        await client.arm()
        assert (await client.get_progress())["completed_points"] == 0
        await client.stop_graceful()
        assert (await client.get_last_time())["last_result"] == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_continuous_groups_silent_and_rearm_resets_only_on_arm():
    client = SimulatedRtcClient()
    try:
        await client.connect()
        await configure(client, mode="continuous", waves=0, points=2)
        await client.arm()
        for index in range(3):
            assert await client.external_trigger()
            await wait_state(client, "ARMED")
            p = await client.get_progress()
            assert p["completed_groups"] == index + 1
        assert (await client.get_progress())["completed_points"] == 6
        assert not [e for e in client.received_events if e["event"] == "DONE"]
        assert not client.firmware.tx_frames
        await client.arm()
        assert (await client.get_progress())["accepted_groups"] == 0
        await client.stop_graceful()
        await wait_state(client, "COMPLETE")
        assert [e["reason"] for e in client.received_events if e["event"] == "DONE"] == [1]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_busy_external_pulse_latches_first_fault_without_queue_or_recovery_on_query():
    client = SimulatedRtcClient(point_delay_s=0.02)
    try:
        await client.connect()
        await configure(client, mode="continuous")
        await client.arm()
        assert await client.external_trigger()
        assert not await client.external_trigger()
        status = await wait_state(client, "FAULT")
        assert status["fault_code"] == 0x13
        assert not await client.external_trigger()
        await asyncio.sleep(0.03)
        p = await client.get_progress()
        assert p["accepted_groups"] == 1 and p["completed_groups"] == 0
        assert len([e for e in client.received_events if e["event"] == "FAULT"]) == 1
        with pytest.raises(ServiceError):
            await client.stop_immediate()
        client.firmware.rdy = 1  # Test fixture simulates completed external recovery.
        await client.clear_fault()
        assert (await client.get_status())["config_valid"] == 0
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior,stage,triggers", [("low_at_start", 4, 0), ("no_low", 7, 0), ("no_high", 8, 1)])
async def test_missing_rdy_transition_does_not_count_false_completion(behavior, stage, triggers):
    client = SimulatedRtcClient(point_delay_s=0.001)
    try:
        await client.connect()
        await configure(client, points=1, vna_timeout_ms=20)
        client.firmware.rdy_behavior = behavior
        await client.arm()
        await client.software_trigger()
        status = await wait_state(client, "FAULT")
        assert status["stage"] == stage and status["fault_code"] == 0x0B
        p = await client.get_progress()
        assert p["valid_triggers"] == triggers and p["completed_points"] == 0
        assert p["accepted_groups"] == 1 and p["completed_groups"] == 0
        assert len([r for r in client.firmware.sent if r[4] == 8]) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_debug_tr_configuration_send_and_rx_are_independent():
    observed = []
    client = SimulatedRtcClient(on_event=observed.append)
    try:
        await client.connect()
        await configure(client, waves=0)
        tr = await client.get_tr_config()
        assert (tr["mode"], tr["period_us"], tr["high_us"], tr["delay_us"], tr["tr_state"]) == ("TX", 100, 20, 1, 0)
        await client.start_debug_tr()
        client.firmware.rdy = 0
        await asyncio.sleep(0.03)
        skipped = (await client.get_antenna_io_status())["debug_pulses"]
        await asyncio.sleep(0.03)
        assert (await client.get_antenna_io_status())["debug_pulses"] == skipped
        result = await client.transfer_antenna(WAVE)
        assert result["transmitted_length"] == 22 and "response" not in result
        assert (await client.get_tr_config())["tr_state"] == 2
        assert not [e for e in client.received_events if e["event"] == "ANTENNA_RX"]
        client.inject_antenna_rx(bytes(22))
        client.inject_antenna_rx(WAVE)
        await asyncio.sleep(0.02)
        assert [e["sequence"] for e in client.received_events if e["event"] == "ANTENNA_RX"] == [1, 2]
        assert (await client.get_antenna_io_status())["tx_state"] == 2
        assert (await client.get_progress())["completed_points"] == 0
        client.firmware.rdy = 1
        await asyncio.sleep(0.02)
        assert (await client.get_antenna_io_status())["debug_pulses"] > skipped
        await client.stop_debug_tr()
        assert (await client.get_tr_config())["tr_state"] == 0
        assert len(client.firmware.tx_frames) == 1
        tx = [e for e in observed if e.get("direction") == "TX" and e["event"] == "raw"]
        assert len(tx) == len(client.firmware.sent)
        rx = [e for e in observed if e.get("direction") == "RX" and e["event"] == "raw"]
        assert len(rx) >= len(tx)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_activity_status_continues_during_blocking_vna_wait_without_periodic_ping():
    client = SimulatedRtcClient()
    try:
        await client.connect()
        await configure(client, mode="continuous", host_timeout_ms=100)
        await client.arm()
        count = sum(r[4] == 3 for r in client.firmware.sent)
        time.sleep(0.3)  # Represents an existing blocking VNA call on the event loop.
        assert sum(r[4] == 3 for r in client.firmware.sent) > count
        assert sum(r[4] == 14 for r in client.firmware.sent) == 1
        assert (await client.get_status())["state"] == "ARMED"
        await client.stop_graceful()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_unknown_action_allows_query_but_blocks_another_action_without_replay():
    transport = None
    def factory(**kwargs):
        nonlocal transport
        transport = _SimulatedSerial(**kwargs)
        emit = transport._emit
        def dropping_emit(opcode, *args, **kw):
            if opcode != 0x88:
                emit(opcode, *args, **kw)
        transport._emit = dropping_emit
        return transport
    client = RtcClient(RtcEndpoint("FAKE", timeout_ms=40), serial_factory=factory)
    try:
        await client.connect()
        await configure(client, points=1)
        await client.arm()
        with pytest.raises(ServiceError) as error:
            await client.software_trigger()
        assert error.value.side_effect_possible
        assert (await client.get_status())["state"] == "COMPLETE"
        assert (await client.get_progress())["completed_groups"] == 1
        with pytest.raises(ServiceError, match="上一动作结果未知"):
            await client.arm()
        assert len([r for r in transport.sent if r[4] == 8]) == 1
        client.acknowledge_unknown_result()
        await client.arm()
        await client.stop_graceful()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_read_queries_can_complete_while_physical_tx_ack_is_pending():
    transport = None
    held = []
    def factory(**kwargs):
        nonlocal transport
        transport = _SimulatedSerial(**kwargs)
        emit = transport._emit
        def holding_emit(opcode, *args, **kw):
            if opcode == 0xB0:
                held.append((opcode, args, kw))
            else:
                emit(opcode, *args, **kw)
        transport._emit = holding_emit
        transport.release = lambda: emit(held[0][0], *held[0][1], **held[0][2])
        return transport
    client = RtcClient(RtcEndpoint("FAKE", timeout_ms=100), serial_factory=factory)
    try:
        await client.connect()
        await configure(client, waves=0)
        action = asyncio.create_task(client.transfer_antenna(WAVE))
        while not held:
            await asyncio.sleep(0.001)
        assert not action.done()
        assert (await client.get_antenna_io_status())["tx_state"] == 2
        await client.ping()
        transport.release()
        assert (await action)["transmitted_length"] == 22
        assert len([r for r in transport.sent if r[4] == 0x30]) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_immediate_stop_keeps_partial_counts_and_uncertain_result():
    client = SimulatedRtcClient(point_delay_s=0.015)
    try:
        await client.connect()
        await configure(client, points=20)
        await client.arm()
        await client.software_trigger()
        await asyncio.sleep(0.06)
        await client.stop_immediate()
        status = await wait_state(client, "COMPLETE")
        p = await client.get_progress()
        assert status["result_uncertain"] and status["last_result"] == 3
        assert 0 < p["completed_points"] < 20 and p["completed_groups"] == 0
        assert not status["tr_running"] and not status["io_busy"]
        assert [e["reason"] for e in client.received_events if e["event"] == "DONE"] == [2]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tr_uses_reported_clock_and_rejects_width_duty_window_before_send():
    client = SimulatedRtcClient(clock_hz=50_000_000)
    try:
        await client.connect()
        await client.set_timing()
        await client.set_tr_config("TX", 100, 20, 1)
        tr = await client.get_tr_config()
        assert (tr["period_ticks"], tr["high_ticks"], tr["delay_ticks"]) == (5000, 1000, 50)
        count = len(client.firmware.sent)
        for values in [(100, 31, 1), (100, 20, 19), (1000, 656, 1), (100, 20, float("nan"))]:
            with pytest.raises(ServiceError):
                await client.set_tr_config("TX", *values)
        assert len(client.firmware.sent) == count
        with pytest.raises(ServiceError):
            await client.transact(build_fixed(0x31))
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_quantized_high_width_cannot_drop_below_one_microsecond():
    client = SimulatedRtcClient(clock_hz=1_400_000)
    try:
        await client.connect()
        await client.set_timing()
        before = len(client.firmware.sent)
        # RX has a large phase window, so this isolates the engineering minimum:
        # 1 us rounds to one tick, which is only 0.714 us at the reported clock.
        with pytest.raises(ServiceError, match="高宽"):
            await client.set_tr_config("RX", 100, 1, 0)
        assert len(client.firmware.sent) == before
        client.firmware.tr = bytes([2, 0]) + (140).to_bytes(3, "big") + (1).to_bytes(2, "big") + bytes(2) + b"\x01"
        with pytest.raises(ServiceError, match="TR 回读"):
            await client.get_tr_config()
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("opcode,index,value,action", [
    (0x81, 16, 1, "get_info"), (0x83, 5, 1, "get_status"),
    (0x8D, 29, 0x10, "get_progress"), (0x92, 9, 1, "get_tr_config"),
    (0x92, 8, 3, "get_tr_config"), (0x86, 29, 1, "set_trigger_mode"),
])
async def test_crc_valid_response_reserved_or_mode_fields_are_rejected(opcode, index, value, action):
    client = SimulatedRtcClient()
    try:
        await client.connect()
        emit = client.firmware._emit
        def corrupting_emit(op, data=b"", *, control=0, aux=0):
            if op == opcode:
                frame = bytearray(build_fixed(op, data=data, control=control, aux=aux))
                frame[index] = value
                client.firmware.incoming.put(append_crc_be(bytes(frame[:30])))
            else:
                emit(op, data, control=control, aux=aux)
        client.firmware._emit = corrupting_emit
        with pytest.raises(ServiceError, match="保留位或枚举") as captured:
            if action == "set_trigger_mode":
                await client.set_trigger_mode("software")
            else:
                await getattr(client, action)()
        assert captured.value.code == "DATA_INTEGRITY"
        if action == "set_trigger_mode":
            assert captured.value.side_effect_possible and client.unknown_result["opcode"] == 6
        else:
            assert client.unknown_result is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_idle_reader_disconnect_notifies_once_and_blocks_further_writes():
    observed = []
    transport = None
    class BrokenSerial(_SimulatedSerial):
        broken = False
        def read(self, length):
            if self.broken:
                raise OSError("test unplug")
            return super().read(length)
    def factory(**kwargs):
        nonlocal transport
        transport = BrokenSerial(**kwargs)
        return transport
    client = RtcClient(RtcEndpoint("FAKE", timeout_ms=50), serial_factory=factory, on_event=observed.append)
    try:
        await client.connect()
        transport.broken = True
        deadline = time.monotonic() + 1
        while client._transport_error is None and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        assert client._transport_error is not None
        await asyncio.sleep(0.01)
        errors = [event for event in observed if event["event"] == "transport_error"]
        assert len(errors) == 1 and errors[0]["error"]["code"] == "DEVICE_DISCONNECTED"
        before = len(transport.sent)
        with pytest.raises(ServiceError) as captured:
            await client.get_status()
        assert captured.value.code == "DEVICE_DISCONNECTED"
        assert len(transport.sent) == before
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_idle_and_completed_connections_send_no_periodic_commands():
    client = SimulatedRtcClient()
    try:
        await client.connect()
        await configure(client, host_timeout_ms=80)
        count = len(client.firmware.sent)
        await asyncio.sleep(0.12)
        assert len(client.firmware.sent) == count
        assert sum(frame[4] == 14 for frame in client.firmware.sent) == 1
        await client.ping()  # Explicit connection check remains available.
        assert sum(frame[4] == 14 for frame in client.firmware.sent) == 2
        await client.arm()
        await client.stop_graceful()
        await wait_state(client, "COMPLETE")
        count = len(client.firmware.sent)
        await asyncio.sleep(0.12)
        assert len(client.firmware.sent) == count
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_successful_foreground_queries_postpone_activity_status_reads():
    client = SimulatedRtcClient()
    try:
        await client.connect()
        await configure(client, mode="continuous", host_timeout_ms=200)
        await client.arm()
        count = sum(frame[4] == 3 for frame in client.firmware.sent)
        for _ in range(12):
            await client.get_progress()
            await asyncio.sleep(0.008)
        assert sum(frame[4] == 3 for frame in client.firmware.sent) == count
        assert sum(frame[4] == 14 for frame in client.firmware.sent) == 1
        await client.stop_graceful()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_debug_tr_uses_status_monitor_and_monitor_reports_latched_fault():
    observed = []
    client = SimulatedRtcClient(on_event=observed.append)
    try:
        await client.connect()
        await configure(client, waves=0, host_timeout_ms=80)
        await client.start_debug_tr()
        await asyncio.sleep(0.2)
        assert client.firmware.state == 9 and not client.firmware.fault
        assert sum(frame[4] == 3 for frame in client.firmware.sent) > 1
        assert sum(frame[4] == 14 for frame in client.firmware.sent) == 1
        emit = client.firmware._emit
        client.firmware._emit = lambda opcode, *args, **kwargs: None if opcode == 0xE1 else emit(opcode, *args, **kwargs)
        with client.firmware.lock:
            client.firmware._fault(0x13, 11)
        deadline = time.monotonic() + 1
        while not any(event.get("detected_by") == "activity_status" for event in observed):
            assert time.monotonic() < deadline
            await asyncio.sleep(0.005)
        faults = [event for event in observed if event.get("detected_by") == "activity_status"]
        assert len(faults) == 1 and faults[0]["fault_code"] == 0x13
        count = len(client.firmware.sent)
        await asyncio.sleep(0.08)
        assert len(client.firmware.sent) == count  # Drained fault is no longer active.
    finally:
        await client.close()


def rtc_adapter():
    adapter = RtcDeviceAdapter(DeviceSource.SIMULATED)
    adapter.client = SimulatedRtcClient()
    return adapter


def antenna_reply(*, opcode=WAVE[4], array_id=WAVE[5], payload=None):
    raw = bytearray(WAVE[:20])
    raw[4], raw[5] = opcode, array_id
    if payload is not None:
        raw[6:20] = payload
    return append_crc_be(bytes(raw))


@pytest.mark.asyncio
async def test_antenna_exchange_excludes_old_rx_and_b0_never_substitutes_response():
    adapter = rtc_adapter()
    try:
        await adapter.connect()
        await configure(adapter.client, waves=0)
        adapter.client.inject_antenna_rx(WAVE)
        deadline = time.monotonic() + 1
        while adapter.client.antenna_rx_cursor() == 0:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.002)
        with pytest.raises(ServiceError) as captured:
            await adapter.send_frame(WAVE, timeout_ms=25)
        assert captured.value.code == "TIMEOUT" and captured.value.side_effect_possible
        assert adapter.snapshot.state == DeviceState.UNKNOWN
        assert len(adapter.client.firmware.tx_frames) == 1
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_antenna_exchange_keeps_early_e2_and_matches_full_request_echo():
    adapter = rtc_adapter()
    try:
        await adapter.connect()
        await configure(adapter.client, waves=0)
        client = adapter.client
        emit = client.firmware._emit
        wrong_echo = antenna_reply(payload=bytes(14))
        def early_emit(opcode, *args, **kwargs):
            if opcode == 0xB0:
                client.firmware.inject_antenna_rx(antenna_reply(opcode=0x31))
                client.firmware.inject_antenna_rx(antenna_reply(array_id=1))
                client.firmware.inject_antenna_rx(wrong_echo)
                client.firmware.inject_antenna_rx(WAVE[:-1] + bytes([WAVE[-1] ^ 1]))
                client.firmware.inject_antenna_rx(WAVE)
            emit(opcode, *args, **kwargs)
        client.firmware._emit = early_emit
        result = await adapter.send_frame(WAVE, timeout_ms=100, response_matcher=lambda frame: frame == WAVE)
        assert result == WAVE
        assert len(client.firmware.tx_frames) == 1
        assert client.antenna_rx_cursor() == 5
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_explicit_simulated_response_travels_through_independent_e2():
    adapter = rtc_adapter()
    try:
        await adapter.connect()
        await configure(adapter.client, waves=0)
        adapter.client.queue_antenna_response(WAVE, [WAVE])
        assert await adapter.send_frame(WAVE, timeout_ms=100) == WAVE
        assert len(adapter.client.firmware.tx_frames) == 1
        assert [event["payload_hex"] for event in adapter.client.received_events if event["event"] == "ANTENNA_RX"] == [WAVE.hex(" ").upper()]
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("indices,expected_error", [([1, 2], None), ([1, 1], "DATA_INTEGRITY"), ([2, 1], "DATA_INTEGRITY"), ([1], "TIMEOUT")])
async def test_rtc_antenna_exchange_assembles_configured_multiframe_response(indices, expected_error):
    adapter = rtc_adapter()
    try:
        await adapter.connect()
        await configure(adapter.client, waves=0)
        rule = ResponseRule("RX", "test", 0x22, "MULTI", 2, 1, 2, 3, 2, True, 100)
        replies = [antenna_reply(payload=bytes([2, index]) + bytes(12)) for index in indices]
        adapter.client.queue_antenna_response(WAVE, replies)
        if expected_error:
            with pytest.raises(ServiceError) as captured:
                await adapter.send_frame(WAVE, timeout_ms=40, response_rule=rule)
            assert captured.value.code == expected_error and captured.value.side_effect_possible
        else:
            assert await adapter.send_frame(WAVE, timeout_ms=100, response_rule=rule) == replies
        assert len(adapter.client.firmware.tx_frames) == 1
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_local_rx_cursor_survives_firmware_sequence_wrap():
    adapter = rtc_adapter()
    try:
        await adapter.connect()
        await configure(adapter.client, waves=0)
        adapter.client.firmware.rx_sequence = 65535
        adapter.client.queue_antenna_response(WAVE, [WAVE])
        assert await adapter.send_frame(WAVE, timeout_ms=100) == WAVE
        event = [event for event in adapter.client.received_events if event["event"] == "ANTENNA_RX"][-1]
        assert event["sequence"] == 0 and event["receive_cursor"] > 0
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_status_monitor_remains_active_while_tx_ack_is_pending():
    client = SimulatedRtcClient()
    try:
        await client.connect()
        await configure(client, waves=0, host_timeout_ms=80)
        emit = client.firmware._emit
        held = []
        def holding_emit(opcode, *args, **kwargs):
            if opcode == 0xB0:
                held.append((opcode, args, kwargs))
            else:
                emit(opcode, *args, **kwargs)
        client.firmware._emit = holding_emit
        task = asyncio.create_task(client.transfer_antenna(WAVE))
        while not held:
            await asyncio.sleep(0.002)
        before = sum(frame[4] == 3 for frame in client.firmware.sent)
        await asyncio.sleep(0.12)
        assert sum(frame[4] == 3 for frame in client.firmware.sent) > before
        assert not task.done() and sum(frame[4] == 14 for frame in client.firmware.sent) == 1
        emit(held[0][0], *held[0][1], **held[0][2])
        assert (await task)["tx_state"] == "COMPLETED"
        before = len(client.firmware.sent)
        await asyncio.sleep(0.08)
        assert len(client.firmware.sent) == before
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_receive_overflow_prevents_false_antenna_success():
    client = SimulatedRtcClient()
    try:
        await client.connect()
        cursor = client.antenna_rx_cursor()
        for _ in range(1025):
            client.inject_antenna_rx(WAVE)
        deadline = time.monotonic() + 2
        while client.antenna_rx_cursor() < cursor + 1025:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.002)
        with pytest.raises(ServiceError) as captured:
            await client.wait_antenna_response(cursor, lambda frame: frame == WAVE, timeout_ms=25)
        assert captured.value.code == "DATA_INTEGRITY"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_disconnect_wakes_pending_antenna_response_without_resend():
    adapter = rtc_adapter()
    try:
        await adapter.connect()
        await configure(adapter.client, waves=0)
        task = asyncio.create_task(adapter.send_frame(WAVE, timeout_ms=5000))
        while not adapter.client.firmware.tx_frames:
            await asyncio.sleep(0.002)
        await asyncio.sleep(0.01)
        await adapter.client.close()
        with pytest.raises(ServiceError) as captured:
            await asyncio.wait_for(task, timeout=0.5)
        assert captured.value.code == "DEVICE_DISCONNECTED" and captured.value.side_effect_possible
        assert len(adapter.client.firmware.tx_frames) == 1
    finally:
        await adapter.client.close()
