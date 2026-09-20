from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from antenna_service import __version__
from antenna_service.coordinates import CoordinateLoader, validate_profile_coordinate_pair
from antenna_service.devices.manager import DeviceManager
from antenna_service.devices.serial_ports import discover_serial_ports
from antenna_service.errors import ServiceError
from antenna_service.events import EventBus
from antenna_service.models import (
    CompensationRequest,
    DataViewRequest,
    DeviceCommandRequest,
    DeviceConnectRequest,
    FlashPrepareRequest,
    FlashWriteRequest,
    LoadPathRequest,
    RunPlan,
    RtcWaveRequest,
)
from antenna_service.protocol.profile import AssetRegistry, ProfileLoader
from antenna_service.storage.hdf5_store import inspect_data_file, read_data_view
from antenna_service.workflows.compensation import CompensationService
from antenna_service.workflows.engine import RunEngine
from antenna_service.workflows.rtc import compile_rtc_wave_table, read_rtc_wave_table
from antenna_service.workflows.flash import FlashService


class CompileRequest(BaseModel):
    profile_id: str
    command_id: str
    array_id: int = Field(default=0, ge=0, le=255)
    parameters: dict[str, Any] = Field(default_factory=dict)
    transport: Literal["SERIAL", "RTC"] = "SERIAL"


class DecodeRequest(BaseModel):
    profile_id: str
    frame_hex: str


events = EventBus()
assets = AssetRegistry()
devices = DeviceManager(events, frame_decoder=assets.decode_matching_frame)
runs = RunEngine(assets, devices, events)
compensation = CompensationService()
flash = FlashService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    devices.shutdown_requested = True
    record = runs.current()
    if record and record.task and not record.task.done():
        if record.state.value in {"RUNNING", "PAUSED"}:
            await runs.stop(record.run_id)
        await record.task
    for adapter in list(devices.devices.values()):
        try:
            await adapter.disconnect()
        except Exception:
            pass


app = FastAPI(title="Antenna Control Service", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_origin_regex=r"http://(?:localhost|127\.0\.0\.1):\d+",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ServiceError)
async def service_error_handler(_: Request, exc: ServiceError) -> JSONResponse:
    status = 404 if exc.code == "NOT_FOUND" else 409 if exc.code in {"CONTROL_LOCKED", "NOT_RUNNABLE"} else 400
    return JSONResponse(status_code=status, content={"error": exc.as_dict()})


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"status": "healthy", "service_version": __version__, "api_version": "1.0"}


@app.post("/api/assets/profile/load")
async def load_profile(request: LoadPathRequest) -> dict[str, Any]:
    profile = assets.add_profile(ProfileLoader().load(request.path))
    await events.publish("asset.loaded", asset_type="profile", asset=profile.summary())
    return profile.summary()


@app.post("/api/assets/coordinates/load")
async def load_coordinates(request: LoadPathRequest) -> dict[str, Any]:
    coordinates = assets.add_coordinates(CoordinateLoader().load(request.path))
    await events.publish("asset.loaded", asset_type="coordinates", asset=coordinates.summary())
    return coordinates.summary()


@app.post("/api/assets/validate-pair")
async def validate_pair(profile_id: str, coordinate_id: str) -> dict[str, Any]:
    profile = assets.profile(profile_id)
    coordinates = assets.coordinate(coordinate_id)
    validate_profile_coordinate_pair(profile, coordinates)
    return {"valid": True, "profile": profile.summary(), "coordinates": coordinates.summary()}


@app.get("/api/assets")
async def list_assets() -> dict[str, Any]:
    return {
        "profiles": [profile.summary() for profile in assets.profiles.values()],
        "coordinates": [coordinates.summary() for coordinates in assets.coordinates.values()],
    }


@app.post("/api/protocol/compile")
async def compile_command(request: CompileRequest) -> dict[str, Any]:
    profile = assets.profile(request.profile_id)
    frame = profile.encode(request.command_id, request.array_id, request.parameters)
    return {"frame_hex": frame.hex(" ").upper(), "length": len(frame), "decoded": profile.decode(frame)}


@app.post("/api/protocol/decode")
async def decode_frame(request: DecodeRequest) -> dict[str, Any]:
    profile = assets.profile(request.profile_id)
    try:
        frame = bytes.fromhex(request.frame_hex)
    except ValueError as exc:
        raise ServiceError("INVALID_REQUEST", "HEX 字符串格式错误", "profile_decode", profile.profile_id) from exc
    return profile.decode(frame)


@app.post("/api/devices/beam_controller/send")
async def send_beam_command(request: CompileRequest) -> dict[str, Any]:
    async with devices.manual_control("beam_debug"):
        return await _send_beam_command_unlocked(request)


async def _send_beam_command_unlocked(request: CompileRequest) -> dict[str, Any]:
    """Compile and write one profile command; receive processing remains independent."""
    profile = assets.profile(request.profile_id)
    command = profile.commands.get(request.command_id)
    if command is None:
        raise ServiceError("NOT_FOUND", "配置包中不存在所选指令", "beam_debug", request.command_id)
    frame = profile.encode(request.command_id, request.array_id, request.parameters)
    if request.transport == "RTC":
        if len(frame) != 22:
            raise ServiceError("NOT_RUNNABLE", "RTC V1.0只转发完整22字节，FLASH长帧扩展未启用", "rtc_debug")
        adapter = devices.require("rtc")
        devices.note_configuration_change("rtc")
        # B0 proves physical J30J TX completion only. E2 is independently received
        # and decoded by the adapter; no waveform command is retried for missing RX.
        timing = await getattr(adapter, "get_timing")()
        if not timing.get("sync_io_timeout_ms") or not timing.get("antenna_tx_timeout_ms"):
            await getattr(adapter, "set_timing")()
        result = await getattr(adapter, "transfer_antenna")(frame)
        from antenna_service.protocol.rtc import build_fixed
        return {"command_id": command.command_id, "tx_hex": frame.hex(" ").upper(),
                "rtc_frame_hex": build_fixed(0x30, data=frame).hex(" ").upper(),
                "length": 22, "written": True, "transport": "RTC", "rtc_result": result}
    adapter = devices.require("beam_controller")
    devices.note_configuration_change("beam_controller")
    if not hasattr(adapter, "send_only"):
        raise ServiceError("NOT_RUNNABLE", "当前波控适配器不支持独立发送", "beam_debug")
    try:
        # Manual protocol debugging is fire-and-forget. The serial background reader logs
        # every later RX message and independently decides whether a loaded profile matches.
        await getattr(adapter, "send_only")(frame)
    except Exception as exc:
        await events.publish(
            "device.raw",
            device_id="beam_controller",
            direction="ERROR",
            command_id=command.command_id,
            raw_hex="",
            context="MANUAL_DEBUG",
            error=str(exc),
        )
        raise
    await events.publish(
        "device.raw",
        device_id="beam_controller",
        direction="TX",
        command_id=command.command_id,
        raw_hex=frame.hex(" ").upper(),
        transport="SERIAL" if adapter.snapshot.source.value == "REAL" else "SIMULATED",
        context="MANUAL_DEBUG",
    )
    return {
        "command_id": command.command_id,
        "tx_hex": frame.hex(" ").upper(),
        "length": len(frame),
        "written": True,
    }


@app.post("/api/devices/rtc/waves/preview")
async def preview_rtc_waves(request: RtcWaveRequest) -> dict[str, Any]:
    entries = compile_rtc_wave_table(assets, request)
    return {"entries": entries, "count": len(entries), "capacity": 512, "verified": False}


@app.post("/api/devices/rtc/waves/write")
async def write_rtc_waves(request: RtcWaveRequest) -> dict[str, Any]:
    entries = compile_rtc_wave_table(assets, request)
    async with devices.manual_control("rtc_wave_write"):
        rtc = devices.require("rtc")
        devices.note_configuration_change("rtc")
        return await read_rtc_wave_table(rtc, entries, write=True)


@app.post("/api/devices/rtc/waves/read")
async def read_rtc_waves(request: RtcWaveRequest) -> dict[str, Any]:
    entries = compile_rtc_wave_table(assets, request)
    # 24 is not allowed during a real-time group; serialize under the same control
    # lease as writes so a manual table read cannot interrupt an active run.
    async with devices.manual_control("rtc_wave_read"):
        return await read_rtc_wave_table(devices.require("rtc", ready=False), entries)


@app.post("/api/devices/{device_id}/connect")
async def connect_device(device_id: str, request: DeviceConnectRequest) -> dict[str, Any]:
    return await devices.connect(device_id, request.source, request.parameters)


@app.post("/api/devices/{device_id}/disconnect")
async def disconnect_device(device_id: str) -> dict[str, Any]:
    return await devices.disconnect(device_id)


@app.post("/api/devices/{device_id}/command")
async def device_command(device_id: str, request: DeviceCommandRequest) -> Any:
    result = await devices.command(device_id, request.action, request.parameters)
    await events.publish("device.command", device_id=device_id, action=request.action, result=result)
    return result


@app.get("/api/devices")
async def list_devices() -> list[dict[str, Any]]:
    return await devices.list_status()


@app.get("/api/devices/serial-ports")
async def list_serial_ports() -> list[dict[str, str]]:
    """Enumerate Windows serial ports without opening or changing any device."""
    # Windows discovery can touch device metadata, so keep it off the API loop.
    return await asyncio.to_thread(discover_serial_ports)


@app.post("/api/runs/prepare")
async def prepare_run(plan: RunPlan) -> dict[str, Any]:
    return (await runs.prepare(plan)).public()


@app.get("/api/runs/current")
async def current_run() -> dict[str, Any]:
    return runs.snapshot()


@app.post("/api/shutdown/prepare")
async def prepare_shutdown() -> dict[str, Any]:
    # Freeze new device writes first. Existing FLASH writes/manual motion retain their
    # lease to finish readback; an automatic run stops at its next atomic boundary.
    devices.shutdown_requested = True
    record = runs.current()
    if record and record.state.value in {"RUNNING", "PAUSED"}:
        await runs.stop(record.run_id)
    if devices.control_owner is not None:
        return {"ready": False, **runs.snapshot()}
    async with devices.manual_control("shutdown"):
        for adapter in list(devices.devices.values()):
            if adapter.snapshot.state.value == "DISCONNECTED":
                continue
            if adapter.snapshot.device_id == "turntable" and getattr(adapter, "has_connection", True):
                # A previous fault can leave hardware moving despite no active task.
                # Software Stop is sent once and verified before disconnecting PMAC.
                await getattr(adapter, "stop")("all")
            await adapter.disconnect()
            await events.publish("device.status", device=await adapter.status())
    return {"ready": True, **runs.snapshot()}


@app.post("/api/shutdown/cancel")
async def cancel_shutdown() -> dict[str, bool]:
    devices.shutdown_requested = False
    return {"cancelled": True}


@app.post("/api/runs/{run_id}/start")
async def start_run(run_id: str) -> dict[str, Any]:
    return (await runs.start(run_id)).public()


@app.post("/api/runs/{run_id}/pause")
async def pause_run(run_id: str) -> dict[str, Any]:
    return (await runs.pause(run_id)).public()


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str) -> dict[str, Any]:
    return (await runs.resume(run_id)).public()


@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: str) -> dict[str, Any]:
    return (await runs.stop(run_id)).public()


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    return runs.get(run_id).public()


@app.post("/api/data/inspect")
async def inspect_file(request: LoadPathRequest) -> dict[str, Any]:
    return inspect_data_file(request.path)


@app.post("/api/data/view")
async def data_view(request: DataViewRequest) -> dict[str, Any]:
    return read_data_view(request.path, request.frequency_index, request.beam_index)


@app.post("/api/compensation/generate")
async def generate_compensation(request: CompensationRequest) -> dict[str, Any]:
    result = compensation.generate(request, assets.coordinate(request.coordinate_id))
    await events.publish("compensation.complete", result=result)
    return result


@app.post("/api/flash/prepare")
async def prepare_flash(request: FlashPrepareRequest) -> dict[str, Any]:
    result = flash.prepare(request, assets.coordinate(request.coordinate_id))
    await events.publish("flash.prepared", result=result)
    return result


@app.post("/api/flash/write")
async def write_flash(request: FlashWriteRequest) -> dict[str, Any]:
    async with devices.manual_control("flash_write"):
        result = await flash.write_item(request, devices)
        await events.publish("flash.written", result=result)
        return result


@app.post("/api/flash/items")
async def list_flash_items(request: LoadPathRequest) -> list[dict[str, Any]]:
    return flash.list_items(request.path)


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = events.subscribe()
    try:
        while True:
            event = await queue.get()
            if event["type"] == "stream.resync":
                await websocket.close(code=1013, reason="事件积压，需要重新同步运行快照")
                break
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        events.unsubscribe(queue)
