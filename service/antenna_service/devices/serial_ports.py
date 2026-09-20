from __future__ import annotations

import os
import re
from typing import Any


def _port_sort_key(device: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", device.strip())
    if match:
        return match.group(1).upper(), int(match.group(2)), device.upper()
    return device.upper(), -1, device.upper()


def discover_serial_ports() -> list[dict[str, str]]:
    """Merge PySerial metadata with the Windows serial device map.

    Some virtual-pair drivers publish COM names in SERIALCOMM but omit the PnP
    metadata used by ``serial.tools.list_ports``. Reading this registry key is
    read-only and does not open, reserve, or otherwise change a port.
    """
    from serial.tools import list_ports

    discovered: dict[str, dict[str, str]] = {}
    for port in list_ports.comports():
        key = port.device.upper()
        discovered[key] = {
            "device": port.device,
            "description": port.description or "串口设备",
            "hwid": port.hwid or "",
            "manufacturer": port.manufacturer or "",
            "source": "PYSERIAL",
        }

    if os.name == "nt":
        try:
            import winreg

            registry_path = r"HARDWARE\DEVICEMAP\SERIALCOMM"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, registry_path) as key:
                index = 0
                while True:
                    try:
                        value_name, device, _ = winreg.EnumValue(key, index)
                    except OSError:
                        break
                    index += 1
                    device = str(device).strip()
                    if not device:
                        continue
                    normalized = device.upper()
                    if normalized in discovered:
                        discovered[normalized]["registry_name"] = value_name
                        continue
                    discovered[normalized] = {
                        "device": device,
                        "description": "Windows 虚拟或物理串口",
                        "hwid": f"REGISTRY:{value_name}",
                        "manufacturer": "",
                        "source": "WINDOWS_REGISTRY",
                        "registry_name": value_name,
                    }
        except (FileNotFoundError, OSError):
            # Non-standard Windows environments may not expose SERIALCOMM. The
            # PySerial result remains valid and the caller still receives a list.
            pass

    return sorted(discovered.values(), key=lambda item: _port_sort_key(item["device"]))
