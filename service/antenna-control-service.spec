from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


service_dir = Path(SPECPATH).resolve()
hidden_imports = collect_submodules("uvicorn")
# These adapters are imported inside connect methods so hardware-specific packages
# remain optional during source development, but they must exist in the frozen EXE.
hidden_imports.extend(["clr", "pythonnet", "clr_loader", "serial", "pyvisa"])

analysis = Analysis(
    [str(service_dir / "antenna_service" / "main.py")],
    pathex=[str(service_dir)],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
archive = PYZ(analysis.pure)

# The service is spawned by Electron with windowsHide=true.  Keeping console=False
# prevents a second terminal window while logs continue through the parent pipes.
executable = EXE(
    archive,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="antenna-control-service",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
