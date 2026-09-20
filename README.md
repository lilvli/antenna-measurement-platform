# Antenna Measurement Platform

天线测量与标校平台，面向天线研发与实验室设备联动。采用 Electron / React / TypeScript 桌面端和 Python FastAPI 测控服务。

## 功能

- 配置包驱动的波控指令编译、串口调试和独立接收解析。
- TX/RX、H/V 逐通道标校，方向图走停及 RTC 连续扫描。
- RTC V1.0 固定32字节串口、波位预装/回读、独立TR启停。
- 矢网逐点触发和重复扫描缓冲读取。
- 补偿计算、孔径加权、FLASH分页写入与回读验证。
- HDF5测量数据、设备来源标记、历史查看及完整性校验。

RTC固件属于独立项目，不包含在本仓库。当前已通过离线模拟与接口测试；真实硬件的兼容性、脉冲端点、时序和容量仍需按现场设备验证。

## 目录

- `desktop/`：桌面界面、Electron入口及前端回归。
- `service/`：设备适配、协议、测量流程、HDF5存储和Python测试。
- `功能总结文档和必要协议/`：本公开版本仅保留运行离线测试所需的配置模板与模拟坐标样例。

## 开发运行

使用Windows、Python 3.11或更高版本，以及支持项目依赖的Node.js和pnpm。项目锁定的pnpm版本见 `package.json`。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\service[test,build]"
pnpm install
pnpm dev
```

开发启动会使用项目中的Python虚拟环境，启动本机测控服务和桌面窗口。

## 验证

```powershell
pnpm test
pnpm build
```

普通构建不生成安装包。测试默认采用模拟器或mock，不应将离线通过作为实机测量证据。

## 测试样例

- [天线协议配置包](功能总结文档和必要协议/当前模板与示例/x_radar_天线协议配置包_V1.0.xlsx)
- [通用配置包模板](功能总结文档和必要协议/当前模板与示例/通用天线协议配置包模板_V1.0.xlsx)
- [256通道模拟坐标](功能总结文档和必要协议/原协议配置包与坐标表模板与示例/天线通道坐标表/天线通道坐标表_256通道_H极化_8SPIx8芯片x4通道_v2.1_SIMULATED.xlsx)

在设备页显式选择模拟器，再加载样例，可进行离线调试。真实设备使用前应核对实际天线协议、坐标映射和设备参数。

## 真实设备与第三方组件

- 本仓库不分发厂商DLL、商业安装程序、现场报告或真实测量数据。
- 真实转台依赖合法取得的 `ImacFxDll.dll`、`Interop.PCOMMSERVERLib.dll` 和厂商PCommServer环境。开发/打包所需DLL应放在 `desktop/resources/turntable/`，这些文件已被Git忽略。
- 示例设备地址是占位配置；使用真实设备前需按现场网络和厂商登记信息核对。当前转台适配基于PMAC设备0和比例系数10000，不能直接视为适用于任意转台。
- RTC标校通过外层0x30转发关闭帧，并独立验证E2回显；B0只表示物理发送完成。RTC内部关联关闭帧的固件扩展未包含。
- 软件停止不能替代硬件急停；结果未知时不自动重发有副作用的命令。

## 许可证

Copyright (C) 2026 lilvli。

本项目以 **GNU General Public License v3.0 only（GPL-3.0-only）** 发布，完整条款见 [LICENSE](LICENSE)。第三方依赖和由使用者另行取得的厂商组件仍适用各自许可证。
