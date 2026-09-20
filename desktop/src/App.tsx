import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ApiError, WS_URL, api, post } from './api'
import type {
  CoordinateSummary,
  CurrentRunSnapshot,
  DeviceRawLog,
  DeviceStatus,
  LiveSample,
  PageId,
  ProfileSummary,
  RunStatus,
  RtcWaveTable,
  RunTopology,
  SerialPortInfo
} from './types'

type Assets = { profiles: ProfileSummary[]; coordinates: CoordinateSummary[] }
type Message = { id: number; level: 'info' | 'success' | 'error'; text: string }
type TurntableReadback = { positions: Record<string, number>; velocities: Record<string, number> }
type FlashItemName = 'ARRAY_ID' | 'SWITCH_TABLE' | 'COORDINATE' | 'TX_COMPENSATION' | 'RX_COMPENSATION'

const deviceNames: Record<string, string> = {
  beam_controller: '波控机',
  vna: '矢量网络分析仪',
  turntable: '转台',
  rtc: 'RTC'
}

const rtcReadbackLabels: Record<string, string> = {
  state: 'RTC 状态', mode: 'TR 收发模式', tr_state: 'TR 输出状态', rdy: '矢网 RDY',
  period_us: 'TR 周期 (μs)', high_us: 'TR 高宽 (μs)', delay_us: '触发延时 (μs)', trigger_width_us: '触发脉宽 (μs)',
  accepted_groups: '已接受组数', completed_groups: '已完成组数', valid_triggers: '有效触发数', completed_points: '已完成点数',
  group_in_flight: '组执行中', io_busy: 'I/O 忙', fault: '故障锁存', fault_code: '故障码', result_uncertain: '结果不明',
  wave_address: '当前波位地址', vna_trigger_index: '组内触发序号', valid_wave_entries: '有效波位数', config_valid: '配置有效',
  tx_busy: '波控发送忙', tx_state: '波控发送状态', tx_error: '波控发送错误', rx_frames: '接收帧数',
  rx_dropped_frames: '丢失接收帧数', debug_pulses: '调试触发数', debug_tr_running: 'TR 调试运行', rx_overflow: '接收溢出',
  unknown_result: '未确认操作', activity_monitor_error: '活动状态监视异常'
}

const turntableAxes = [
  { id: 1, name: '方位', unit: '°' },
  { id: 2, name: '俯仰', unit: '°' },
  { id: 3, name: '极化', unit: '°' },
  { id: 4, name: '馈源', unit: '°' },
  { id: 7, name: '平移', unit: 'mm' }
]

function acceptsFourDecimalInput(value: string): boolean {
  return value === '' || /^\d*(?:\.\d{0,4})?$/.test(value)
}

function acceptsSignedFourDecimalInput(value: string): boolean {
  return value === '' || /^-?\d*(?:\.\d{0,4})?$/.test(value)
}

function finiteTarget(value: string): number | null {
  const trimmed = value.trim()
  if (!/^-?(?:\d+(?:\.\d{0,4})?|\.\d{1,4})$/.test(trimmed)) return null
  const result = Number(trimmed)
  return Number.isFinite(result) ? result : null
}

function positiveFourDecimals(value: string | number): string {
  const numeric = Number(value)
  return Number.isFinite(numeric) && numeric > 0 ? numeric.toFixed(4) : '0.0001'
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    const target = typeof error.details.target === 'string' && error.details.target ? `；位置 ${error.details.target}` : ''
    const values = error.details.details as Record<string, unknown> | undefined
    const attenuation = typeof values?.required_db === 'number'
      ? `；需要 ${values.required_db.toFixed(3)} dB（标校 ${Number(values.calibration_db).toFixed(3)} + 孔径 ${Number(values.taper_db).toFixed(3)}），上限 ${Number(values.maximum_db).toFixed(3)} dB`
      : ''
    return `${error.message}${target}${attenuation}${error.code ? `（${error.code}）` : ''}`
  }
  return error instanceof Error ? error.message : String(error)
}

function shortHash(value?: string): string {
  return value ? `${value.slice(0, 8)}…${value.slice(-6)}` : '—'
}

function relativeHeatColor(valueDb: number | null | undefined, maximumDb: number | null): string | undefined {
  if (valueDb == null || maximumDb == null || !Number.isFinite(valueDb)) return undefined
  // The live array is normalized to its strongest measured channel. Values 30 dB
  // or more below the maximum share the same deep-blue endpoint.
  const relativeDb = Math.max(-30, Math.min(0, valueDb - maximumDb))
  const ratio = (relativeDb + 30) / 30
  const hue = 225 * (1 - ratio)
  const lightness = 27 + ratio * 19
  return `hsl(${hue.toFixed(1)} 84% ${lightness.toFixed(1)}%)`
}

function inclusiveAxisValues(start: number, stop: number, step: number): number[] {
  if (!(step > 0) || stop < start) return []
  const count = Math.floor((stop - start) / step + 1e-9) + 1
  const values = Array.from({ length: count }, (_, index) => start + index * step)
  if (values.length > 0 && values[values.length - 1] < stop - 1e-9) values.push(stop)
  return values
}

function parseBeamPoints(text: string) {
  const rows = text.split(/[;；\n]+/).map((item) => item.trim()).filter(Boolean)
  if (rows.length === 0) throw new Error('至少需要一个电子波束方向')
  return rows.map((row, index) => {
    const values = row.split(/[,，\s]+/).filter(Boolean).map(Number)
    if (values.length !== 2 || values.some((value) => !Number.isFinite(value))) {
      throw new Error(`电子波束第 ${index + 1} 项应为“离轴角, 方位角”`)
    }
    return { beam_id: `BEAM-${index + 1}`, off_axis_deg: values[0], azimuth_deg: values[1] }
  })
}

function frequencyValues(startGhz: number, stopGhz: number, count: number): number[] {
  if (count <= 1) return [startGhz * 1e9]
  return Array.from({ length: count }, (_, index) => (startGhz + (stopGhz - startGhz) * index / (count - 1)) * 1e9)
}

function desktopBridge() {
  if (!window.antennaDesktop) {
    throw new Error('Electron preload 桥未加载，请从项目根目录执行 pnpm dev 后重试')
  }
  return window.antennaDesktop
}

function StatusPill({ value, source }: { value: string; source?: string | null }) {
  const tone = ['READY', 'COMPLETED', 'VERIFIED', 'SUCCESS', 'healthy', 'ok', 'content_validated'].includes(value)
    ? 'good'
    : ['FAULT', 'FAULTED', 'UNKNOWN', 'DISCONNECTED'].includes(value)
      ? 'bad'
      : 'neutral'
  return (
    <span className={`status-pill ${tone}`}>
      <i /> {source ? `${source} · ` : ''}{value === 'ok' ? '内容与摘要复验通过' : value === 'content_validated' ? '内容检查通过（无历史摘要）' : value}
    </span>
  )
}

function Card({ title, eyebrow, children, actions }: {
  title: string
  eyebrow?: string
  children: React.ReactNode
  actions?: React.ReactNode
}) {
  return (
    <section className="card">
      <div className="card-head">
        <div>{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h3>{title}</h3></div>
        {actions && <div className="card-actions">{actions}</div>}
      </div>
      {children}
    </section>
  )
}

function Field({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) {
  return <label className="field"><span>{label}</span>{children}{hint && <small>{hint}</small>}</label>
}

function App() {
  const [page, setPage] = useState<PageId>('run')
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')
  const [health, setHealth] = useState('连接中')
  const [assets, setAssets] = useState<Assets>({ profiles: [], coordinates: [] })
  const [devices, setDevices] = useState<DeviceStatus[]>([])
  const [run, setRunState] = useState<RunStatus | null>(null)
  const discardedPreparedRuns = useRef(new Set<string>())
  const setRun = useCallback<React.Dispatch<React.SetStateAction<RunStatus | null>>>((next) => {
    setRunState((current) => {
      const result = typeof next === 'function' ? next(current) : next
      if (!result && current?.state === 'PREPARED') discardedPreparedRuns.current.add(current.run_id)
      if (result?.state === 'PREPARED' && discardedPreparedRuns.current.has(result.run_id)) return current
      return result
    })
  }, [])
  const [samples, setSamples] = useState<LiveSample[]>([])
  const [deviceLogs, setDeviceLogs] = useState<DeviceRawLog[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [arrayIdText, setArrayIdText] = useState('0')
  const [arrayIdBusySources, setArrayIdBusySources] = useState<string[]>([])

  const arrayId = /^\d+$/.test(arrayIdText) && Number(arrayIdText) <= 255 ? Number(arrayIdText) : null

  const setArrayIdOperationBusy = useCallback((source: string, busy: boolean) => {
    setArrayIdBusySources((current) => busy
      ? current.includes(source) ? current : [...current, source]
      : current.filter((item) => item !== source))
  }, [])

  const notify = useCallback((level: Message['level'], text: string) => {
    const id = Date.now() + Math.random()
    setMessages((current) => [{ id, level, text }, ...current].slice(0, 6))
    // Errors remain visible slightly longer; routine information should clear quickly
    // so it never covers the controls or the live heatmap indefinitely.
    window.setTimeout(() => {
      setMessages((current) => current.filter((message) => message.id !== id))
    }, level === 'error' ? 8000 : 4500)
  }, [])

  const updateArrayId = useCallback((value: string) => {
    if (value !== '' && (!/^\d{1,3}$/.test(value) || Number(value) > 255)) return
    if (value === arrayIdText) return
    setArrayIdText(value)
    if (run?.state === 'PREPARED') {
      setRun(null)
      notify('info', '当前阵面 ID 已修改，原测试准备已失效，请重新点击“准备测试”确认')
    }
  }, [arrayIdText, notify, run?.state])

  const refresh = useCallback(async () => {
    try {
      const healthResult = await api<{ status: string }>('/api/health')
      setHealth(healthResult.status)
      const [assetResult, deviceResult] = await Promise.all([
        api<Assets>('/api/assets'),
        api<DeviceStatus[]>('/api/devices')
      ])
      setAssets(assetResult)
      setDevices(deviceResult)
    } catch (error) {
      setHealth('不可达')
      notify('error', errorMessage(error))
    }
  }, [notify])

  useEffect(() => { void refresh() }, [refresh])

  useEffect(() => {
    let socket: WebSocket | null = null
    let retry: number | null = null
    let stopped = false
    const connect = () => {
      if (stopped) return
      const connection = new WebSocket(WS_URL)
      socket = connection
      let syncing = true
      let watermark = 0
      const pending: Record<string, any>[] = []
      const applyMessage = (message: Record<string, any>) => {
        if (message.sequence <= watermark) return
        if (message.type === 'device.status') {
          setDevices((current) => current.map((item) => item.device_id === message.device.device_id ? message.device : item))
        } else if (message.type === 'asset.loaded') {
          if (message.asset_type === 'profile') {
            setAssets((current) => ({
              ...current,
              profiles: [...current.profiles.filter((item) => item.asset_id !== message.asset.asset_id), message.asset]
            }))
          } else if (message.asset_type === 'coordinates') {
            setAssets((current) => ({
              ...current,
              coordinates: [...current.coordinates.filter((item) => item.asset_id !== message.asset.asset_id), message.asset]
            }))
          }
        } else if (message.type === 'device.raw') {
          setDeviceLogs((current) => [...current, message as DeviceRawLog].slice(-2000))
        } else if (message.type === 'run.status') {
          setRun(message.run)
          if (['COMPLETED', 'STOPPED'].includes(message.run.state)) notify('success', `运行已结束：${message.run.state}`)
          if (['FAULTED', 'UNKNOWN'].includes(message.run.state)) notify('error', `运行异常：${message.run.error?.message ?? message.run.state}`)
        } else if (message.type === 'run.post_complete') {
          setRun(message.run)
          if (message.cleanup?.status === 'SUCCESS') {
            notify('success', '测试数据已完成保存，方位轴寻零完成')
          } else {
            notify('error', `测试数据已完成保存，但方位轴寻零失败：${message.cleanup?.error?.message ?? '请检查转台'}`)
          }
        } else if (message.type === 'run.sample') {
          setSamples((current) => [...current.filter((sample) => sample.run_id === message.run_id && sample.completed !== message.completed), message as LiveSample])
          setRun((current) => current && current.run_id === message.run_id
            ? { ...current, completed: message.completed, total: message.total, progress: message.total ? message.completed / message.total : 0 }
            : current)
        } else if (message.type?.startsWith('flash.') || message.type === 'compensation.complete') {
          notify('success', message.type === 'compensation.complete' ? '补偿文件生成完成' : `FLASH：${message.type}`)
        }
      }
      connection.onmessage = (event) => {
        const message = JSON.parse(event.data)
        if (syncing) pending.push(message)
        else applyMessage(message)
      }
      connection.onopen = async () => {
        try {
          // Buffer events while fetching the snapshot. The sequence watermark prevents
          // duplicate samples and stale statuses from rolling back restored state.
          const snapshot = await api<CurrentRunSnapshot>('/api/runs/current')
          if (stopped || socket !== connection || connection.readyState !== WebSocket.OPEN) return
          watermark = snapshot.event_sequence
          const restored = snapshot.run
          if (!(restored?.state === 'PREPARED' && discardedPreparedRuns.current.has(restored.run_id))) {
            setRun(restored)
            setSamples(snapshot.samples)
            if (restored && (['PREPARED', 'RUNNING', 'PAUSED', 'STOPPING'].includes(restored.state) || restored.cleanup_pending)) {
              setArrayIdText(String(restored.plan.array_id))
            }
          }
          syncing = false
          pending.forEach(applyMessage)
          pending.length = 0
          void refresh()
        } catch (error) {
          if (stopped || socket !== connection) return
          setHealth('同步失败')
          notify('error', `运行状态恢复失败：${errorMessage(error)}`)
          connection.close()
        }
      }
      connection.onclose = () => {
        if (!stopped) {
          setHealth('重连中')
          retry = window.setTimeout(connect, 1200)
        }
      }
    }
    connect()
    return () => {
      stopped = true
      if (retry !== null) window.clearTimeout(retry)
      socket?.close()
    }
  }, [notify, refresh, setRun])

  const activeDevices = devices.filter((device) => device.state !== 'DISCONNECTED').length
  const deviceControlLocked = arrayIdBusySources.includes('run') || Boolean(run && (['RUNNING', 'PAUSED', 'STOPPING'].includes(run.state) || run.cleanup_pending))
  const arrayIdLocked = arrayIdBusySources.length > 0 || deviceControlLocked
  const invalidateDevicePreparation = () => {
    if (run?.state !== 'PREPARED') return
    setRun(null)
    notify('info', '设备调试设置已修改，原测试准备已失效，请重新准备测试')
  }
  const nav = [
    { id: 'run' as const, icon: '⌁', label: '测试执行', note: '标校 / 方向图' },
    { id: 'history' as const, icon: '◫', label: '历史分析', note: '数据校验' },
    { id: 'compensation' as const, icon: '◎', label: '补偿与 FLASH', note: '生成 / 写入' },
    { id: 'devices' as const, icon: '⎋', label: '设备调试', note: '真实设备 / 模拟器' }
  ]

  return (
    <div className={`app ${theme}`}>
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark"><span /></div><div><strong>ANTENNA</strong><small>RANGE CONTROL</small></div></div>
        <nav>
          {nav.map((item) => (
            <button className={page === item.id ? 'active' : ''} key={item.id} onClick={() => setPage(item.id)}>
              <span className="nav-icon">{item.icon}</span><span><b>{item.label}</b><small>{item.note}</small></span>
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className="mini-metric"><span>设备在线</span><strong>{activeDevices}<em>/ 4</em></strong></div>
          <div className="mini-metric"><span>天线资料</span><strong>{assets.profiles.length && assets.coordinates.length ? '已加载' : '待加载'}</strong></div>
          <button className="theme-button" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
            {theme === 'dark' ? '☀ 切换明亮主题' : '◐ 切换暗色主题'}
          </button>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div><span className="breadcrumb">天线紧缩场 / {nav.find((item) => item.id === page)?.label}</span><h1>{nav.find((item) => item.id === page)?.label}</h1></div>
          <div className="top-status">
            <label className="global-array-id">
              <span>当前阵面 ID</span>
              <input
                id="global-array-id"
                type="number"
                min="0"
                max="255"
                step="1"
                value={arrayIdText}
                disabled={arrayIdLocked}
                onChange={(event) => updateArrayId(event.target.value)}
                onBlur={() => { if (arrayId === null) setArrayIdText('0') }}
              />
              <small>{arrayIdLocked ? '相关操作进行中，已锁定' : '测试、FLASH、调试统一引用'}</small>
            </label>
            <StatusPill value={health} /><span className="clock">本机服务 · API 1.0</span>
          </div>
        </header>
        <div className="content">
          <div className={`page-view ${page === 'run' ? '' : 'inactive'}`} aria-hidden={page !== 'run'}>
            <RunPage assets={assets} setAssets={setAssets} devices={devices} run={run} setRun={setRun} samples={samples} setSamples={setSamples} arrayId={arrayId} onArrayIdBusyChange={setArrayIdOperationBusy} notify={notify} />
          </div>
          <div className={`page-view ${page === 'history' ? '' : 'inactive'}`} aria-hidden={page !== 'history'}>
            <HistoryPage notify={notify} />
          </div>
          <div className={`page-view ${page === 'compensation' ? '' : 'inactive'}`} aria-hidden={page !== 'compensation'}>
            <CompensationPage assets={assets} devices={devices} arrayId={arrayId} onArrayIdBusyChange={setArrayIdOperationBusy} notify={notify} />
          </div>
          <div className={`page-view ${page === 'devices' ? '' : 'inactive'}`} aria-hidden={page !== 'devices'}>
            <DevicesPage profiles={assets.profiles} devices={devices} setDevices={setDevices} logs={deviceLogs} clearLogs={() => setDeviceLogs([])} controlLocked={deviceControlLocked} onSettingsChange={invalidateDevicePreparation} arrayId={arrayId} onArrayIdBusyChange={setArrayIdOperationBusy} notify={notify} />
          </div>
        </div>
      </main>
      <div className="toast-stack">
        {messages.slice(0, 3).map((message) => <div className={`toast ${message.level}`} key={message.id}>{message.text}</div>)}
      </div>
    </div>
  )
}

function RunPage({ assets, setAssets, devices, run, setRun, samples, setSamples, arrayId, onArrayIdBusyChange, notify }: {
  assets: Assets
  setAssets: React.Dispatch<React.SetStateAction<Assets>>
  devices: DeviceStatus[]
  run: RunStatus | null
  setRun: React.Dispatch<React.SetStateAction<RunStatus | null>>
  samples: LiveSample[]
  setSamples: React.Dispatch<React.SetStateAction<LiveSample[]>>
  arrayId: number | null
  onArrayIdBusyChange: (source: string, busy: boolean) => void
  notify: (level: Message['level'], text: string) => void
}) {
  const [loading, setLoading] = useState(false)
  const [outputDirectory, setOutputDirectory] = useState('')
  const [liveFrequencyIndex, setLiveFrequencyIndex] = useState(0)
  const [selectedLiveBeamId, setSelectedLiveBeamId] = useState('BEAM-1')
  const [form, setForm] = useState({
    test_type: 'CALIBRATION', topology: 'SOFTWARE_VNA_SWEEP' as RunTopology, beam_control_mode: 'SOFTWARE_DIRECT', signal_path: 'TX', polarization: 'H',
    s_parameter: 'S21', base_filename: 'antenna_test', frequency_start_ghz: 8,
    frequency_stop_ghz: 8, frequency_points: 1, if_bandwidth_hz: 1000,
    source_power_dbm: -10, averaging_enabled: false, averaging_count: 1, settle_ms: 10,
    azimuth_start_deg: '0', azimuth_stop_deg: '0', azimuth_step_deg: 1,
    elevation_start_deg: '0', elevation_stop_deg: '0', elevation_step_deg: 1, move_speed_deg_s: '1.0000',
    beam_points_text: '0, 0'
  })
  useEffect(() => {
    if (!run) return
    const plan = run.plan
    setForm({
      test_type: plan.test_type, topology: plan.topology, beam_control_mode: plan.beam_control_mode,
      signal_path: plan.signal_path, polarization: plan.polarization, s_parameter: plan.s_parameter,
      base_filename: plan.base_filename, frequency_start_ghz: plan.frequency_start_hz / 1e9,
      frequency_stop_ghz: plan.frequency_stop_hz / 1e9, frequency_points: plan.frequency_points,
      if_bandwidth_hz: plan.if_bandwidth_hz, source_power_dbm: plan.source_power_dbm,
      averaging_enabled: plan.averaging_enabled, averaging_count: plan.averaging_count, settle_ms: plan.settle_ms,
      azimuth_start_deg: String(plan.azimuth_start_deg), azimuth_stop_deg: String(plan.azimuth_stop_deg),
      azimuth_step_deg: plan.azimuth_step_deg, elevation_start_deg: String(plan.elevation_start_deg),
      elevation_stop_deg: String(plan.elevation_stop_deg), elevation_step_deg: plan.elevation_step_deg,
      move_speed_deg_s: plan.move_speed_deg_s.toFixed(4),
      beam_points_text: plan.beams.map((beam) => `${beam.off_axis_deg}, ${beam.azimuth_deg}`).join('; ')
    })
    setOutputDirectory(plan.output_directory)
    setLiveFrequencyIndex(0)
    setSelectedLiveBeamId(plan.beams[0]?.beam_id ?? 'BEAM-1')
  }, [run?.run_id])
  const profile = assets.profiles[assets.profiles.length - 1]
  const coordinates = assets.coordinates[assets.coordinates.length - 1]
  const rtcMeasurement = form.topology !== 'SOFTWARE_VNA_SWEEP'
  const rtcContinuous = rtcMeasurement && form.topology === 'RTC_CONTINUOUS'
  const acquisitionOnly = form.test_type === 'PATTERN' && form.beam_control_mode === 'EXTERNAL_FIXED'
  const signalPathHint = form.test_type !== 'PATTERN'
    ? undefined
    : acquisitionOnly
      ? '用于VNA激励端口和结果标识；仅采集模式不发送波控指令'
      : profile && !profile.capabilities.beam_signal_path
        ? '当前配置包的BEAM_SET不含收发字段；选择仍用于VNA和结果记录，开始前请在设备调试页另行设置收发状态'
        : '用于VNA激励端口，并随BEAM_SET发送给波控机'
  const requiredIds = form.test_type === 'PATTERN'
    ? rtcMeasurement ? ['vna', 'rtc', 'turntable'] : acquisitionOnly ? ['vna', 'turntable'] : ['vna', 'beam_controller', 'turntable']
    : rtcMeasurement ? ['vna', 'rtc'] : ['vna', 'beam_controller']
  const ready = requiredIds.every((id) => devices.find((device) => device.device_id === id)?.state === 'READY' || devices.find((device) => device.device_id === id)?.state === 'CONNECTED')
  const planLocked = Boolean(run && (['RUNNING', 'PAUSED', 'STOPPING'].includes(run.state) || run.cleanup_pending))
  const planControlsDisabled = loading || planLocked
  const rtcReady = devices.some((device) => device.device_id === 'rtc' && ['READY', 'CONNECTED'].includes(device.state))
  const [rtcWaveResult, setRtcWaveResult] = useState<{ key: string; data: RtcWaveTable | null; error?: string } | null>(null)
  const rtcWaveRequestVersion = useRef(0)
  const [rtcWavesVerifiedFor, setRtcWavesVerifiedFor] = useState<string | null>(null)
  const waveInputs = useMemo(() => {
    try {
      if (!profile || (form.test_type === 'CALIBRATION' && !coordinates) || arrayId === null) return { payload: null, error: form.test_type === 'CALIBRATION' ? '请先加载配置包、坐标表并设置阵面 ID。' : '请先加载配置包并设置阵面 ID。' }
      const referenceFrequencyHz = (Number(form.frequency_start_ghz) + Number(form.frequency_stop_ghz)) * 0.5e9
      const beams = (form.test_type === 'PATTERN' ? parseBeamPoints(form.beam_points_text) : [{ beam_id: 'BEAM-1', off_axis_deg: 0, azimuth_deg: 0 }])
        .map((beam) => ({ ...beam, reference_frequency_hz: referenceFrequencyHz }))
      return { payload: {
        profile_id: profile.asset_id, coordinate_id: coordinates?.asset_id, test_type: form.test_type,
        array_id: arrayId, signal_path: form.signal_path, polarization: form.polarization,
        reference_frequency_hz: referenceFrequencyHz, beams
      }, error: '' }
    } catch (error) { return { payload: null, error: errorMessage(error) } }
  }, [profile, coordinates, arrayId, form.test_type, form.signal_path, form.polarization, form.frequency_start_ghz, form.frequency_stop_ghz, form.beam_points_text])
  const rtcWaveKey = JSON.stringify(waveInputs.payload)
  const rtcWaveKeyRef = useRef(rtcWaveKey)
  rtcWaveKeyRef.current = rtcWaveKey
  const visibleRtcWaves = rtcWaveResult?.key === rtcWaveKey ? rtcWaveResult : null
  const rtcWaveVerified = rtcWavesVerifiedFor === rtcWaveKey && rtcReady

  useEffect(() => {
    const version = ++rtcWaveRequestVersion.current
    setRtcWavesVerifiedFor(null)
    if (!rtcMeasurement || acquisitionOnly || !waveInputs.payload || planLocked) return
    setRtcWaveResult(null)
    // Compile a preview only. This endpoint never connects to or writes an instrument.
    void post<RtcWaveTable>('/api/devices/rtc/waves/preview', waveInputs.payload).then((data) => {
      if (version === rtcWaveRequestVersion.current && rtcWaveKeyRef.current === rtcWaveKey) setRtcWaveResult({ key: rtcWaveKey, data })
    }).catch((error) => {
      if (version === rtcWaveRequestVersion.current && rtcWaveKeyRef.current === rtcWaveKey) setRtcWaveResult({ key: rtcWaveKey, data: null, error: errorMessage(error) })
    })
    return () => { ++rtcWaveRequestVersion.current }
  }, [rtcWaveKey, rtcMeasurement, acquisitionOnly])

  useEffect(() => {
    if (!rtcReady) {
      setRtcWavesVerifiedFor(null)
      setRtcWaveResult((current) => current?.data ? { ...current, data: { ...current.data, verified: false, entries: current.data.entries.map((entry) => ({ ...entry, status: 'PREVIEW' })) } } : current)
    }
  }, [rtcReady])

  useEffect(() => {
    onArrayIdBusyChange('run', loading)
    return () => onArrayIdBusyChange('run', false)
  }, [loading, onArrayIdBusyChange])

  const invalidatePreparedPlan = () => {
    if (run?.state !== 'PREPARED') return
    setRun(null)
    notify('info', '测试计划已修改，原准备已失效，请重新点击“准备测试”确认')
  }

  const updatePlanForm = (next: typeof form) => {
    if (JSON.stringify(next) === JSON.stringify(form)) return
    setForm(next)
    invalidatePreparedPlan()
  }

  const loadAsset = async (kind: 'profile' | 'coordinates') => {
    try {
      const path = await desktopBridge().openFile(kind)
      if (!path) return
      setLoading(true)
      if (kind === 'profile') {
        const result = await post<ProfileSummary>('/api/assets/profile/load', { path })
        setAssets((current) => ({ ...current, profiles: [...current.profiles.filter((item) => item.asset_id !== result.asset_id), result] }))
        if (profile?.asset_id !== result.asset_id) invalidatePreparedPlan()
        notify('success', `配置包已验证：${result.profile_name}，${result.vectors.length} 个固定向量通过`)
      } else {
        const result = await post<CoordinateSummary>('/api/assets/coordinates/load', { path })
        setAssets((current) => ({ ...current, coordinates: [...current.coordinates.filter((item) => item.asset_id !== result.asset_id), result] }))
        if (coordinates?.asset_id !== result.asset_id) invalidatePreparedPlan()
        notify('success', `坐标表已验证：${result.channel_count} 通道，启用 ${result.enabled_count}`)
      }
    } catch (error) { notify('error', errorMessage(error)) } finally { setLoading(false) }
  }

  const connectSimulated = async () => {
    setLoading(true)
    try {
      for (const deviceId of requiredIds) {
        await post(`/api/devices/${deviceId}/connect`, { source: 'SIMULATED', parameters: {} })
      }
      notify('success', `${requiredIds.map((id) => deviceNames[id]).join('、')}模拟设备已就绪`)
    } catch (error) { notify('error', errorMessage(error)) } finally { setLoading(false) }
  }

  const chooseOutput = async () => {
    try {
      const result = await desktopBridge().openDirectory()
      if (result && result !== outputDirectory) {
        setOutputDirectory(result)
        invalidatePreparedPlan()
      }
    } catch (error) { notify('error', errorMessage(error)) }
  }

  const prepare = async () => {
    if (!profile || !coordinates || !outputDirectory || arrayId === null) return
    setLoading(true)
    setSamples([])
    if (rtcMeasurement) {
      setRtcWavesVerifiedFor(null)
      setRtcWaveResult((current) => current?.data ? { ...current, data: { ...current.data, verified: false, entries: current.data.entries.map((entry) => ({ ...entry, status: 'PREVIEW' })) } } : current)
    }
    try {
      if (!waveInputs.payload) throw new Error(waveInputs.error)
      const parsedPatternBeams = waveInputs.payload.beams
      if (acquisitionOnly && parsedPatternBeams.length !== 1) {
        throw new Error('仅采集模式每次运行只能填写一个外部固定波位')
      }
      const beams = form.test_type === 'PATTERN'
        ? acquisitionOnly
          ? [{ ...parsedPatternBeams[0], beam_id: 'EXTERNAL-FIXED' }]
          : parsedPatternBeams
        : parsedPatternBeams
      const result = await post<RunStatus>('/api/runs/prepare', {
        name: form.test_type === 'CALIBRATION' ? `${form.signal_path}-${form.polarization} 定点标校` : '方向图测试',
        test_type: form.test_type,
        topology: form.topology,
        beam_control_mode: acquisitionOnly ? 'EXTERNAL_FIXED' : 'SOFTWARE_DIRECT',
        signal_path: form.signal_path,
        polarization: form.polarization,
        s_parameter: form.s_parameter,
        profile_id: profile.asset_id,
        coordinate_id: coordinates.asset_id,
        array_id: arrayId,
        output_directory: outputDirectory,
        base_filename: form.base_filename,
        frequency_start_hz: Number(form.frequency_start_ghz) * 1e9,
        frequency_stop_hz: Number(form.frequency_stop_ghz) * 1e9,
        frequency_points: Number(form.frequency_points),
        if_bandwidth_hz: Number(form.if_bandwidth_hz),
        source_power_dbm: Number(form.source_power_dbm),
        averaging_enabled: form.averaging_enabled,
        averaging_count: form.averaging_enabled ? Number(form.averaging_count) : 1,
        settle_ms: Number(form.settle_ms),
        azimuth_start_deg: Number(form.azimuth_start_deg),
        azimuth_stop_deg: Number(form.azimuth_stop_deg),
        azimuth_step_deg: Number(form.azimuth_step_deg),
        elevation_start_deg: Number(form.elevation_start_deg),
        elevation_stop_deg: Number(form.elevation_stop_deg),
        elevation_step_deg: Number(form.elevation_step_deg),
        move_speed_deg_s: Number(positiveFourDecimals(form.move_speed_deg_s)),
        beams
      })
      setRun(result)
      if (result.rtc_configuration?.waves_verified) setRtcWavesVerifiedFor(rtcWaveKey)
      setLiveFrequencyIndex(0)
      setSelectedLiveBeamId(beams[0].beam_id)
      notify('success', `计划已冻结，共 ${result.total} 个原子测量单元${rtcMeasurement && !acquisitionOnly ? '；RTC 波位已写入并校验' : ''}`)
    } catch (error) { notify('error', errorMessage(error)) } finally { setLoading(false) }
  }

  const rtcWaveAction = async (action: 'write' | 'read') => {
    if (!rtcMeasurement || acquisitionOnly || !rtcReady || !waveInputs.payload || planControlsDisabled) return
    const key = rtcWaveKey
    const version = ++rtcWaveRequestVersion.current
    setLoading(true)
    setRtcWavesVerifiedFor(null)
    if (action === 'write') invalidatePreparedPlan()
    setRtcWaveResult(visibleRtcWaves?.data ? { key, data: { ...visibleRtcWaves.data, entries: visibleRtcWaves.data.entries.map((entry) => ({ ...entry, status: 'PREVIEW' })) } } : null)
    try {
      const data = await post<RtcWaveTable>(`/api/devices/rtc/waves/${action}`, waveInputs.payload)
      if (version !== rtcWaveRequestVersion.current || rtcWaveKeyRef.current !== key) return
      setRtcWaveResult({ key, data })
      if (data.verified) setRtcWavesVerifiedFor(key)
      if (action === 'read' && !data.verified) invalidatePreparedPlan()
      notify(data.verified ? 'success' : 'info', action === 'write'
        ? `已写入并逐条回读校验 ${data.count} 条 RTC 波位`
        : data.verified ? `已读取 ${data.count} 条 RTC 波位，全部与当前计划一致` : '已读取当前计划对应的 RTC 地址，请检查未写入或不一致的条目')
    } catch (error) {
      if (version === rtcWaveRequestVersion.current && rtcWaveKeyRef.current === key) {
        setRtcWaveResult((current) => ({ key, data: current?.key === key ? current.data : null, error: errorMessage(error) }))
        if (action === 'read') invalidatePreparedPlan()
        notify('error', errorMessage(error))
      }
    } finally { setLoading(false) }
  }

  const runAction = async (action: string) => {
    if (!run || loading || (action === 'pause' && run.pause_requested)) return
    setLoading(true)
    try {
      const result = await post<RunStatus>(`/api/runs/${run.run_id}/${action}`)
      setRun(result)
    } catch (error) { notify('error', errorMessage(error)) } finally { setLoading(false) }
  }

  const calibrationLayout = coordinates?.polarization_layouts?.[form.polarization]
  const calibrationMap = useMemo(() => {
    const map = new Map<number, LiveSample>()
    samples.forEach((sample) => {
      if (sample.kind === 'CALIBRATION' && sample.channel != null) {
        const index = calibrationLayout
          ? sample.channel.grid_row * calibrationLayout.columns + sample.channel.grid_column
          : sample.channel.element
        map.set(index, sample)
      }
    })
    return map
  }, [samples, calibrationLayout])
  const frozenPatternPlan = run?.plan.test_type === 'PATTERN' ? run.plan : null
  const frozenFrequencyPlan = run?.plan ?? null
  const azimuthValues = useMemo(() => inclusiveAxisValues(
    frozenPatternPlan?.azimuth_start_deg ?? Number(form.azimuth_start_deg),
    frozenPatternPlan?.azimuth_stop_deg ?? Number(form.azimuth_stop_deg),
    frozenPatternPlan?.azimuth_step_deg ?? Number(form.azimuth_step_deg)
  ), [frozenPatternPlan, form.azimuth_start_deg, form.azimuth_stop_deg, form.azimuth_step_deg])
  const elevationValues = useMemo(() => inclusiveAxisValues(
    frozenPatternPlan?.elevation_start_deg ?? Number(form.elevation_start_deg),
    frozenPatternPlan?.elevation_stop_deg ?? Number(form.elevation_stop_deg),
    frozenPatternPlan?.elevation_step_deg ?? Number(form.elevation_step_deg)
  ), [frozenPatternPlan, form.elevation_start_deg, form.elevation_stop_deg, form.elevation_step_deg])
  const liveBeams = useMemo(() => {
    if (frozenPatternPlan) return frozenPatternPlan.beams
    try { return parseBeamPoints(form.beam_points_text).map((beam) => ({ ...beam, reference_frequency_hz: null })) } catch { return [] }
  }, [frozenPatternPlan, form.beam_points_text])
  const liveFrequencies = useMemo(() => frozenFrequencyPlan
    ? frequencyValues(frozenFrequencyPlan.frequency_start_hz / 1e9, frozenFrequencyPlan.frequency_stop_hz / 1e9, frozenFrequencyPlan.frequency_points)
    : frequencyValues(form.frequency_start_ghz, form.frequency_stop_ghz, form.frequency_points),
  [frozenFrequencyPlan, form.frequency_start_ghz, form.frequency_stop_ghz, form.frequency_points])
  const displayedBeamId = liveBeams.some((beam) => beam.beam_id === selectedLiveBeamId)
    ? selectedLiveBeamId
    : liveBeams[0]?.beam_id ?? 'BEAM-1'
  const patternMap = useMemo(() => {
    const map = new Map<number, LiveSample>()
    samples.forEach((sample) => {
      if (sample.kind !== 'PATTERN' || !sample.point || (sample.beam && sample.beam.beam_id !== displayedBeamId)) return
      const column = azimuthValues.findIndex((value) => Math.abs(value - sample.point!.azimuth_deg) < 1e-6)
      const row = elevationValues.findIndex((value) => Math.abs(value - sample.point!.elevation_deg) < 1e-6)
      if (row >= 0 && column >= 0) map.set(row * azimuthValues.length + column, sample)
    })
    return map
  }, [samples, azimuthValues, elevationValues, displayedBeamId])
  const heatCount = form.test_type === 'CALIBRATION'
    ? Math.max(1, calibrationLayout?.channel_count ?? coordinates?.channel_count ?? 64)
    : Math.max(1, azimuthValues.length * elevationValues.length)
  const heatColumns = form.test_type === 'CALIBRATION'
    ? Math.max(1, calibrationLayout?.columns ?? Math.ceil(Math.sqrt(heatCount)))
    : Math.max(1, azimuthValues.length)
  const heatRows = form.test_type === 'CALIBRATION'
    ? Math.max(1, calibrationLayout?.rows ?? Math.ceil(heatCount / heatColumns))
    : Math.max(1, elevationValues.length)
  const activeHeatMap = form.test_type === 'CALIBRATION' ? calibrationMap : patternMap
  const displayedMagnitude = (sample: LiveSample | undefined) => sample?.magnitudes_db?.[liveFrequencyIndex] ?? sample?.magnitude_db
  const displayedPhase = (sample: LiveSample | undefined) => sample?.phases_deg?.[liveFrequencyIndex] ?? sample?.phase_deg
  const measuredMagnitudes = Array.from(activeHeatMap.values())
    .map((sample) => displayedMagnitude(sample))
    .filter((value): value is number => value != null && Number.isFinite(value))
  const heatMaximum = measuredMagnitudes.length > 0 ? Math.max(...measuredMagnitudes) : null

  return (
    <>
      <div className="hero-strip">
        <div><span className="eyebrow">WORKFLOW</span><h2>从两份天线资料开始</h2><p>加载配置包与坐标表后，软件离线编译全部通道，再进入设备、计划和采集流程。</p></div>
        <div className="hero-steps"><span className={profile ? 'done' : 'active'}>1<small>配置包</small></span><i /><span className={coordinates ? 'done' : ''}>2<small>坐标表</small></span><i /><span className={ready ? 'done' : ''}>3<small>设备</small></span><i /><span className={run ? 'done' : ''}>4<small>运行</small></span></div>
      </div>
      <div className="grid two">
        <Card title="天线输入" eyebrow="01 · ASSETS">
          <div className="asset-row">
            <div className="asset-icon">XLSX</div><div className="asset-copy"><b>{profile?.profile_name ?? '天线协议配置包'}</b><span>{profile ? `${profile.profile_id} · 协议 ${profile.protocol_version}` : '尚未选择 .xlsx'}</span>{profile && <small>SHA {shortHash(profile.file_sha256)} · {profile.vectors.length} 个向量通过</small>}</div>
            <button className="button secondary" disabled={planControlsDisabled} onClick={() => loadAsset('profile')}>{profile ? '更换' : '选择'}</button>
          </div>
          <div className="asset-row">
            <div className="asset-icon coord">XYZ</div><div className="asset-copy"><b>{coordinates?.antenna_id ?? '天线通道坐标表'}</b><span>{coordinates ? `${coordinates.channel_count} 通道 · ${coordinates.polarizations.join('/')}` : '尚未选择 .xlsx'}</span>{coordinates && <small><span className={`evidence ${coordinates.evidence.toLowerCase()}`}>{coordinates.evidence}</span> 启用 {coordinates.enabled_count}</small>}</div>
            <button className="button secondary" disabled={planControlsDisabled} onClick={() => loadAsset('coordinates')}>{coordinates ? '更换' : '选择'}</button>
          </div>
        </Card>
        <Card title="设备就绪" eyebrow="02 · DEVICES" actions={<button className="button ghost" disabled={planControlsDisabled} onClick={connectSimulated}>连接测试模拟设备</button>}>
          <div className="device-compact-grid">
            {devices.map((device) => <div className="device-compact" key={device.device_id}><span>{deviceNames[device.device_id]}</span><StatusPill value={device.state} source={device.source} /><small>{device.identity ?? '未连接'}</small></div>)}
          </div>
          <p className="inline-note">真实设备请在“设备调试”页逐台配置并连接；运行页不会自动访问硬件。</p>
        </Card>
      </div>
      <Card title="测试计划" eyebrow="03 · PLAN">
        <fieldset className="plan-fields" disabled={planControlsDisabled}>
          <div className="segmented">
            <button className={form.test_type === 'CALIBRATION' ? 'selected' : ''} onClick={() => updatePlanForm({ ...form, test_type: 'CALIBRATION', topology: form.topology === 'RTC_CONTINUOUS' ? 'RTC_STOP_AND_GO' : form.topology })}>逐通道定点标校</button>
            <button className={form.test_type === 'PATTERN' ? 'selected' : ''} onClick={() => updatePlanForm({ ...form, test_type: 'PATTERN' })}>方向图扫描</button>
          </div>
          <p className="inline-note test-flow-note">{form.test_type === 'CALIBRATION'
            ? rtcMeasurement ? 'RTC 逐通道：按坐标表打开当前通道 → RTC 逐点触发完整扫频 → 读取并写盘 → 经 RTC 关闭通道并确认 → 下一通道。' : '执行逻辑：波控机按坐标表打开一个通道 → 稳定等待 → VNA 采集完整频率轴 → HDF5 确认写盘 → 关闭通道 → 下一通道。'
            : rtcMeasurement
              ? rtcContinuous
                ? 'RTC 连续：每个方位位置脉冲执行一组采集 → 行末确认排空与数量 → 读取矢网缓冲并写盘；正常暂停在整行完成后生效。'
                : 'RTC 走停：转台到点静止 → RTC 执行一组采集 → 确认排空与数量 → 读取矢网缓冲并写盘 → 下一点。'
              : acquisitionOnly
              ? '仅采集：波位由调试页或外部软件预先设置；正式扫描不访问波控机，只控制转台并采集 VNA。'
              : '执行逻辑：转台到达机械空间点 → 波控机设置独立电子波束 → 稳定等待 → VNA 一次采集完整频率轴 → HDF5 确认写盘 → 下一空间点。'}</p>
          <div className="form-grid">
          <Field label="组网模式" hint={form.test_type === 'PATTERN' ? '方向图扫描另外使用转台完成方位/俯仰运动' : undefined}><input readOnly value={rtcMeasurement ? form.test_type === 'CALIBRATION' ? '矢量网络分析仪 + RTC' : '矢量网络分析仪 + RTC + 转台' : acquisitionOnly ? '矢量网络分析仪（仅采集）' : '矢量网络分析仪 + 波控机'} /></Field>
          <Field label="采集模式"><select value={form.topology} onChange={(e) => updatePlanForm({ ...form, topology: e.target.value as RunTopology })}><option value="SOFTWARE_VNA_SWEEP">{form.test_type === 'CALIBRATION' ? '软件逐通道' : '软件走停'}</option><option value="RTC_STOP_AND_GO">{form.test_type === 'CALIBRATION' ? 'RTC 逐通道' : 'RTC 走停'}</option>{form.test_type === 'PATTERN' && <option value="RTC_CONTINUOUS">RTC 连续</option>}</select></Field>
          {form.test_type === 'PATTERN' && <Field label="波控方式"><select value={form.beam_control_mode} onChange={(e) => updatePlanForm({ ...form, beam_control_mode: e.target.value })}><option value="SOFTWARE_DIRECT">{rtcMeasurement ? '通过 RTC 设置波位' : '软件直控'}</option><option value="EXTERNAL_FIXED">仅采集</option></select></Field>}
          <Field label="收发模式" hint={signalPathHint}><select value={form.signal_path} onChange={(e) => updatePlanForm({ ...form, signal_path: e.target.value })}><option value="TX">发射（TX）</option><option value="RX">接收（RX）</option></select></Field>
          <Field label="极化"><select value={form.polarization} onChange={(e) => updatePlanForm({ ...form, polarization: e.target.value })}>{(coordinates?.polarizations ?? ['H', 'V']).map((value) => <option key={value}>{value}</option>)}</select></Field>
          <Field label="S 参数"><select value={form.s_parameter} onChange={(e) => updatePlanForm({ ...form, s_parameter: e.target.value })}>{['S11', 'S21', 'S12', 'S22'].map((value) => <option key={value}>{value}</option>)}</select></Field>
          <div className="shared-array-id-reference"><span>阵面 ID</span><strong>{arrayId ?? '未设置'}</strong><small>由页面顶部统一设置</small></div>
          <Field label="起始频率 (GHz)"><input type="number" step="0.001" value={form.frequency_start_ghz} onChange={(e) => updatePlanForm({ ...form, frequency_start_ghz: Number(e.target.value) })} /></Field>
          <Field label="终止频率 (GHz)"><input type="number" step="0.001" value={form.frequency_stop_ghz} onChange={(e) => updatePlanForm({ ...form, frequency_stop_ghz: Number(e.target.value) })} /></Field>
          <Field label="频点数"><input type="number" min="1" value={form.frequency_points} onChange={(e) => updatePlanForm({ ...form, frequency_points: Number(e.target.value) })} /></Field>
          <Field label="IFBW (Hz)"><input type="number" min="1" value={form.if_bandwidth_hz} onChange={(e) => updatePlanForm({ ...form, if_bandwidth_hz: Number(e.target.value) })} /></Field>
          <Field label="源功率 (dBm)" hint="按S参数自动作用于激励端口，并由矢网回读确认"><input type="number" min="-120" max="30" step="0.1" value={form.source_power_dbm} onChange={(e) => updatePlanForm({ ...form, source_power_dbm: Number(e.target.value) })} /></Field>
          <Field label="矢网内部平均"><select value={form.averaging_enabled ? 'ON' : 'OFF'} onChange={(e) => updatePlanForm({ ...form, averaging_enabled: e.target.value === 'ON', averaging_count: e.target.value === 'ON' ? Math.max(2, form.averaging_count) : 1 })}><option value="OFF">关闭</option><option value="ON">启用（{rtcMeasurement ? '逐点平均' : '扫频平均'}）</option></select></Field>
          <Field label="平均次数" hint={rtcMeasurement ? '矢网对每次外部触发的频点进行内部平均' : '启用后每个测量束执行对应次数的矢网内部扫频'}><input type="number" min="2" max="65536" disabled={!form.averaging_enabled} value={form.averaging_enabled ? form.averaging_count : 1} onChange={(e) => updatePlanForm({ ...form, averaging_count: Number(e.target.value) })} /></Field>
          <Field label="稳定等待 (ms)"><input type="number" min="0" value={form.settle_ms} onChange={(e) => updatePlanForm({ ...form, settle_ms: Number(e.target.value) })} /></Field>
          <div className="output-fields">
            <Field label="文件名" hint={`保存为：${form.base_filename}_${form.test_type === 'CALIBRATION' ? '标校' : '方向图'}_YYYYMMDD_HHMMSS_mmm.hdf5`}><input value={form.base_filename} onChange={(e) => updatePlanForm({ ...form, base_filename: e.target.value })} /></Field>
            <Field label="保存目录"><div className="input-button"><input readOnly value={outputDirectory} placeholder="请选择目录" /><button onClick={chooseOutput}>浏览</button></div></Field>
          </div>
          </div>
          {form.test_type === 'PATTERN' && <div className="scan-grid">
            <Field label="方位起始角 (°)"><input type="text" inputMode="decimal" value={form.azimuth_start_deg} onChange={(e) => acceptsSignedFourDecimalInput(e.target.value) && updatePlanForm({ ...form, azimuth_start_deg: e.target.value })} /></Field>
            <Field label="方位终止角 (°)"><input type="text" inputMode="decimal" value={form.azimuth_stop_deg} onChange={(e) => acceptsSignedFourDecimalInput(e.target.value) && updatePlanForm({ ...form, azimuth_stop_deg: e.target.value })} /></Field>
            <Field label="方位步进 (°)" hint={rtcContinuous ? '同时用于位置触发间隔与数量，无需重复输入' : '软件据此自动计算热力图列'}><input type="number" min="0.0001" step="0.0001" value={form.azimuth_step_deg} onChange={(e) => updatePlanForm({ ...form, azimuth_step_deg: Number(e.target.value) })} /></Field>
            <Field label="俯仰起始角 (°)"><input type="text" inputMode="decimal" value={form.elevation_start_deg} onChange={(e) => acceptsSignedFourDecimalInput(e.target.value) && updatePlanForm({ ...form, elevation_start_deg: e.target.value })} /></Field>
            <Field label="俯仰终止角 (°)"><input type="text" inputMode="decimal" value={form.elevation_stop_deg} onChange={(e) => acceptsSignedFourDecimalInput(e.target.value) && updatePlanForm({ ...form, elevation_stop_deg: e.target.value })} /></Field>
            <Field label="俯仰步进 (°)" hint="软件据此自动计算热力图行"><input type="number" min="0.0001" step="0.0001" value={form.elevation_step_deg} onChange={(e) => updatePlanForm({ ...form, elevation_step_deg: Number(e.target.value) })} /></Field>
            <Field label="转台速度 (°/s)" hint="必须大于 0，最多保留四位小数"><input type="number" min="0.0001" step="0.0001" value={form.move_speed_deg_s} onChange={(e) => acceptsFourDecimalInput(e.target.value) && updatePlanForm({ ...form, move_speed_deg_s: e.target.value })} onBlur={() => updatePlanForm({ ...form, move_speed_deg_s: positiveFourDecimals(form.move_speed_deg_s) })} /></Field>
            <Field label={acquisitionOnly ? '当前外部波位标注：离轴角, 方位角 (°)' : '电子波束方向：离轴角, 方位角 (°)'} hint={acquisitionOnly ? '仅写入结果作为用户声明，不发送、不回读波控指令；每次运行只填一项' : '多个方向用分号或换行分隔，例如 0,0; 10,30'}><textarea value={form.beam_points_text} onChange={(e) => updatePlanForm({ ...form, beam_points_text: e.target.value })} /></Field>
            {rtcMeasurement && <p className="scan-combination-note">TR 周期、高宽与延时在“设备调试”统一设置；准备时读取并冻结，未配置时采用默认值，收发方向使用本计划设置。开始前关闭 TR 调试输出。{acquisitionOnly ? '仅采集保持预设波位，RTC 只负责触发。' : ''}</p>}
            {acquisitionOnly && <p className="scan-combination-note">仅采集模式不要求连接波控机；即使已连接，自动测试也不会访问它。开始前请确认波位已经由设备调试页或外部软件设置完成并保持不变。</p>}
            <p className="scan-combination-note">软件按起始角、终止角和步进生成离散点；组合按（俯仰, 方位）执行：{elevationValues.length} × {azimuthValues.length} = {elevationValues.length * azimuthValues.length} 个机械点。</p>
          </div>}
        </fieldset>
        {rtcMeasurement && <div className="rtc-wave-manager">
          <div className="rtc-wave-heading"><b>RTC 波位存储</b><span>{acquisitionOnly ? '仅采集保持外部预设波位，无需写表' : `${visibleRtcWaves?.data?.count ?? 0} / 512 条 · 地址自动分配${visibleRtcWaves?.data?.source ? ` · ${visibleRtcWaves.data.source}` : ''}`}</span></div>
          {!acquisitionOnly && <>
            <div className="protocol-actions">
              <button className="button secondary" disabled={!waveInputs.payload || !rtcReady || planControlsDisabled} onClick={() => rtcWaveAction('write')}>写入 RTC 并校验</button>
              <button className="button ghost" disabled={!waveInputs.payload || !rtcReady || planControlsDisabled} onClick={() => rtcWaveAction('read')}>读取 RTC 波位</button>
              <span>{rtcWaveVerified ? '已回读确认与当前计划一致' : '当前计划尚未校验'}</span>
            </div>
            <p className="inline-note">复用当前{form.test_type === 'CALIBRATION' ? '坐标表中的启用通道' : '电子波束方向'}，只需连接 RTC 即可预装。准备测试会自动写入并校验，开始时核对表内容。RTC 复位后需重新写入。</p>
            {form.test_type === 'CALIBRATION' && <p className="inline-note">每通道测完经 RTC 关闭并确认后再测下一通道。</p>}
            {(waveInputs.error || visibleRtcWaves?.error) && <p className="reason">{waveInputs.error || visibleRtcWaves?.error}</p>}
            {visibleRtcWaves?.data?.entries && <div className="rtc-wave-table"><table><thead><tr><th>地址</th><th>波位 / 通道</th><th>22 字节指令</th><th>回读状态</th></tr></thead><tbody>{visibleRtcWaves.data.entries.map((entry) => <tr key={entry.address}>
              <td>{entry.address}</td><td>{entry.label}</td><td><code>{entry.frame_hex}</code>{entry.status === 'MISMATCH' && entry.readback_hex && <small>实际回读：<code>{entry.readback_hex}</code></small>}</td><td>{rtcWaveVerified ? '校验一致' : { PREVIEW: '未校验', VERIFIED: '校验一致', MATCH: '一致', MISMATCH: '不一致', EMPTY: '尚未写入' }[entry.status]}</td>
            </tr>)}</tbody></table></div>}
          </>}
        </div>}
        {run?.state === 'PREPARED' && <p className="inline-note warning">当前计划已准备；修改任一参数后，必须重新点击“准备测试”确认。</p>}
        {planLocked && <p className="inline-note warning">{run?.cleanup_pending ? '数据已保存，正在完成设备收尾；收尾结束后解锁。' : '测试已经开始，当前计划参数已锁定；本次运行结束后可再次修改。'}</p>}
      </Card>
      <div className="grid run-grid">
        <Card title="运行控制" eyebrow="04 · CONTROL">
          <div className="run-state"><div><span>当前状态</span><StatusPill value={run?.state ?? '未准备'} /></div><strong>{run ? `${run.completed} / ${run.total}` : '—'}</strong></div>
          <div className="progress"><i style={{ width: `${(run?.progress ?? 0) * 100}%` }} /></div>
          <div className="control-buttons">
            {!run || !['PREPARED', 'RUNNING', 'PAUSED', 'STOPPING'].includes(run.state) ? <button className="button primary" disabled={!profile || !coordinates || !ready || !outputDirectory || arrayId === null || planControlsDisabled} onClick={prepare}>准备测试</button> : null}
            {run?.state === 'PREPARED' && <button className="button primary" disabled={loading} onClick={() => runAction('start')}>开始</button>}
            {run?.state === 'RUNNING' && <button className="button secondary" disabled={loading || run.pause_requested} onClick={() => runAction('pause')}>{rtcMeasurement && run.pause_requested ? rtcContinuous ? '等待本行完成后暂停' : '等待本组完成后暂停' : '暂停'}</button>}
            {run?.state === 'PAUSED' && <button className="button primary" disabled={loading} onClick={() => runAction('resume')}>继续</button>}
            {run && ['RUNNING', 'PAUSED'].includes(run.state) && <button className="button danger" disabled={loading} onClick={() => runAction('stop')}>正常停止</button>}
          </div>
          {rtcContinuous && run?.state === 'RUNNING' && <p className="inline-note">暂停请求在本行采集、读取和写盘完成后生效。</p>}
          {!ready && <p className="reason">所需设备尚未就绪：{requiredIds.join('、')}</p>}
          {rtcMeasurement && run?.rtc_configuration && <p className="inline-note rtc-plan-summary">已冻结 TR：{run.rtc_configuration.tr.mode} · 周期 {run.rtc_configuration.tr.period_us} μs · 高宽 {run.rtc_configuration.tr.high_us} μs · 延时 {run.rtc_configuration.tr.delay_us} μs。</p>}
          {run?.output_path && <p className="output-path">输出：{run.output_path}</p>}
        </Card>
        <Card title="实时测量" eyebrow="LIVE · CONFIRMED DATA">
          <div className="live-view-controls">
            {form.test_type === 'PATTERN' && <Field label="展示电子波束方向"><select value={displayedBeamId} onChange={(e) => setSelectedLiveBeamId(e.target.value)}>{liveBeams.map((beam) => <option key={beam.beam_id} value={beam.beam_id}>{beam.beam_id} · 离轴 {beam.off_axis_deg}° / 方位 {beam.azimuth_deg}°</option>)}</select></Field>}
            <Field label="展示频点"><select value={Math.min(liveFrequencyIndex, Math.max(0, liveFrequencies.length - 1))} onChange={(e) => setLiveFrequencyIndex(Number(e.target.value))}>{liveFrequencies.map((frequency, index) => <option key={index} value={index}>{index + 1} · {(frequency / 1e9).toFixed(6)} GHz</option>)}</select></Field>
          </div>
          <p className="heatmap-axis-note">{form.test_type === 'PATTERN' ? `方向图机械网格：${heatColumns} 个方位列 × ${heatRows} 个俯仰行；当前只显示所选方向和频点` : `标校通道网格：${heatColumns} 列 × ${heatRows} 行；当前只显示所选频点`}</p>
          <div
            className={`channel-map ${form.test_type === 'PATTERN' ? 'pattern-heatmap' : ''}`}
            data-channel-count={heatCount}
            data-heatmap-kind={form.test_type}
            style={{
              gridTemplateColumns: `repeat(${heatColumns}, minmax(0, 1fr))`,
              gridTemplateRows: `repeat(${heatRows}, minmax(0, 1fr))`
            }}
          >
            {Array.from({ length: heatCount }, (_, index) => {
              const sample = activeHeatMap.get(index)
              const db = displayedMagnitude(sample)
              const phase = displayedPhase(sample)
              const relativeDb = db != null && heatMaximum != null ? db - heatMaximum : null
              const position = form.test_type === 'CALIBRATION'
                ? `通道 ${index}`
                : `方位 ${azimuthValues[index % heatColumns]?.toFixed(2)}° · 俯仰 ${elevationValues[Math.floor(index / heatColumns)]?.toFixed(2)}°`
              return <div
                key={index}
                title={sample?.status === 'SKIPPED_DISABLED'
                  ? `${position} · 已禁用，未测量`
                  : sample && db != null
                  ? `${position} · ${db.toFixed(2)} dB · 相对 ${relativeDb?.toFixed(2)} dB${phase != null ? ` · ${phase.toFixed(2)}°` : ''}`
                  : `${position} · 未采集`}
                style={{ background: relativeHeatColor(db, heatMaximum) }}
              />
            })}
          </div>
          <div className="legend heat-legend">
            <span><i className="empty" />未采集</span>
            <div className="relative-scale" aria-label="相对最高幅度颜色指示">
              <i />
              <div><b>≤ -30 dB</b><span>相对最高幅度</span><b>0 dB（最高）</b></div>
            </div>
            <span>{heatMaximum == null ? '尚无测量值' : `当前最高 ${heatMaximum.toFixed(2)} dB`}</span>
          </div>
        </Card>
      </div>
    </>
  )
}

type ChartSeries = { label: string; color: string; points: Array<{ x: number; y: number }> }

function chartColor(index: number): string {
  return `hsl(${(index * 137.508) % 360} 72% 55%)`
}

function CartesianLineChart({ series }: { series: ChartSeries[] }) {
  const width = 900; const height = 310
  const margin = { left: 62, right: 22, top: 18, bottom: 42 }
  const all = series.flatMap((item) => item.points).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
  if (all.length === 0) return <div className="empty-log">当前频点没有可绘制的方向图数据</div>
  let xMin = Math.min(...all.map((point) => point.x)); let xMax = Math.max(...all.map((point) => point.x))
  let yMin = Math.min(...all.map((point) => point.y)); let yMax = Math.max(...all.map((point) => point.y))
  if (xMin === xMax) { xMin -= 1; xMax += 1 }
  if (yMin === yMax) { yMin -= 1; yMax += 1 }
  const yPadding = Math.max(1, (yMax - yMin) * 0.08); yMin -= yPadding; yMax += yPadding
  const plotWidth = width - margin.left - margin.right; const plotHeight = height - margin.top - margin.bottom
  const sx = (value: number) => margin.left + (value - xMin) / (xMax - xMin) * plotWidth
  const sy = (value: number) => margin.top + (yMax - value) / (yMax - yMin) * plotHeight
  const ticks = Array.from({ length: 5 }, (_, index) => index / 4)
  return <div className="line-chart-wrap">
    <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="直角坐标二维方向图">
      {ticks.map((ratio) => {
        const y = margin.top + ratio * plotHeight; const value = yMax - ratio * (yMax - yMin)
        return <g key={`y-${ratio}`}><line x1={margin.left} y1={y} x2={width - margin.right} y2={y} className="chart-grid-line" /><text x={margin.left - 9} y={y + 4} textAnchor="end">{value.toFixed(1)}</text></g>
      })}
      {ticks.map((ratio) => {
        const x = margin.left + ratio * plotWidth; const value = xMin + ratio * (xMax - xMin)
        return <g key={`x-${ratio}`}><line x1={x} y1={margin.top} x2={x} y2={height - margin.bottom} className="chart-grid-line" /><text x={x} y={height - margin.bottom + 20} textAnchor="middle">{value.toFixed(1)}</text></g>
      })}
      <text className="chart-axis-title" x={margin.left + plotWidth / 2} y={height - 6} textAnchor="middle">转台方位角 (°)</text>
      <text className="chart-axis-title" transform={`translate(14 ${margin.top + plotHeight / 2}) rotate(-90)`} textAnchor="middle">原始幅度 (dB)</text>
      {series.map((item) => {
        const points = item.points.filter((point) => Number.isFinite(point.y))
        const path = points.map((point, index) => `${index ? 'L' : 'M'} ${sx(point.x).toFixed(2)} ${sy(point.y).toFixed(2)}`).join(' ')
        return <path key={item.label} d={path} fill="none" stroke={item.color} strokeWidth="2" />
      })}
    </svg>
    <div className="chart-legend">{series.map((item) => <span key={item.label}><i style={{ background: item.color }} />{item.label}</span>)}</div>
  </div>
}

function HistoryPage({ notify }: { notify: (level: Message['level'], text: string) => void }) {
  const [result, setResult] = useState<Record<string, any> | null>(null)
  const [view, setView] = useState<Record<string, any> | null>(null)
  const [frequencyIndex, setFrequencyIndex] = useState(0)
  const [beamIndex, setBeamIndex] = useState(0)
  const choose = async () => {
    try {
      const path = await desktopBridge().openFile('data')
      if (!path) return
      const inspected = await post<Record<string, any>>('/api/data/inspect', { path })
      setResult(inspected); setView(null); setFrequencyIndex(0); setBeamIndex(0)
    } catch (error) { notify('error', errorMessage(error)) }
  }
  useEffect(() => {
    if (!result?.path || !['PATTERN', 'CALIBRATION'].includes(result.analysis?.kind)) return
    let cancelled = false
    post<Record<string, any>>('/api/data/view', { path: result.path, frequency_index: frequencyIndex, beam_index: result.analysis.kind === 'PATTERN' ? -1 : 0 })
      .then((data) => { if (!cancelled) setView(data) })
      .catch((error) => { if (!cancelled) notify('error', errorMessage(error)) })
    return () => { cancelled = true }
  }, [result, frequencyIndex, notify])

  const patternPoints = view?.kind === 'PATTERN' ? view.points as Array<Record<string, any>> : []
  const visiblePatternPoints = patternPoints.filter((point) => point.beam_index === beamIndex)
  const patternRows = visiblePatternPoints.length ? Math.max(...visiblePatternPoints.map((point) => point.row_index)) + 1 : 1
  const patternColumns = visiblePatternPoints.length ? Math.max(...visiblePatternPoints.map((point) => point.point_index)) + 1 : 1
  const patternMap = new Map(visiblePatternPoints.map((point) => [point.row_index * patternColumns + point.point_index, point]))
  const patternMagnitudes = visiblePatternPoints.map((point) => point.magnitude_db).filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
  const patternMaximum = patternMagnitudes.length ? Math.max(...patternMagnitudes) : null
  const historyBeams = view?.beams ?? []
  const directionSeries: ChartSeries[] = historyBeams.flatMap((beam: any, directionIndex: number) => {
    const beamPoints = patternPoints.filter((point) => point.beam_index === beam.beam_index)
    const rowCount = beamPoints.length ? Math.max(...beamPoints.map((point) => point.row_index)) + 1 : 0
    return Array.from({ length: rowCount }, (_, rowIndex) => {
      const row = beamPoints.filter((point) => point.row_index === rowIndex).sort((a, b) => a.point_index - b.point_index)
      return {
        label: `${beam.beam_id}（离轴 ${beam.off_axis_deg}° / 方位 ${beam.azimuth_deg}°）· 俯仰 ${row[0]?.elevation_deg ?? '—'}°`,
        color: chartColor(directionIndex + rowIndex * Math.max(1, historyBeams.length)),
        points: row.filter((point) => typeof point.magnitude_db === 'number').map((point) => ({ x: point.azimuth_deg, y: point.magnitude_db }))
      }
    })
  }).filter((series: ChartSeries) => series.points.length > 0)

  return <>
    <div className="page-lead"><span className="eyebrow">OFFLINE</span><h2>历史测量分析</h2><p>方向图可按电子波束和频点查看原始热力图及二维曲线；标校文件可查看每个通道的原始幅度、相位和最大幅度差。</p><button className="button primary" onClick={choose}>选择 HDF5 数据文件</button></div>
    {result && <>
      <div className="grid two"><Card title="文件身份" eyebrow="INTEGRITY"><dl className="facts"><dt>路径</dt><dd>{result.path}</dd><dt>SHA-256</dt><dd>{shortHash(result.sha256)}</dd><dt>完整性</dt><dd><StatusPill value={result.integrity} /></dd><dt>Schema</dt><dd>{result.schema?.schema_name} / {result.schema?.schema_version}</dd><dt>角色与状态</dt><dd>{result.schema?.file_role ?? '—'} · {result.schema?.status}</dd></dl></Card><Card title="HDF5 数据集" eyebrow="DATASETS"><div className="table-counts">{Object.entries(result.tables ?? {}).map(([name, count]) => <div key={name}><span>{name}</span><strong>{String(count)}</strong></div>)}</div></Card></div>
      {['PATTERN', 'CALIBRATION'].includes(result.analysis?.kind) && <Card title="分析选择" eyebrow="VIEW"><div className="history-controls"><Field label="频点"><select value={frequencyIndex} onChange={(e) => setFrequencyIndex(Number(e.target.value))}>{(result.analysis.frequencies_hz ?? []).map((frequency: number, index: number) => <option key={index} value={index}>{index + 1} · {(frequency / 1e9).toFixed(6)} GHz</option>)}</select></Field>{result.analysis.kind === 'PATTERN' && <Field label="原始热力图电子波束"><select value={beamIndex} onChange={(e) => setBeamIndex(Number(e.target.value))}>{(result.analysis.beams ?? []).map((beam: any) => <option key={beam.beam_index} value={beam.beam_index}>{beam.beam_id} · 离轴 {beam.off_axis_deg}° / 方位 {beam.azimuth_deg}°</option>)}</select></Field>}</div></Card>}
      {view?.kind === 'PATTERN' && <div className="history-analysis-grid">
        <Card title="原始方向图热力图" eyebrow="RAW HEATMAP"><p className="heatmap-axis-note">{patternColumns} 个方位列 × {patternRows} 个俯仰行 · {(view.frequency_hz / 1e9).toFixed(6)} GHz</p><div className="history-heatmap" style={{ gridTemplateColumns: `repeat(${patternColumns}, minmax(0,1fr))`, gridTemplateRows: `repeat(${patternRows}, minmax(0,1fr))` }}>{Array.from({ length: patternRows * patternColumns }, (_, index) => { const point = patternMap.get(index); const db = point?.magnitude_db; return <div key={index} title={point && db != null ? `俯仰 ${point.elevation_deg}° · 方位 ${point.azimuth_deg}° · ${db.toFixed(2)} dB` : '未采集'} style={{ background: relativeHeatColor(db, patternMaximum) }} /> })}</div><div className="legend heat-legend"><div className="relative-scale"><i /><div><b>≤ -30 dB</b><span>相对最高幅度</span><b>0 dB</b></div></div></div></Card>
        <Card title="直角坐标二维方向图（当前频点全部电子方向）" eyebrow="CARTESIAN"><CartesianLineChart series={directionSeries} /></Card>
      </div>}
      {view?.kind === 'CALIBRATION' && <Card title="各通道初始幅度与相位" eyebrow="CALIBRATION RAW"><div className="calibration-summary"><span>当前频点</span><strong>{(view.frequency_hz / 1e9).toFixed(6)} GHz</strong><span>最大幅度差</span><strong>{view.max_amplitude_difference_db == null ? '—' : `${view.max_amplitude_difference_db.toFixed(3)} dB`}</strong></div><div className="calibration-history-table"><table><thead><tr><th>通道</th><th>状态</th><th>初始幅度 (dB)</th><th>初始相位 (°)</th></tr></thead><tbody>{view.channels.map((channel: any) => <tr key={channel.element}><td>{channel.element}</td><td>{channel.status}</td><td>{channel.magnitude_db == null ? '—' : channel.magnitude_db.toFixed(4)}</td><td>{channel.phase_deg == null ? '—' : channel.phase_deg.toFixed(4)}</td></tr>)}</tbody></table></div></Card>}
    </>}
  </>
}

const flashAddressItems: Array<{ name: FlashItemName; label: string }> = [
  { name: 'ARRAY_ID', label: '阵面 ID' },
  { name: 'SWITCH_TABLE', label: '开关表' },
  { name: 'COORDINATE', label: '坐标数据' },
  { name: 'TX_COMPENSATION', label: 'TX 补偿' },
  { name: 'RX_COMPENSATION', label: 'RX 补偿' }
]

function CompensationPage({ assets, devices, arrayId, onArrayIdBusyChange, notify }: {
  assets: Assets
  devices: DeviceStatus[]
  arrayId: number | null
  onArrayIdBusyChange: (source: string, busy: boolean) => void
  notify: (level: Message['level'], text: string) => void
}) {
  const coordinates = assets.coordinates[assets.coordinates.length - 1]
  const [pathMode, setPathMode] = useState<'TX' | 'RX'>('TX')
  const [calibrationFiles, setCalibrationFiles] = useState<Record<string, string>>({})
  const [calibrationFrequencies, setCalibrationFrequencies] = useState<number[]>([])
  const [selectedCompFrequencyIndices, setSelectedCompFrequencyIndices] = useState<number[]>([])
  const [sidelobeAlgorithm, setSidelobeAlgorithm] = useState<'NONE' | 'TAYLOR' | 'HAMMING' | 'KAISER'>('NONE')
  const [sidelobeAxes, setSidelobeAxes] = useState<'ROW' | 'COLUMN' | 'BOTH'>('BOTH')
  const [taylorNbar, setTaylorNbar] = useState('4')
  const [taylorSllDb, setTaylorSllDb] = useState('30')
  const [kaiserBeta, setKaiserBeta] = useState('6')
  const [txComp, setTxComp] = useState('')
  const [rxComp, setRxComp] = useState('')
  const [flashPackage, setFlashPackage] = useState('')
  const [flashItems, setFlashItems] = useState<any[]>([])
  const [flashPreparedFor, setFlashPreparedFor] = useState('')
  const [flashAddresses, setFlashAddresses] = useState<Record<FlashItemName, string>>({
    ARRAY_ID: '0x0000', SWITCH_TABLE: '0x1000', COORDINATE: '0x2000',
    TX_COMPENSATION: '0x4000', RX_COMPENSATION: '0x6000'
  })
  const [busy, setBusy] = useState(false)
  const flashInputKey = JSON.stringify([coordinates?.asset_id, arrayId, txComp, rxComp, flashAddresses])
  const flashInputVersion = useRef({ key: flashInputKey, revision: 0 })
  const coordinateContext = useRef(coordinates?.asset_id)
  coordinateContext.current = coordinates?.asset_id
  if (flashInputVersion.current.key !== flashInputKey) {
    flashInputVersion.current = { key: flashInputKey, revision: flashInputVersion.current.revision + 1 }
  }
  const visibleFlashItems = flashPreparedFor === flashInputKey ? flashItems : []
  const allCalibrationsSelected = Boolean(coordinates?.polarizations.every((polarization) => calibrationFiles[polarization]))

  useEffect(() => {
    onArrayIdBusyChange('compensation', busy)
    return () => onArrayIdBusyChange('compensation', false)
  }, [busy, onArrayIdBusyChange])

  useEffect(() => {
    if (flashPackage || flashItems.length > 0) {
      notify('info', 'FLASH 输入已修改，原准备结果已失效，请重新生成')
    }
    setFlashPackage('')
    setFlashItems([])
    setFlashPreparedFor('')
  }, [flashInputKey])

  useEffect(() => {
    setCalibrationFiles({})
    setCalibrationFrequencies([])
    setSelectedCompFrequencyIndices([])
  }, [coordinates?.asset_id, pathMode])

  const chooseHdf = async (setter: (value: string) => void) => {
    setBusy(true)
    try {
      const path = await desktopBridge().openFile('hdf')
      if (path) {
        setter(path)
        setFlashPackage('')
        setFlashItems([])
      }
    } catch (error) { notify('error', errorMessage(error)) } finally { setBusy(false) }
  }
  const chooseCalibration = async (polarization: string) => {
    setBusy(true)
    try {
      const path = await desktopBridge().openFile('hdf')
      if (!path) return
      const inspected = await post<Record<string, any>>('/api/data/inspect', { path })
      if (coordinateContext.current !== coordinates?.asset_id) return
      if (inspected.analysis?.kind !== 'CALIBRATION') throw new Error('请选择完整的标校 HDF5 文件')
      if (inspected.metadata?.signal_path !== pathMode || inspected.metadata?.polarization !== polarization) {
        throw new Error(`请选择 ${pathMode} / ${polarization} 标校文件`)
      }
      const frequencies = (inspected.analysis.frequencies_hz ?? []) as number[]
      if (frequencies.length === 0) throw new Error('标校文件中没有可用频点')
      const hasOtherPolarization = Object.keys(calibrationFiles).some((key) => key !== polarization)
      if (hasOtherPolarization && JSON.stringify(frequencies) !== JSON.stringify(calibrationFrequencies)) {
        throw new Error('H、V 标校文件的频率轴必须一致')
      }
      setCalibrationFiles((current) => ({ ...current, [polarization]: path }))
      setCalibrationFrequencies(frequencies)
      if (!hasOtherPolarization) setSelectedCompFrequencyIndices(frequencies.map((_, index) => index))
    } catch (error) { notify('error', errorMessage(error)) } finally { setBusy(false) }
  }
  const changePathMode = (mode: 'TX' | 'RX') => {
    setPathMode(mode)
    setCalibrationFiles({})
    setCalibrationFrequencies([])
    setSelectedCompFrequencyIndices([])
  }
  const toggleCompFrequency = (index: number) => {
    setSelectedCompFrequencyIndices((current) => current.includes(index)
      ? current.filter((item) => item !== index)
      : [...current, index])
  }
  const updateAddress = (name: FlashItemName, value: string) => {
    if (value === '' || /^(?:0[xX])?[0-9a-fA-F]*$/.test(value)) {
      setFlashAddresses((current) => ({ ...current, [name]: value }))
      setFlashPackage('')
      setFlashItems([])
    }
  }
  const parseAddress = (value: string, label: string): number => {
    const normalized = value.trim()
    if (!/^(?:0[xX])?[0-9a-fA-F]+$/.test(normalized)) throw new Error(`${label}起始地址必须是十六进制`)
    const parsed = Number.parseInt(normalized.replace(/^0[xX]/, ''), 16)
    if (!Number.isSafeInteger(parsed) || parsed < 0 || parsed > 0xFFFF00) throw new Error(`${label}起始地址必须在 0x000000..0xFFFF00 范围内`)
    if (parsed % 0x100 !== 0) throw new Error(`${label}起始地址必须按 0x100 页对齐`)
    return parsed
  }
  const displayAddress = (value: number) => `0x${value.toString(16).toUpperCase().padStart(6, '0')}`

  const generate = async () => {
    if (!coordinates || !allCalibrationsSelected || selectedCompFrequencyIndices.length === 0) return
    setBusy(true)
    try {
      const nbar = Number(taylorNbar)
      const sidelobeLevel = Number(taylorSllDb)
      const beta = Number(kaiserBeta)
      if (sidelobeAlgorithm === 'TAYLOR' && (!Number.isInteger(nbar) || nbar < 2 || nbar > 20)) throw new Error('Taylor nbar 必须是 2..20 的整数')
      if (sidelobeAlgorithm === 'TAYLOR' && (!Number.isFinite(sidelobeLevel) || sidelobeLevel < 10 || sidelobeLevel > 80)) throw new Error('Taylor 目标副瓣必须是 10..80 dB')
      if (sidelobeAlgorithm === 'KAISER' && (!Number.isFinite(beta) || beta < 0 || beta > 20)) throw new Error('Kaiser β 必须是 0..20')
      const output = await desktopBridge().saveFile('hdf5', `${pathMode.toLowerCase()}_compensation.hdf5`)
      if (!output) return
      const result = await post<any>('/api/compensation/generate', {
        signal_path: pathMode, coordinate_id: coordinates.asset_id,
        calibration_files: coordinates.polarizations.map((polarization) => calibrationFiles[polarization]), frequency_indices: selectedCompFrequencyIndices, output_path: output,
        phase_step_deg: 5.625, attenuation_step_db: 0.5, max_attenuation_db: 31.5,
        sidelobe_algorithm: sidelobeAlgorithm, sidelobe_axes: sidelobeAxes,
        taylor_nbar: nbar, taylor_sll_db: sidelobeLevel, kaiser_beta: beta
      })
      if (coordinateContext.current !== coordinates.asset_id) {
        notify('info', '生成期间坐标表已变化，请为当前坐标表重新生成补偿')
        return
      }
      pathMode === 'TX' ? setTxComp(result.path) : setRxComp(result.path)
      setFlashPackage('')
      setFlashItems([])
      const weighting = result.aperture_weighting
      const weightingText = weighting.algorithm === 'NONE' ? '未启用副瓣压低' : `${weighting.algorithm}/${weighting.axes}，最大孔径衰减 ${weighting.maximum_taper_db.toFixed(3)} dB`
      notify('success', `${pathMode} 补偿 HDF5 已生成：${result.frequencies_hz.length} 个升序频点；${weightingText}`)
    } catch (error) { notify('error', errorMessage(error)) } finally { setBusy(false) }
  }
  const prepareFlash = async () => {
    if (!coordinates || !txComp || !rxComp || arrayId === null) return
    const inputRevision = flashInputVersion.current.revision
    const preparedKey = flashInputKey
    setBusy(true)
    try {
      const startAddresses = Object.fromEntries(flashAddressItems.map(({ name, label }) => [name, parseAddress(flashAddresses[name], label)]))
      const output = await desktopBridge().saveFile('hdf5', 'antenna_flash_package.hdf5')
      if (!output) return
      const result = await post<any>('/api/flash/prepare', {
        coordinate_id: coordinates.asset_id,
        tx_compensation_file: txComp,
        rx_compensation_file: rxComp,
        output_path: output,
        array_id: arrayId,
        start_addresses: startAddresses
      })
      if (inputRevision !== flashInputVersion.current.revision) {
        notify('info', '准备期间 FLASH 输入已变化，请按当前输入重新准备')
        return
      }
      setFlashPreparedFor(preparedKey)
      setFlashPackage(result.path)
      setFlashItems(result.items)
      notify('success', `阵面 ${arrayId} 的五类 FLASH 数据已按页编码，终止地址已计算`)
    } catch (error) { notify('error', errorMessage(error)) } finally { setBusy(false) }
  }
  const writeItem = async (name: string) => {
    if (busy || !flashPackage || flashPreparedFor !== flashInputVersion.current.key) return
    setBusy(true)
    try {
      const result = await post<any>('/api/flash/write', { package_path: flashPackage, item_name: name, device_id: 'beam_controller' })
      setFlashItems((items) => items.map((item) => item.item_name === name ? { ...item, status: result.status } : item))
      notify('success', `${name} 写入并全量读回一致`)
    } catch (error) { notify('error', errorMessage(error)) } finally { setBusy(false) }
  }
  const beam = devices.find((device) => device.device_id === 'beam_controller')

  return <fieldset className="plan-fields" disabled={busy}>
    <div className="page-lead"><span className="eyebrow">POST PROCESS</span><h2>生成 HDF5 补偿，逐项写入 FLASH</h2><p>TX、RX 补偿及五类 FLASH 工程数据均保存为普通 HDF5 数据集。</p></div>
    <div className="grid two">
      <Card title="生成补偿" eyebrow="COMPENSATION">
        <div className="segmented"><button className={pathMode === 'TX' ? 'selected' : ''} onClick={() => changePathMode('TX')}>TX</button><button className={pathMode === 'RX' ? 'selected' : ''} onClick={() => changePathMode('RX')}>RX</button></div>
        {(coordinates?.polarizations ?? ['H']).map((polarization) => <div className="asset-row" key={polarization}><div className="asset-icon">CAL</div><div className="asset-copy"><b>{pathMode} / {polarization}</b><span>{calibrationFiles[polarization] || `选择 ${pathMode} / ${polarization} 完整标校 HDF5 文件`}</span></div><button className="button secondary" onClick={() => chooseCalibration(polarization)}>选择</button></div>)}
        {calibrationFrequencies.length > 0 && <div className="comp-frequency-picker">
          <div className="comp-frequency-head"><span>补偿及 FLASH 上注频点</span><small>已选 {selectedCompFrequencyIndices.length} / {calibrationFrequencies.length}，保存时按频率升序排列</small><button onClick={() => setSelectedCompFrequencyIndices(calibrationFrequencies.map((_, index) => index))}>全选</button><button onClick={() => setSelectedCompFrequencyIndices([])}>清空</button></div>
          <div className="comp-frequency-grid">{calibrationFrequencies.map((frequency, index) => <label key={index}><input type="checkbox" checked={selectedCompFrequencyIndices.includes(index)} onChange={() => toggleCompFrequency(index)} /><span>{index + 1}</span><b>{(frequency / 1e9).toFixed(6)} GHz</b></label>)}</div>
        </div>}
        <div className="comp-weighting-panel">
          <div className="comp-weighting-grid">
            <Field label="副瓣压低算法"><select value={sidelobeAlgorithm} onChange={(e) => setSidelobeAlgorithm(e.target.value as typeof sidelobeAlgorithm)}><option value="NONE">不压低（均匀孔径）</option><option value="TAYLOR">Taylor</option><option value="HAMMING">Hamming</option><option value="KAISER">Kaiser</option></select></Field>
            <Field label="作用方向"><select value={sidelobeAxes} disabled={sidelobeAlgorithm === 'NONE'} onChange={(e) => setSidelobeAxes(e.target.value as typeof sidelobeAxes)}><option value="ROW">阵面行</option><option value="COLUMN">阵面列</option><option value="BOTH">阵面行 × 列</option></select></Field>
            {sidelobeAlgorithm === 'TAYLOR' && <><Field label="Taylor nbar"><input type="number" min="2" max="20" step="1" value={taylorNbar} onChange={(e) => setTaylorNbar(e.target.value)} /></Field><Field label="目标副瓣压低 (dB)"><input type="number" min="10" max="80" step="0.1" value={taylorSllDb} onChange={(e) => setTaylorSllDb(e.target.value)} /></Field></>}
            {sidelobeAlgorithm === 'KAISER' && <Field label="Kaiser β"><input type="number" min="0" max="20" step="0.1" value={kaiserBeta} onChange={(e) => setKaiserBeta(e.target.value)} /></Field>}
          </div>
          <p>孔径加权只叠加幅度衰减，不改变标校相位；标校均衡 + 孔径衰减超过 31.5 dB 时停止生成并指出通道。</p>
        </div>
        <button className="button primary wide" disabled={!coordinates || !allCalibrationsSelected || selectedCompFrequencyIndices.length === 0 || busy} onClick={generate}>按所选频点生成 {pathMode} 补偿 HDF5</button>
      </Card>
      <Card title="FLASH 输入" eyebrow="PACKAGE">
        <div className="file-line"><span>TX 补偿</span><b>{txComp || '未选择 HDF5'}</b><button onClick={() => chooseHdf(setTxComp)}>选择</button></div>
        <div className="file-line"><span>RX 补偿</span><b>{rxComp || '未选择 HDF5'}</b><button onClick={() => chooseHdf(setRxComp)}>选择</button></div>
        <div className="file-line"><span>坐标表</span><b>{coordinates?.antenna_id ?? '未加载'}</b></div>
        <div className="shared-array-id-reference"><span>阵面 ID / tile_id</span><strong>{arrayId ?? '未设置'}</strong><small>由页面顶部统一设置；准备后固定写入 HDF5 包</small></div>
        <button className="button primary wide" disabled={!coordinates || !txComp || !rxComp || arrayId === null || busy} onClick={prepareFlash}>准备五类 FLASH HDF5 数据</button>
      </Card>
    </div>
    <Card title="FLASH HDF5 项目" eyebrow="WRITE & READBACK">
      <p className="inline-note">在每个项目卡片内输入 HEX 起始地址；准备后软件计算终止地址，并把五类载荷保存到同一个 HDF5 文件。{flashPackage && ` 当前文件：${flashPackage}`}</p>
      <div className="flash-grid">{flashAddressItems.map(({ name, label }) => {
        const item = visibleFlashItems.find((current) => current.item_name === name)
        return <div className="flash-item" key={name}>
          <span>{name}</span><StatusPill value={item?.status ?? '未准备'} />
          <Field label={`${label === '阵面 ID' ? '阵面ID数据项' : label}起始地址`} hint="HEX"><input value={flashAddresses[name]} onChange={(e) => updateAddress(name, e.target.value)} /></Field>
          <dl><dt>终止</dt><dd>{item ? displayAddress(item.end_address) : '准备后计算'}</dd><dt>有效</dt><dd>{item ? `${item.effective_length} B` : '—'}</dd><dt>占用</dt><dd>{item ? `${item.occupied_length} B` : '—'}</dd><dt>补零</dt><dd>{item ? `${item.padding_length} B` : '—'}</dd><dt>页数</dt><dd>{item?.page_count ?? '—'}</dd></dl>
          <button className="button secondary wide" disabled={!item || !beam || busy || !['READY', 'CONNECTED'].includes(beam.state) || item.status === 'SUCCESS'} onClick={() => writeItem(name)}>{item?.status === 'SUCCESS' ? '已验证' : '写入并读回'}</button>
        </div>
      })}</div>
    </Card>
  </fieldset>
}

function DevicesPage({ profiles, devices, setDevices, logs, clearLogs, arrayId, controlLocked = false, onSettingsChange = () => {}, onArrayIdBusyChange, notify }: {
  profiles: ProfileSummary[]
  devices: DeviceStatus[]
  setDevices: React.Dispatch<React.SetStateAction<DeviceStatus[]>>
  logs: DeviceRawLog[]
  clearLogs: () => void
  controlLocked?: boolean
  onSettingsChange?: () => void
  arrayId: number | null
  onArrayIdBusyChange: (source: string, busy: boolean) => void
  notify: (level: Message['level'], text: string) => void
}) {
  const [sources, setSources] = useState<Record<string, 'REAL' | 'SIMULATED'>>({
    beam_controller: 'REAL', vna: 'REAL', turntable: 'REAL', rtc: 'REAL'
  })
  const [parameters, setParameters] = useState<Record<string, any>>({
    beam_controller: { port: '', baud_rate: 115200 },
    vna: { resource: 'TCPIP0::192.168.1.100::hislip0::INSTR', timeout_ms: 10000 },
    rtc: { port: '', baud_rate: 115200 },
    turntable: { dll_path: '', controller_ip: '192.168.1.101', device_number: 0 }
  })
  const [serialPorts, setSerialPorts] = useState<SerialPortInfo[]>([])
  const [serialLoading, setSerialLoading] = useState(false)
  const [axis, setAxis] = useState(1)
  const [target, setTarget] = useState('0')
  const [speed, setSpeed] = useState('1.0000')
  const [turntableReadback, setTurntableReadback] = useState<TurntableReadback | null>(null)
  const [turntableMotionBusy, setTurntableMotionBusy] = useState(false)
  const [commandId, setCommandId] = useState('')
  const [commandInputs, setCommandInputs] = useState<Record<string, string>>({})
  const [exchangeBusy, setExchangeBusy] = useState(false)
  const [sendTransport, setSendTransport] = useState<'SERIAL' | 'RTC'>('SERIAL')
  const [rtcBusy, setRtcBusy] = useState(false)
  const [trInputs, setTrInputs] = useState({ mode: 'TX', period_us: '100', high_us: '20', delay_us: '1' })
  const [rtcReadback, setRtcReadback] = useState<Record<string, unknown> | null>(null)
  const [logDevice, setLogDevice] = useState('ALL')
  const [logDirection, setLogDirection] = useState('ALL')
  const [exchangeResult, setExchangeResult] = useState<Record<string, any> | null>(null)
  const profile = profiles[profiles.length - 1]
  const profileCommands = profile?.commands ?? []
  const selectedCommand = profileCommands.find((item) => item.command_id === commandId)
  const editableFields = selectedCommand?.fields.filter((field) => !['CONSTANT', 'MIRROR_FIELD'].includes(field.source.toUpperCase())) ?? []
  const beam = devices.find((device) => device.device_id === 'beam_controller')
  const beamReady = beam != null && ['READY', 'CONNECTED'].includes(beam.state)
  const rtc = devices.find((device) => device.device_id === 'rtc')
  const rtcReady = rtc != null && ['READY', 'CONNECTED'].includes(rtc.state)
  const rtcAvailable = rtc != null && rtc.state !== 'DISCONNECTED'
  const sendReady = sendTransport === 'RTC' ? rtcReady : beamReady
  const visibleLogs = logs.filter((item) => (logDevice === 'ALL' || item.device_id === logDevice) && (logDirection === 'ALL' || item.direction === logDirection))
  const turntable = devices.find((device) => device.device_id === 'turntable')
  const turntableReady = turntable != null && ['READY', 'CONNECTED'].includes(turntable.state)
  const turntableRecoveryAvailable = turntable != null && ['READY', 'CONNECTED', 'UNKNOWN', 'FAULT'].includes(turntable.state)
  const speedValue = Number(speed)
  const targetValue = finiteTarget(target)

  useEffect(() => {
    if (profileCommands.length === 0) {
      setCommandId('')
    } else if (!profileCommands.some((item) => item.command_id === commandId)) {
      setCommandId(profileCommands[0].command_id)
    }
  }, [profileCommands, commandId])

  useEffect(() => {
    if (!selectedCommand) return
    setCommandInputs(Object.fromEntries(editableFields.map((field) => [
      field.key,
      field.default == null ? '' : String(field.default)
    ])))
    setExchangeResult(null)
  }, [profile?.asset_id, selectedCommand])

  useEffect(() => {
    if (exchangeResult) notify('info', '当前阵面 ID 已修改，原协议编译预览已清除')
    setExchangeResult(null)
  }, [arrayId])

  useEffect(() => {
    onArrayIdBusyChange('protocol', exchangeBusy || rtcBusy)
    return () => onArrayIdBusyChange('protocol', false)
  }, [exchangeBusy, rtcBusy, onArrayIdBusyChange])

  useEffect(() => {
    const bridge = window.antennaDesktop
    if (!bridge) {
      notify('error', 'Electron preload 桥未加载；设备页已保持可见，请查看启动终端中的 preload 错误')
      return
    }
    // The path is supplied internally to the service and is deliberately not editable.
    void bridge.appPaths()
      .then((paths) => setParameters((current) => ({
        ...current,
        turntable: { ...current.turntable, dll_path: paths.turntableDll }
      })))
      .catch((error) => notify('error', errorMessage(error)))
  }, [notify])

  const refreshSerialPorts = useCallback(async (announce = true) => {
    setSerialLoading(true)
    try {
      // Discovery is read-only: the backend enumerates Windows COM devices but does not
      // open a port until the operator presses the explicit connect button below.
      const result = await api<SerialPortInfo[]>('/api/devices/serial-ports')
      setSerialPorts(result)
      setParameters((current) => {
        const currentPort = current.beam_controller.port as string
        const port = result.some((item) => item.device === currentPort)
          ? currentPort
          : (result[0]?.device ?? '')
        const rtcPort = result.some((item) => item.device === current.rtc.port) ? current.rtc.port : ''
        return { ...current, beam_controller: { ...current.beam_controller, port }, rtc: { ...current.rtc, port: rtcPort } }
      })
      if (announce) notify('info', result.length > 0 ? `发现 ${result.length} 个可用串口` : '未发现可用串口')
    } catch (error) {
      notify('error', `串口扫描失败：${errorMessage(error)}`)
    } finally {
      setSerialLoading(false)
    }
  }, [notify])

  useEffect(() => { void refreshSerialPorts(false) }, [refreshSerialPorts])

  const connect = async (deviceId: string) => {
    if (controlLocked) return
    onSettingsChange()
    try {
      // This is the only manual connection path. For REAL sources the Python adapter
      // opens the selected hardware and must complete identity/status readback.
      const result = await post<DeviceStatus>(`/api/devices/${deviceId}/connect`, {
        source: sources[deviceId],
        parameters: parameters[deviceId]
      })
      setDevices((items) => items.map((item) => item.device_id === deviceId ? result : item))
      if (deviceId === 'turntable') setTurntableReadback(null)
      notify('success', `${deviceNames[deviceId]}已连接：${result.identity}`)
    } catch (error) {
      notify('error', errorMessage(error))
    }
  }

  const disconnect = async (deviceId: string) => {
    if (controlLocked) return
    onSettingsChange()
    try {
      // Explicit disconnect releases the serial/VISA/turntable handle before the
      // connection parameters become editable again.
      const result = await post<DeviceStatus>(`/api/devices/${deviceId}/disconnect`, {})
      setDevices((items) => items.map((item) => item.device_id === deviceId ? result : item))
      if (deviceId === 'turntable') setTurntableReadback(null)
      notify('success', `${deviceNames[deviceId]}已断开`)
    } catch (error) {
      notify('error', errorMessage(error))
    }
  }

  const command = async (action: string, args: Record<string, unknown> = {}) => {
    if (controlLocked && action !== 'read_axes') return
    const isMotion = action === 'move_to' || action === 'home'
    if (isMotion) setTurntableMotionBusy(true)
    try {
      // Motion requests are sent to the already-connected turntable adapter. The backend
      // applies the fixed scale 10000. The long wait does not block a concurrent read_axes
      // or Stop request; duplicate move/home commands remain disabled in the UI.
      const result = await post<any>('/api/devices/turntable/command', { action, parameters: args })
      if (result?.positions && result?.velocities) {
        setTurntableReadback({ positions: result.positions, velocities: result.velocities })
      }
      if (action === 'read_axes') {
        notify('success', '已读取转台五轴当前位置与速度')
      } else {
        notify('success', `转台 ${action} 完成：${JSON.stringify(result)}`)
      }
    } catch (error) {
      notify('error', errorMessage(error))
    } finally {
      if (isMotion) setTurntableMotionBusy(false)
    }
  }

  const executeProfileCommand = async (send: boolean) => {
    if (!profile || !selectedCommand || arrayId === null || exchangeBusy || (send && (controlLocked || !sendReady))) return
    if (send) onSettingsChange()
    setExchangeBusy(true)
    try {
      const commandParameters = Object.fromEntries(editableFields.map((field) => {
        const value = commandInputs[field.key] ?? ''
        if (!value.trim()) throw new Error(`请填写或选择 ${field.display_name || field.key}`)
        if (/^(u?int)/i.test(field.data_type)) {
          const numeric = Number(value)
          if (!Number.isFinite(numeric)) throw new Error(`${field.display_name || field.key} 必须是有限数值`)
          return [field.key, numeric]
        }
        return [field.key, value]
      }))
      const payload = {
        profile_id: profile.asset_id,
        command_id: selectedCommand.command_id,
        array_id: arrayId,
        parameters: commandParameters,
        transport: sendTransport
      }
      const result = send
        ? await post<Record<string, any>>('/api/devices/beam_controller/send', payload)
        : await post<Record<string, any>>('/api/protocol/compile', payload)
      setExchangeResult(result)
      notify('success', send ? `${selectedCommand.display_name} ${sendTransport === 'RTC' ? '已由 RTC 确认发送完成' : '已写入'}；接收与解析在后台独立进行` : '指令编译预览完成，未发送到设备')
    } catch (error) {
      notify('error', errorMessage(error))
    } finally {
      setExchangeBusy(false)
    }
  }

  const updateTrInputs = (next: typeof trInputs) => {
    if (controlLocked || rtcBusy || JSON.stringify(next) === JSON.stringify(trInputs)) return
    setTrInputs(next)
    onSettingsChange()
  }

  const rtcCommand = async (action: string) => {
    const readOnly = action.startsWith('get_')
    if (!rtcAvailable || rtcBusy || (!readOnly && controlLocked)) return
    setRtcBusy(true)
    try {
      let parameters: Record<string, unknown> = {}
      if (action === 'configure_tr') {
        const { mode } = trInputs
        const period_us = Number(trInputs.period_us)
        const high_us = Number(trInputs.high_us)
        const delay_us = Number(trInputs.delay_us)
        if ([trInputs.period_us, trInputs.high_us, trInputs.delay_us].some((value) => !value.trim())
          || ![period_us, high_us, delay_us].every(Number.isFinite)
          || period_us <= 0 || high_us < 1 || high_us > 655 || delay_us < 0 || delay_us > 655
          || high_us / period_us > 0.3) {
          throw new Error('TR 周期必须大于 0；高宽 1～655 μs、占空比不超过 30%；延时 0～655 μs，且数值不能为空')
        }
        parameters = { mode, period_us, high_us, delay_us }
      }
      if (!readOnly) onSettingsChange()
      // Configuration, enabling, and disabling are separate operations. A successful
      // configuration never starts TR or sends another beam-controller command.
      const result = await post<Record<string, unknown>>('/api/devices/rtc/command', { action, parameters })
      setRtcReadback(result)
      if (action === 'get_tr_config' || action === 'configure_tr') {
        if ((result.mode === 'TX' || result.mode === 'RX') && ['period_us', 'high_us', 'delay_us'].every((key) => typeof result[key] === 'number')) {
          setTrInputs({ mode: result.mode === 'RX' ? 'RX' : 'TX', period_us: String(result.period_us), high_us: String(result.high_us), delay_us: String(result.delay_us) })
        }
      }
      const messages: Record<string, string> = {
        configure_tr: 'TR 参数已配置并回读；开启输出请单独操作', start_debug_tr: 'TR 调试输出已开启',
        stop_debug_tr: 'TR 调试输出已关闭', clear_fault: 'RTC 清故障操作完成'
      }
      notify('success', messages[action] ?? 'RTC 状态已回读')
    } catch (error) {
      notify('error', errorMessage(error))
    } finally { setRtcBusy(false) }
  }

  return <>
    <div className="page-lead">
      <span className="eyebrow">HARDWARE</span>
      <h2>真实设备与模拟器使用同一业务接口</h2>
      <p>每台设备独立连接。真实动作只由此页或冻结的自动计划触发。</p>
    </div>
    {controlLocked && <p className="inline-note warning">自动测试或设备收尾进行中，设备连接和调试写操作已锁定；仍可读取状态。</p>}
    <div className="device-cards">
      {devices.map((device) => {
        const connected = device.state !== 'DISCONNECTED'
        const serialDevice = device.device_id === 'beam_controller' || device.device_id === 'rtc'
        const realSerialWithoutPort = serialDevice && sources[device.device_id] === 'REAL' && !parameters[device.device_id].port
        return <Card
          key={device.device_id}
          title={deviceNames[device.device_id]}
          eyebrow={device.device_id.toUpperCase()}
          actions={<StatusPill value={device.state} source={device.source} />}
        >
          <div className="segmented small">
            <button disabled={connected || controlLocked} className={sources[device.device_id] === 'REAL' ? 'selected' : ''} onClick={() => setSources({ ...sources, [device.device_id]: 'REAL' })}>真实设备</button>
            <button disabled={connected || controlLocked} className={sources[device.device_id] === 'SIMULATED' ? 'selected' : ''} onClick={() => setSources({ ...sources, [device.device_id]: 'SIMULATED' })}>模拟器</button>
          </div>
          {sources[device.device_id] === 'REAL' && <div className="device-fields">
            {serialDevice && <>
              <Field label="COM 口">
                <div className="select-with-action">
                  <select
                    id={device.device_id === 'rtc' ? 'rtc-port-select' : 'beam-port-select'}
                    disabled={connected || controlLocked}
                    value={parameters[device.device_id].port}
                    onChange={(event) => setParameters({
                      ...parameters,
                      [device.device_id]: { ...parameters[device.device_id], port: event.target.value }
                    })}
                  >
                    <option value="">{serialPorts.length > 0 ? '请选择串口' : '未发现可用串口'}</option>
                    {serialPorts.map((port) => <option key={port.device} value={port.device}>
                      {port.device}{port.description ? ` · ${port.description}` : ''}
                    </option>)}
                  </select>
                  <button type="button" disabled={connected || serialLoading || controlLocked} onClick={() => refreshSerialPorts()}>{serialLoading ? '扫描中' : '刷新'}</button>
                </div>
              </Field>
              <Field label="波特率">
                <select
                  id={device.device_id === 'rtc' ? 'rtc-baud-select' : 'beam-baud-select'}
                  disabled={connected || controlLocked}
                  value={parameters[device.device_id].baud_rate}
                  onChange={(event) => setParameters({
                    ...parameters,
                    [device.device_id]: { ...parameters[device.device_id], baud_rate: Number(event.target.value) }
                  })}
                >
                  {[9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600].map((baudRate) => <option key={baudRate} value={baudRate}>{baudRate}</option>)}
                </select>
              </Field>
            </>}
            {device.device_id === 'vna' && <Field label="VISA 资源">
              <input disabled={connected || controlLocked} value={parameters.vna.resource} onChange={(event) => setParameters({ ...parameters, vna: { ...parameters.vna, resource: event.target.value } })} />
            </Field>}
            {device.device_id === 'turntable' && <>
              <p className="internal-driver" id="turntable-internal-driver"><b>转台驱动已内置</b><span>软件自动使用内部运行库，无需选择 DLL 文件</span></p>
              <Field label="控制器 IP"><input disabled={connected || controlLocked} value={parameters.turntable.controller_ip} onChange={(event) => setParameters({ ...parameters, turntable: { ...parameters.turntable, controller_ip: event.target.value } })} /></Field>
              <div className="locked-value"><span>设备号</span><b>0</b><span>比例系数</span><b>10000</b></div>
            </>}
          </div>}
          {connected
            ? <button className="button danger wide device-disconnect" disabled={controlLocked} onClick={() => disconnect(device.device_id)}>断开设备</button>
            : <button className="button primary wide" disabled={realSerialWithoutPort || controlLocked} onClick={() => connect(device.device_id)}>
                连接{sources[device.device_id] === 'REAL' ? '真实设备' : '模拟器'}
              </button>}
          <p className="identity">{device.identity ?? '尚无设备身份回读'}</p>
        </Card>
      })}
    </div>
    <Card title="波控协议调试" eyebrow="PROFILE COMMAND · TX / RX">
      {!profile || profileCommands.length === 0
        ? <p className="inline-note warning">请先在“测试执行”页加载天线配置包，配置包中的启用指令才会出现在这里。</p>
        : <>
          <div className="protocol-debug-grid">
            <Field label="发送通道"><select id="beam-transport-select" disabled={exchangeBusy || controlLocked} value={sendTransport} onChange={(event) => { setSendTransport(event.target.value as 'SERIAL' | 'RTC'); setExchangeResult(null) }}><option value="SERIAL">波控串口直连</option><option value="RTC">通过 RTC 发送</option></select></Field>
            <Field label="天线配置包"><input readOnly value={`${profile.profile_name} · ${profile.protocol_version}`} /></Field>
            <Field label="配置包指令">
              <select id="beam-command-select" value={commandId} onChange={(event) => setCommandId(event.target.value)}>
                {profileCommands.map((item) => <option key={item.command_id} value={item.command_id}>
                  {item.display_name} · {item.command_id} · 0x{item.opcode_hex}
                </option>)}
              </select>
            </Field>
            <div className="shared-array-id-reference"><span>阵面 ID</span><strong>{arrayId ?? '未设置'}</strong><small>由页面顶部统一设置</small></div>
          </div>
          {selectedCommand && <p className="command-description">
            <b>{selectedCommand.auto_role || 'MANUAL_ONLY'}</b>
            <span>{selectedCommand.description || '配置包未填写指令说明'} · 超时 {selectedCommand.timeout_ms} ms</span>
          </p>}
          {editableFields.length > 0 && <div className="protocol-field-grid">
            {editableFields.map((field) => <Field
              key={field.key}
              label={`${field.display_name || field.key}${field.unit ? ` (${field.unit})` : ''}`}
              hint={field.description || ([field.minimum, field.maximum].some((value) => value != null) ? `范围 ${field.minimum ?? '—'} ～ ${field.maximum ?? '—'}` : undefined)}
            >
              {field.enum_options.length > 0
                ? <select value={commandInputs[field.key] ?? ''} onChange={(event) => setCommandInputs({ ...commandInputs, [field.key]: event.target.value })}>
                    <option value="">请选择</option>
                    {field.enum_options.map((option) => <option key={`${field.key}-${option.logical}`} value={option.logical}>{option.display} · {option.logical}</option>)}
                  </select>
                : <input
                    type={/^(u?int)/i.test(field.data_type) ? 'number' : 'text'}
                    min={field.minimum ?? undefined}
                    max={field.maximum ?? undefined}
                    value={commandInputs[field.key] ?? ''}
                    onChange={(event) => setCommandInputs({ ...commandInputs, [field.key]: event.target.value })}
                  />}
            </Field>)}
          </div>}
          <div className="protocol-actions">
            <button className="button secondary" disabled={arrayId === null || exchangeBusy} onClick={() => executeProfileCommand(false)}>编译预览（不发送）</button>
            <button className="button primary" disabled={arrayId === null || !sendReady || exchangeBusy || controlLocked} onClick={() => executeProfileCommand(true)}>发送数据（不等待波控应答）</button>
            {!sendReady && <span>请先连接{sendTransport === 'RTC' ? 'RTC' : '波控机'}真实设备或模拟器</span>}
          </div>
          {exchangeResult && <div className="protocol-result">
            <div><span>波控帧</span><code>{exchangeResult.tx_hex ?? exchangeResult.frame_hex}</code></div>
            {exchangeResult.rtc_frame_hex && <div><span>RTC 包</span><code>{exchangeResult.rtc_frame_hex}</code></div>}
            <p>发送与接收相互独立；任何收到的数据都会自动进入下方日志，命中配置包时才显示关键解析信息。</p>
          </div>}
        </>}
    </Card>
    <Card title="RTC 触发调试" eyebrow="RTC V1.0 · TR">
      <fieldset className="plan-fields" disabled={controlLocked || rtcBusy}>
        <div className="rtc-tr-grid">
          <Field label="TR 收发模式"><select value={trInputs.mode} onChange={(e) => updateTrInputs({ ...trInputs, mode: e.target.value })}><option value="TX">发射（TX）</option><option value="RX">接收（RX）</option></select></Field>
          <Field label="TR 周期 (μs)"><input type="number" min="0.01" step="0.01" value={trInputs.period_us} onChange={(e) => updateTrInputs({ ...trInputs, period_us: e.target.value })} /></Field>
          <Field label="TR 高宽 (μs)" hint="1～655 μs，占空比不超过 30%"><input type="number" min="1" max="655" step="0.01" value={trInputs.high_us} onChange={(e) => updateTrInputs({ ...trInputs, high_us: e.target.value })} /></Field>
          <Field label="触发延时 (μs)" hint="TX 以上升沿、RX 以下降沿为参考"><input type="number" min="0" max="655" step="0.01" value={trInputs.delay_us} onChange={(e) => updateTrInputs({ ...trInputs, delay_us: e.target.value })} /></Field>
        </div>
        <div className="protocol-actions">
          <button className="button secondary" disabled={!rtcReady || rtcBusy || controlLocked} onClick={() => rtcCommand('configure_tr')}>配置 TR 参数</button>
          <button className="button primary" disabled={!rtcReady || rtcBusy || controlLocked} onClick={() => rtcCommand('start_debug_tr')}>开启 TR 调试</button>
          <button className="button danger" disabled={!rtcAvailable || rtcBusy || controlLocked} onClick={() => rtcCommand('stop_debug_tr')}>关闭 TR 调试</button>
          <button className="button secondary" disabled={!rtcAvailable || rtcBusy || controlLocked} onClick={() => rtcCommand('clear_fault')}>清除 RTC 故障</button>
        </div>
      </fieldset>
      <div className="protocol-actions rtc-read-actions">
        <button className="button ghost" disabled={!rtcAvailable || rtcBusy} onClick={() => rtcCommand('get_status')}>读取 RTC 状态</button>
        <button className="button ghost" disabled={!rtcAvailable || rtcBusy} onClick={() => rtcCommand('get_progress')}>读取采集计数</button>
        <button className="button ghost" disabled={!rtcAvailable || rtcBusy} onClick={() => rtcCommand('get_tr_config')}>读取 TR 参数</button>
        <button className="button ghost" disabled={!rtcAvailable || rtcBusy} onClick={() => rtcCommand('get_antenna_io_status')}>读取波控 I/O 状态</button>
      </div>
      <p className="inline-note">TR 参数配置、开启与关闭分别执行；触发脉宽固定 1 μs。调试时 RDY 低则跳过当前周期，不补发；此页不读取矢网测量数据。</p>
      {rtcReadback && <div className="rtc-readback" aria-live="polite">{Object.entries(rtcReadback).filter(([key, value]) => rtcReadbackLabels[key] && value != null).map(([key, value]) => <div key={key}><span>{rtcReadbackLabels[key]}</span><b>{typeof value === 'boolean' ? value ? '是' : '否' : typeof value === 'object' ? JSON.stringify(value) : String(value)}</b></div>)}</div>}
    </Card>
    <Card title="转台结构化调试" eyebrow="REAL PMAC · AXES 1/2/3/4/7">
      <div className="motion-panel">
        <Field label="轴"><select value={axis} onChange={(event) => setAxis(Number(event.target.value))}><option value="1">1 · 方位</option><option value="2">2 · 俯仰</option><option value="3">3 · 极化</option><option value="4">4 · 馈源</option><option value="7">7 · 平移</option></select></Field>
        <Field label={axis === 7 ? '目标 (mm)' : '目标 (°)'}><input type="text" inputMode="decimal" value={target} onChange={(event) => acceptsSignedFourDecimalInput(event.target.value) && setTarget(event.target.value)} /></Field>
        <Field label={axis === 7 ? '速度 (mm/s)' : '速度 (°/s)'}><input id="turntable-speed-input" type="number" min="0.0001" step="0.0001" value={speed} onChange={(event) => acceptsFourDecimalInput(event.target.value) && setSpeed(event.target.value)} onBlur={() => setSpeed(positiveFourDecimals(speed))} /></Field>
        <button className="button primary" disabled={controlLocked || !turntableReady || turntableMotionBusy || targetValue === null || !(speedValue > 0)} onClick={() => { if (targetValue !== null) void command('move_to', { axis, target: targetValue, speed: Number(speedValue.toFixed(4)) }) }}>{turntableMotionBusy ? '运动中，等待到位…' : '移动并等待到位'}</button>
        <button className="button secondary" disabled={controlLocked || !turntableReady || turntableMotionBusy} onClick={() => command('home', { axis })}>寻零</button>
        <button className="button secondary" id="turntable-read-position" disabled={!turntableRecoveryAvailable} onClick={() => command('read_axes')}>读取当前位置</button>
        <button className="button danger" disabled={controlLocked || !turntableRecoveryAvailable} onClick={() => command('stop', { axis: 'all' })}>软件停止全部轴</button>
      </div>
      <p className="inline-note warning">转台速度必须大于 0，最多保留四位小数。运动期间可随时读取五轴当前位置与速度，也可调用厂商 Stop；软件停止不是硬件急停。比例系数固定为 10000；连接后必须取得五轴位置与速度才会显示 READY。</p>
      {turntableReadback && <div className="turntable-readback" id="turntable-position-readback">
        {turntableAxes.map((item) => <div key={item.id}>
          <span>轴 {item.id} · {item.name}</span>
          <strong>{Number(turntableReadback.positions[String(item.id)] ?? 0).toFixed(4)} {item.unit}</strong>
          <small>速度 {Number(turntableReadback.velocities[String(item.id)] ?? 0).toFixed(4)} {item.unit}/s</small>
        </div>)}
      </div>}
    </Card>
    <Card title="设备通信日志" eyebrow="RAW DATA · LAST 2000" actions={<button className="button ghost" disabled={logs.length === 0} onClick={clearLogs}>清空日志</button>}>
      <p className="inline-note">发送完成与原始 RX 独立记录。RTC 的 E2 接收不作为波控发送成功应答；命中配置包时附加关键解析字段。</p>
      <div className="log-filters">
        <Field label="日志设备"><select value={logDevice} onChange={(e) => setLogDevice(e.target.value)}><option value="ALL">全部设备</option>{Object.entries(deviceNames).map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></Field>
        <Field label="日志方向"><select value={logDirection} onChange={(e) => setLogDirection(e.target.value)}><option value="ALL">全部方向</option><option value="TX">发送 TX</option><option value="RX">独立接收 RX</option><option value="ERROR">异常</option></select></Field>
      </div>
      <div className="device-log-list" id="device-raw-log">
        {visibleLogs.length === 0
          ? <div className="empty-log">当前筛选下没有通信记录</div>
          : visibleLogs.map((item, index) => <div className={`device-log-row ${item.direction.toLowerCase()}`} key={`${item.timestamp}-${index}`}>
              <time>{new Date(item.timestamp).toLocaleTimeString('zh-CN', { hour12: false })}</time>
              <b>{item.direction}</b>
              <span>{deviceNames[item.device_id] ?? item.device_id} · {item.command_id ?? '—'}{item.transport ? ` · ${item.transport}` : ''}{item.context ? ` · ${item.context}` : ''}</span>
              <code>{item.raw_hex || item.error || '—'}</code>
              {item.parsed && <div className="parsed-fields">
                {item.parsed.fields.length > 0
                  ? item.parsed.fields.map((field) => <span key={field.key}>
                      <b>{field.label}</b>{String(field.value ?? '—')}{field.unit ? ` ${field.unit}` : ''}
                    </span>)
                  : <span><b>已匹配</b>{item.parsed.profile_name}</span>}
              </div>}
            </div>)}
      </div>
    </Card>
  </>
}

export default App
