import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import vm from 'node:vm'
import test from 'node:test'
const { transformSync } = createRequire(import.meta.resolve('electron-vite'))('esbuild')

// Exercise the actual component handlers and effects without opening Electron,
// starting a service, accessing files through the app, or connecting any device.
const root = fileURLToPath(new URL('..', import.meta.url))
const equalDeps = (a, b) => a && b && a.length === b.length && a.every((value, i) => Object.is(value, b[i]))
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done }); return { promise, resolve } }
const children = (node) => node && typeof node === 'object' ? [node, ...[node.props?.children].flat(Infinity).flatMap(children)] : []
const label = (node) => node == null || typeof node === 'boolean' ? '' : typeof node === 'string' || typeof node === 'number' ? String(node) : [node?.props?.children].flat(Infinity).map(label).join('')
const find = (tree, predicate) => children(tree).find(predicate)
const button = (tree, text) => find(tree, (node) => node.type === 'button' && label(node).includes(text))

function loadRenderer(handlers = {}) {
  let active
  const timers = []
  const sockets = []
  const calls = []
  class Socket {
    static OPEN = 1
    readyState = 1
    constructor() { sockets.push(this) }
    close() { this.readyState = 3; this.onclose?.() }
    emit(message) { this.onmessage?.({ data: JSON.stringify(message) }) }
  }
  const react = {
    useState(value) {
      const harness = active; const index = harness.index++
      if (!(index in harness.slots)) harness.slots[index] = typeof value === 'function' ? value() : value
      return [harness.slots[index], (next) => {
        const result = typeof next === 'function' ? next(harness.slots[index]) : next
        if (!Object.is(result, harness.slots[index])) { harness.slots[index] = result; harness.dirty = true }
      }]
    },
    useRef(value) { const index = active.index++; return active.slots[index] ??= { current: value } },
    useMemo(callback, deps) {
      const index = active.index++; const previous = active.slots[index]
      if (!previous || !equalDeps(previous.deps, deps)) active.slots[index] = { deps, value: callback() }
      return active.slots[index].value
    },
    useCallback(callback, deps) { return react.useMemo(() => callback, deps) },
    useEffect(callback, deps) {
      const harness = active; const index = harness.index++; const previous = harness.slots[index]
      if (!previous || !equalDeps(previous.deps, deps)) {
        harness.slots[index] = { deps, cleanup: previous?.cleanup }
        harness.effects.push(() => {
          harness.slots[index].cleanup?.()
          harness.slots[index].cleanup = callback()
        })
      }
    }
  }
  const jsx = (type, props) => ({ type, props: props ?? {} })
  const api = async (path, body) => {
    calls.push({ path, body })
    if (handlers.api) return handlers.api(path, body)
    if (path === '/api/health') return { status: 'healthy' }
    if (path === '/api/assets') return { profiles: [], coordinates: [] }
    if (path === '/api/devices' || path === '/api/devices/serial-ports') return []
    throw new Error(`Unexpected API: ${path}`)
  }
  const post = async (path, body) => {
    calls.push({ path, body })
    if (handlers.post) return handlers.post(path, body)
    return {}
  }
  const context = {
    exports: {}, module: { exports: {} }, console, WebSocket: Socket,
    window: {
      setTimeout: (callback, delay) => { timers.push({ callback, delay }); return timers.length }, clearTimeout() {},
      antennaDesktop: { openFile: async () => '/chosen.hdf5', saveFile: async () => '/output.hdf5', appPaths: async () => ({ turntableDll: '/unused.dll' }), ...handlers.bridge }
    },
    require: (name) => name === 'react' ? react : name === 'react/jsx-runtime' ? { jsx, jsxs: jsx, Fragment: 'fragment' } : {
      api, post, WS_URL: 'ws://unused', ApiError: class extends Error {}
    }
  }
  const source = readFileSync(resolve(root, 'src/App.tsx'), 'utf8') + '\nexport { App, RunPage, DevicesPage, CompensationPage, HistoryPage, finiteTarget };'
  vm.runInNewContext(transformSync(source, { loader: 'tsx', format: 'cjs', jsx: 'automatic', target: 'es2022' }).code, context)
  context.exports = context.module.exports
  function mount(name, props = {}) {
    const harness = { props, slots: [], index: 0, effects: [], dirty: true, tree: null }
    harness.flush = () => {
      for (let count = 0; harness.dirty || harness.effects.length; count++) {
        assert.ok(count < 30, 'component effects must settle')
        if (harness.dirty) {
          active = harness; harness.index = 0; harness.dirty = false
          harness.tree = context.exports[name](harness.props)
        }
        const effects = harness.effects.splice(0); effects.forEach((effect) => effect())
      }
      return harness.tree
    }
    harness.settle = async () => { for (let i = 0; i < 8; i++) { await Promise.resolve(); harness.flush() } }
    harness.flush()
    return harness
  }
  return { mount, calls, sockets, timers, exported: context.exports }
}

const coordinates = { asset_id: 'coord-A', antenna_id: 'A', polarizations: ['H', 'V'] }
const pageProps = () => ({ assets: { profiles: [], coordinates: [coordinates] }, devices: [{ device_id: 'beam_controller', state: 'READY' }], arrayId: 0, onArrayIdBusyChange() {}, notify() {} })
const sample = (completed) => ({ run_id: 'run-1', completed, kind: 'CALIBRATION', channel: { element: completed - 1 }, magnitude_db: -completed })
const run = (state, completed = 0, cleanup_pending = false) => ({ run_id: 'run-1', state, completed, total: 2, cleanup_pending, plan: { array_id: 7 } })

test('blank/incomplete motion targets cannot submit; zero and negative targets remain valid', async () => {
  const renderer = loadRenderer()
  const h = renderer.mount('DevicesPage', { profiles: [], devices: [{ device_id: 'turntable', state: 'READY' }], setDevices() {}, logs: [], clearLogs() {}, arrayId: 0, onArrayIdBusyChange() {}, notify() {} })
  for (const value of ['', '-', '.', '-.']) {
    const field = find(h.tree, (node) => node.props.label === '目标 (°)')
    field.props.children.props.onChange({ target: { value } }); h.flush()
    const move = button(h.tree, '移动并等待到位')
    assert.equal(move.props.disabled, true, value)
    move.props.onClick(); await h.settle()
  }
  assert.equal(renderer.calls.filter((call) => call.path.endsWith('/command')).length, 0)
  for (const value of ['0', '-1.2500']) {
    find(h.tree, (node) => node.props.label === '目标 (°)').props.children.props.onChange({ target: { value } }); h.flush()
    assert.equal(button(h.tree, '移动并等待到位').props.disabled, false)
    button(h.tree, '移动并等待到位').props.onClick(); await h.settle()
    assert.equal(renderer.calls.at(-1).body.parameters.target, Number(value))
  }
})

test('H/V selection generates one compensation request containing both calibration files', async () => {
  const paths = ['/H.hdf5', '/V.hdf5']
  const renderer = loadRenderer({
    bridge: { openFile: async () => paths.shift() },
    post: async (path, body) => path.endsWith('/inspect')
      ? { analysis: { kind: 'CALIBRATION', frequencies_hz: [8e9] }, metadata: { signal_path: 'TX', polarization: body.path.includes('/H') ? 'H' : 'V' } }
      : { path: '/TX.hdf5', frequencies_hz: [8e9], aperture_weighting: { algorithm: 'NONE' } }
  })
  const h = renderer.mount('CompensationPage', pageProps())
  for (const polarization of ['H', 'V']) {
    const row = find(h.tree, (node) => node.props.className === 'asset-row' && label(node).includes(`TX / ${polarization}`))
    await button(row, '选择').props.onClick(); await h.settle()
  }
  const generate = button(h.tree, '按所选频点生成 TX')
  assert.equal(generate.props.disabled, false)
  await generate.props.onClick(); await h.settle()
  assert.equal(JSON.stringify(renderer.calls.find((call) => call.path.endsWith('/generate')).body.calibration_files), JSON.stringify(['/H.hdf5', '/V.hdf5']))
})

test('faulted turntable retains read/stop recovery actions without enabling new motion', () => {
  const renderer = loadRenderer()
  const h = renderer.mount('DevicesPage', { profiles: [], devices: [{ device_id: 'turntable', state: 'UNKNOWN' }], setDevices() {}, logs: [], clearLogs() {}, arrayId: 0, onArrayIdBusyChange() {}, notify() {} })
  assert.equal(button(h.tree, '读取当前位置').props.disabled, false)
  assert.equal(button(h.tree, '软件停止全部轴').props.disabled, false)
  assert.equal(button(h.tree, '移动并等待到位').props.disabled, true)
  assert.equal(button(h.tree, '寻零').props.disabled, true)
})

test('reloading a profile refreshes same-command defaults and empty numbers cannot compile', async () => {
  const makeField = (key, value) => ({ key, display_name: key, data_type: 'uint8', unit: null, source: 'USER', default: value, enum_options: [], description: '' })
  const makeProfile = (id, fields) => ({ asset_id: id, profile_name: id, commands: [{ command_id: 'SAME', display_name: 'Same', auto_role: 'MANUAL_ONLY', opcode_hex: '70', risk: 'SAFE', timeout_ms: 1000, fields }] })
  const renderer = loadRenderer({ post: async () => ({ frame_hex: 'AA55' }) })
  const h = renderer.mount('DevicesPage', { profiles: [makeProfile('old', [makeField('value', 7)])], devices: [], setDevices() {}, logs: [], clearLogs() {}, arrayId: 0, onArrayIdBusyChange() {}, notify() {} })
  const input = (name) => find(h.tree, (node) => node.props.label === name).props.children
  assert.equal(input('value').props.value, '7')
  await button(h.tree, '编译预览').props.onClick(); await h.settle()
  h.props = { ...h.props, profiles: [makeProfile('new', [makeField('value', 8), makeField('added', 77)])] }
  h.dirty = true; h.flush()
  assert.equal(input('value').props.value, '8')
  assert.equal(input('added').props.value, '77')
  assert.ok(!find(h.tree, (node) => node.props.className === 'protocol-result'))
  input('added').props.onChange({ target: { value: '' } }); h.flush()
  const previous = renderer.calls.filter((call) => call.path.endsWith('/compile')).length
  await button(h.tree, '编译预览').props.onClick(); await h.settle()
  assert.equal(renderer.calls.filter((call) => call.path.endsWith('/compile')).length, previous)
})

test('FLASH input changes invalidate prepared items and reject an in-flight stale preparation', async () => {
  const waiting = deferred()
  let prepareCount = 0
  const result = { path: '/flash.hdf5', items: [{ item_name: 'ARRAY_ID', status: 'READY', end_address: 255 }] }
  const renderer = loadRenderer({ post: async (path) => path.endsWith('/prepare') ? (++prepareCount === 1 ? result : waiting.promise) : {} })
  const h = renderer.mount('CompensationPage', pageProps())
  for (const name of ['TX 补偿', 'RX 补偿']) {
    const row = find(h.tree, (node) => node.props.className === 'file-line' && label(node).includes(name))
    await button(row, '选择').props.onClick(); await h.settle()
  }
  await button(h.tree, '准备五类').props.onClick(); await h.settle()
  assert.equal(button(h.tree, '写入并读回').props.disabled, false)
  h.props = { ...h.props, assets: { profiles: [], coordinates: [{ ...coordinates, asset_id: 'coord-B' }] } }; h.dirty = true; h.flush()
  assert.equal(button(h.tree, '写入并读回').props.disabled, true)
  const preparing = button(h.tree, '准备五类').props.onClick(); await h.settle()
  assert.equal(h.tree.props.disabled, true, 'all FLASH inputs locked during preparation')
  h.props = { ...h.props, arrayId: 1 }; h.dirty = true; h.flush()
  waiting.resolve(result); await preparing; await h.settle()
  assert.equal(button(h.tree, '写入并读回').props.disabled, true, 'stale response must not restore a writable item')
})

test('reselecting the same historical path requests and renders the slice again', async () => {
  const renderer = loadRenderer({ post: async (path) => path.endsWith('/inspect')
    ? { path: '/same.hdf5', analysis: { kind: 'CALIBRATION', frequencies_hz: [8e9] } }
    : { kind: 'CALIBRATION', frequency_hz: 8e9, channels: [], max_amplitude_difference_db: 0 } })
  const h = renderer.mount('HistoryPage', { notify() {} })
  for (let i = 0; i < 2; i++) {
    await button(h.tree, '选择 HDF5').props.onClick(); await h.settle()
    assert.ok(find(h.tree, (node) => node.props.title === '各通道初始幅度与相位'))
  }
  assert.equal(renderer.calls.filter((call) => call.path.endsWith('/view')).length, 2)
})

test('reconnect restores missed completion/samples and discards buffered events older than the snapshot', async () => {
  const restore = deferred()
  let snapshots = 0
  const renderer = loadRenderer({ api: async (path) => {
    if (path === '/api/runs/current') return ++snapshots === 1
      ? { run: run('RUNNING', 1), samples: [sample(1)], event_sequence: 5 }
      : restore.promise
    if (path === '/api/assets') return { profiles: [], coordinates: [] }
    if (path === '/api/health') return { status: 'healthy' }
    return []
  } })
  const h = renderer.mount('App'); await renderer.sockets[0].onopen(); await h.settle()
  renderer.sockets[0].close()
  renderer.timers.find((timer) => timer.delay === 1200).callback()
  const reconnecting = renderer.sockets[1].onopen()
  renderer.sockets[1].emit({ type: 'run.status', sequence: 6, run: run('RUNNING', 1) })
  restore.resolve({ run: run('COMPLETED', 2), samples: [sample(1), sample(2)], event_sequence: 9 })
  await reconnecting; await h.settle()
  const page = find(h.tree, (node) => node.type.name === 'RunPage')
  assert.equal(page.props.run.state, 'COMPLETED')
  assert.equal(page.props.samples.length, 2)
  assert.equal(find(h.tree, (node) => node.props.id === 'global-array-id').props.disabled, false)
  renderer.sockets[1].emit({ type: 'run.status', sequence: 10, run: run('COMPLETED', 2, true) }); h.flush()
  assert.equal(find(h.tree, (node) => node.props.id === 'global-array-id').props.disabled, true)
})

function loadMain({ owner = true, active = true, neverReady = false, choices = [0] } = {}) {
  const calls = []; const listeners = {}; let now = 0; let killed = 0; let quit = 0; let polls = 0
  const app = { isPackaged: false, whenReady: () => ({ then() {} }), on: (event, callback) => { listeners[event] = callback }, quit: () => { quit++; listeners['before-quit']?.({ preventDefault() {} }) } }
  const context = {
    exports: {}, module: { exports: {} }, process: { env: {}, platform: 'win32' }, console, AbortSignal,
    Date: { now: () => now }, setTimeout: (callback, delay) => { now += delay; callback() },
    fetch: async (url, options) => {
      const path = new URL(url).pathname; calls.push({ path, method: options.method })
      const body = path.endsWith('/current') ? { control_owner: active ? 'run-1' : null, run: active ? run('RUNNING') : null }
        : { ready: path.endsWith('/prepare') ? !neverReady && ++polls >= 2 : true }
      return { ok: true, json: async () => body }
    },
    require: (name) => name === 'electron'
      ? { app, dialog: { showMessageBox: async () => ({ response: choices.shift() ?? 0 }) }, ipcMain: {}, Menu: {} }
      : name === 'node:child_process' ? { spawn: (command, args) => {
        assert.equal(command, 'taskkill')
        assert.equal(JSON.stringify(args), JSON.stringify(['/PID', '1234', '/T', '/F']))
        assert.ok(polls >= 2, 'must finish shutdown before terminating a PID tree')
        killed++
        const child = { once: (event, callback) => { if (event === 'exit') Promise.resolve().then(() => callback(0)); return child } }
        return child
      } } : name === 'node:path' ? { dirname() {}, join() {}, resolve() {} } : {}
  }
  const source = readFileSync(resolve(root, 'electron/main.ts'), 'utf8') + '\nexport { requestSafeClose }; export function configure(window, child) { mainWindow = window; serviceProcess = child; }'
  vm.runInNewContext(transformSync(source, { loader: 'ts', format: 'cjs', target: 'es2022' }).code, context)
  context.exports = context.module.exports
  context.exports.configure({ isDestroyed: () => false }, owner ? { pid: 1234, exitCode: null, kill: () => { killed++ } } : null)
  return { close: context.exports.requestSafeClose, calls, counts: () => ({ killed, quit }) }
}

test('close waits for ready before terminating only its own service', async () => {
  const main = loadMain(); await main.close()
  assert.equal(main.calls.filter((call) => call.path.endsWith('/prepare')).length, 2)
  assert.deepEqual(main.counts(), { killed: 1, quit: 1 })
})

test('reused service is unfrozen instead of killed; cancelled waiting preserves the window', async () => {
  const reused = loadMain({ owner: false }); await reused.close()
  assert.equal(reused.calls.at(-1).path, '/api/shutdown/cancel')
  assert.deepEqual(reused.counts(), { killed: 0, quit: 1 })
  const cancelled = loadMain({ neverReady: true, choices: [0, 1] }); await cancelled.close()
  assert.equal(cancelled.calls.at(-1).path, '/api/shutdown/cancel')
  assert.deepEqual(cancelled.counts(), { killed: 0, quit: 0 })
})


const rtcProfile = { asset_id: 'rtc-profile', profile_name: 'RTC profile', protocol_version: 'V1.0', vectors: [], capabilities: { beam_signal_path: true }, commands: [{ command_id: 'BEAM_SET', display_name: '设置波束', auto_role: 'BEAM_SET', opcode_hex: '70', timeout_ms: 1000, fields: [] }] }
const deviceProps = (extra = {}) => ({ profiles: [rtcProfile], devices: [{ device_id: 'rtc', state: 'READY' }], setDevices() {}, logs: [], clearLogs() {}, arrayId: 0, onArrayIdBusyChange() {}, notify() {}, ...extra })
const changeField = (h, name, value) => { find(h.tree, (node) => node.props.label === name).props.children.props.onChange({ target: { value } }); h.flush() }

test('RTC connects using an explicitly selected COM port and baud rate', async () => {
  const renderer = loadRenderer({ api: async () => [{ device: 'COM8', description: 'RTC' }], post: async () => ({ device_id: 'rtc', state: 'READY', identity: 'RTC V1.0' }) })
  const h = renderer.mount('DevicesPage', deviceProps({ devices: [{ device_id: 'rtc', state: 'DISCONNECTED' }] })); await h.settle()
  assert.equal(button(h.tree, '连接真实设备').props.disabled, true)
  assert.ok(!find(h.tree, (node) => node.props.label === 'UDP 端口'))
  find(h.tree, (node) => node.props.id === 'rtc-port-select').props.onChange({ target: { value: 'COM8' } }); h.flush()
  find(h.tree, (node) => node.props.id === 'rtc-baud-select').props.onChange({ target: { value: '230400' } }); h.flush()
  await button(h.tree, '连接真实设备').props.onClick(); await h.settle()
  assert.deepEqual(JSON.parse(JSON.stringify(renderer.calls.at(-1).body.parameters)), { port: 'COM8', baud_rate: 230400 })
})

test('RTC forwarding, TR configuration and TR output are independent, and invalid timing never sends', async () => {
  let invalidations = 0
  const renderer = loadRenderer({ post: async (path, body) => path.endsWith('/send') ? { tx_hex: 'A5' } : body.action === 'configure_tr' ? { ...body.parameters, tr_state: 0 } : { state: 'IDLE' } })
  const h = renderer.mount('DevicesPage', deviceProps({ onSettingsChange() { invalidations++ } }))
  assert.equal(button(h.tree, '发送数据').props.disabled, true, 'direct serial route requires beam controller')
  changeField(h, '发送通道', 'RTC')
  assert.equal(button(h.tree, '发送数据').props.disabled, false, 'RTC route only requires RTC')
  await button(h.tree, '发送数据').props.onClick(); await h.settle()
  assert.equal(renderer.calls.filter((call) => call.path.endsWith('/send')).length, 1)
  assert.equal(renderer.calls.find((call) => call.path.endsWith('/send')).body.transport, 'RTC')
  assert.equal(renderer.calls.filter((call) => call.path.endsWith('/rtc/command')).length, 0)
  changeField(h, 'TR 收发模式', 'RX')
  await button(h.tree, '配置 TR 参数').props.onClick(); await h.settle()
  assert.equal(renderer.calls.at(-1).body.action, 'configure_tr')
  assert.deepEqual(JSON.parse(JSON.stringify(renderer.calls.at(-1).body.parameters)), { mode: 'RX', period_us: 100, high_us: 20, delay_us: 1 })
  await button(h.tree, '开启 TR 调试').props.onClick(); await h.settle()
  await button(h.tree, '关闭 TR 调试').props.onClick(); await h.settle()
  assert.deepEqual(renderer.calls.filter((call) => call.path.endsWith('/rtc/command')).map((call) => call.body.action), ['configure_tr', 'start_debug_tr', 'stop_debug_tr'])
  assert.equal(renderer.calls.filter((call) => call.path.endsWith('/send')).length, 1, 'TR operations must never resend beam data')
  for (const [name, value] of [['TR 高宽 (μs)', '31'], ['TR 周期 (μs)', '']]) {
    changeField(h, name, value)
    const before = renderer.calls.length
    await button(h.tree, '配置 TR 参数').props.onClick(); await h.settle()
    assert.equal(renderer.calls.length, before)
  }
  assert.ok(invalidations > 0, 'TR changes invalidate prepared plans')
  h.props = { ...h.props, controlLocked: true }; h.dirty = true; h.flush()
  for (const name of ['配置 TR 参数', '开启 TR 调试', '关闭 TR 调试', '清除 RTC 故障', '发送数据']) {
    const action = button(h.tree, name)
    assert.equal(action.props.disabled, true)
    const before = renderer.calls.length
    await action.props.onClick(); await h.settle()
    assert.equal(renderer.calls.length, before, 'handler also enforces the run lock')
  }
  assert.equal(button(h.tree, '读取 RTC 状态').props.disabled, false)
  await button(h.tree, '读取 RTC 状态').props.onClick(); await h.settle()
  assert.equal(renderer.calls.at(-1).body.action, 'get_status')
})

test('RTC direction-pattern modes reuse scan inputs, invalidate preparation and offer only supported RTC calibration', async () => {
  let currentRun = null
  const renderer = loadRenderer({ bridge: { openDirectory: async () => '/results' }, post: async (path, body) => path.endsWith('/prepare') ? { run_id: 'rtc-run', state: 'PREPARED', total: 1, plan: body } : waveTable(body) })
  const h = renderer.mount('RunPage', { assets: { profiles: [rtcProfile], coordinates: [{ ...coordinates, channel_count: 1, enabled_count: 1, evidence: 'SIMULATED' }] }, setAssets() {}, devices: ['vna', 'rtc', 'turntable'].map((device_id) => ({ device_id, state: 'READY' })), run: null, setRun(value) { currentRun = value }, samples: [], setSamples() {}, arrayId: 0, onArrayIdBusyChange() {}, notify() {} })
  assert.ok(find(h.tree, (node) => node.props.label === '采集模式'))
  assert.ok(!find(h.tree, (node) => node.type === 'option' && node.props.value === 'RTC_CONTINUOUS'))
  button(h.tree, '方向图扫描').props.onClick(); h.flush()
  changeField(h, '采集模式', 'RTC_CONTINUOUS')
  await button(h.tree, '浏览').props.onClick(); await h.settle()
  assert.equal(button(h.tree, '准备测试').props.disabled, false, 'RTC run does not require a direct beam-controller connection')
  await button(h.tree, '准备测试').props.onClick(); await h.settle()
  const request = renderer.calls.find((call) => call.path.endsWith('/prepare')).body
  assert.equal(request.topology, 'RTC_CONTINUOUS')
  assert.equal(request.beam_control_mode, 'SOFTWARE_DIRECT')
  assert.equal(request.azimuth_step_deg, 1)
  h.props = { ...h.props, run: currentRun }; h.dirty = true; h.flush()
  changeField(h, '采集模式', 'RTC_STOP_AND_GO')
  assert.equal(currentRun, null, 'topology change invalidates prepared run')
  h.props = { ...h.props, run: null }; h.dirty = true; h.flush()
  button(h.tree, '逐通道定点标校').props.onClick(); h.flush()
  button(h.tree, '方向图扫描').props.onClick(); h.flush()
  assert.equal(find(h.tree, (node) => node.props.label === '采集模式').props.children.props.value, 'RTC_STOP_AND_GO')
})

test('RTC unsolicited receive logs can be inspected independently of transmit completion', () => {
  const renderer = loadRenderer()
  const h = renderer.mount('DevicesPage', deviceProps({ logs: [
    { timestamp: '2026-09-15T00:00:00Z', device_id: 'rtc', direction: 'TX', command_id: '30', raw_hex: 'TXFRAME' },
    { timestamp: '2026-09-15T00:00:01Z', device_id: 'rtc', direction: 'RX', command_id: 'E2', raw_hex: 'RXFRAME', transport: 'RTC' },
    { timestamp: '2026-09-15T00:00:02Z', device_id: 'beam_controller', direction: 'RX', raw_hex: 'DIRECT' }
  ] }))
  changeField(h, '日志设备', 'rtc')
  changeField(h, '日志方向', 'RX')
  const rows = children(h.tree).filter((node) => node.props.className === 'device-log-row rx')
  assert.equal(rows.length, 1)
  assert.ok(label(rows[0]).includes('RTC · E2'))
  assert.ok(label(rows[0]).includes('RXFRAME'))
  assert.equal(renderer.calls.filter((call) => call.path.endsWith('/send')).length, 0)
})


test('RTC pending pause remains running, labels its completion boundary and keeps stop available', async () => {
  for (const [topology, waitingLabel] of [['RTC_CONTINUOUS', '等待本行完成后暂停'], ['RTC_STOP_AND_GO', '等待本组完成后暂停']]) {
    const current = { run_id: 'rtc-pause', state: 'RUNNING', pause_requested: false, cleanup_pending: false, completed: 0, total: 2, progress: 0, plan: {
      test_type: 'PATTERN', topology, beam_control_mode: 'SOFTWARE_DIRECT', signal_path: 'TX', polarization: 'H', s_parameter: 'S21',
      base_filename: 'test', frequency_start_hz: 8e9, frequency_stop_hz: 8e9, frequency_points: 1, if_bandwidth_hz: 1000,
      source_power_dbm: -10, averaging_enabled: false, averaging_count: 1, settle_ms: 10, azimuth_start_deg: 0, azimuth_stop_deg: 1,
      azimuth_step_deg: 1, elevation_start_deg: 0, elevation_stop_deg: 0, elevation_step_deg: 1, move_speed_deg_s: 1,
      beams: [{ beam_id: 'BEAM-1', off_axis_deg: 0, azimuth_deg: 0 }], output_directory: '/results'
    } }
    let updated
    const renderer = loadRenderer({ post: async () => ({ ...current, pause_requested: true }) })
    const h = renderer.mount('RunPage', { assets: { profiles: [], coordinates: [] }, setAssets() {}, devices: [], run: current,
      setRun(value) { updated = value }, samples: [], setSamples() {}, arrayId: 0, onArrayIdBusyChange() {}, notify() {} })
    await button(h.tree, '暂停').props.onClick(); await h.settle()
    assert.equal(updated.state, 'RUNNING')
    h.props = { ...h.props, run: updated }; h.dirty = true; h.flush()
    assert.equal(button(h.tree, waitingLabel).props.disabled, true)
    assert.equal(button(h.tree, '正常停止').props.disabled, false)
    assert.ok(!button(h.tree, '继续'))
    const before = renderer.calls.length
    await button(h.tree, waitingLabel).props.onClick(); await h.settle()
    assert.equal(renderer.calls.length, before, 'pending pause cannot send duplicate requests')
    h.props = { ...h.props, run: { ...updated, state: 'PAUSED', pause_requested: false } }; h.dirty = true; h.flush()
    assert.equal(button(h.tree, '继续').props.disabled, false)
    assert.ok(!button(h.tree, waitingLabel))
  }
})


const rtcRunProps = (extra = {}) => ({
  assets: { profiles: [rtcProfile], coordinates: [{ ...coordinates, channel_count: 2, enabled_count: 2, evidence: 'SIMULATED' }] },
  setAssets() {}, devices: [{ device_id: 'rtc', state: 'READY' }], run: null, setRun() {},
  samples: [], setSamples() {}, arrayId: 0, onArrayIdBusyChange() {}, notify() {}, ...extra
})
const waveTable = (body, status = 'PREVIEW') => ({
  entries: (body.test_type === 'CALIBRATION' ? [0, 1] : body.beams).map((beam, index) => ({
    address: index + 1, label: body.test_type === 'CALIBRATION' ? `通道 ${index}` : beam.beam_id,
    frame_hex: `${body.array_id}:${body.signal_path}:${body.polarization}:${body.reference_frequency_hz}:${index}`, status
  })),
  count: body.test_type === 'CALIBRATION' ? 2 : body.beams.length, capacity: 512,
  verified: ['VERIFIED', 'MATCH'].includes(status), source: status === 'PREVIEW' ? undefined : 'SIMULATED'
})

test('RTC wave preload uses existing beam inputs with only RTC connected, without a directory or a measurement', async () => {
  const renderer = loadRenderer({ post: async (path, body) => waveTable(body, path.endsWith('/write') ? 'VERIFIED' : path.endsWith('/read') ? 'EMPTY' : 'PREVIEW') })
  const h = renderer.mount('RunPage', rtcRunProps({ assets: { profiles: [rtcProfile], coordinates: [] } }))
  button(h.tree, '方向图扫描').props.onClick(); h.flush()
  changeField(h, '采集模式', 'RTC_STOP_AND_GO')
  changeField(h, '电子波束方向：离轴角, 方位角 (°)', '0,0; 10,30')
  changeField(h, '终止频率 (GHz)', '10')
  await h.settle()
  assert.equal(button(h.tree, '准备测试').props.disabled, true, 'measurement still needs VNA, turntable and a directory')
  assert.equal(button(h.tree, '写入 RTC 并校验').props.disabled, false)
  assert.equal(button(h.tree, '读取 RTC 波位').props.disabled, false)
  assert.equal(renderer.calls.filter((call) => call.path.endsWith('/write') || call.path.endsWith('/read')).length, 0, 'preview does not touch RTC')
  await button(h.tree, '写入 RTC 并校验').props.onClick(); await h.settle()
  const request = renderer.calls.find((call) => call.path.endsWith('/write')).body
  assert.equal(request.reference_frequency_hz, 9e9)
  assert.equal(request.beams.length, 2)
  assert.equal(request.beams[1].off_axis_deg, 10)
  assert.equal(request.beams[1].azimuth_deg, 30)
  assert.equal(request.output_directory, undefined)
  assert.equal(request.coordinate_id, undefined, 'beam preload does not require a coordinate table')
  assert.equal(request.address, undefined)
  assert.equal(request.count, undefined)
  assert.ok(label(h.tree).includes('已回读确认与当前计划一致'))
  assert.ok(label(h.tree).includes('SIMULATED'))
  assert.equal(renderer.calls.filter((call) => call.path.includes('/api/runs/')).length, 0)
  changeField(h, '电子波束方向：离轴角, 方位角 (°)', '0,0')
  assert.ok(!label(h.tree).includes('校验一致'), 'changing a wave clears old verification immediately')
  await h.settle()
  await button(h.tree, '读取 RTC 波位').props.onClick(); await h.settle()
  assert.ok(label(h.tree).includes('尚未写入'))
  assert.ok(!label(h.tree).includes('已回读确认与当前计划一致'))
  changeField(h, '波控方式', 'EXTERNAL_FIXED')
  assert.ok(!button(h.tree, '写入 RTC 并校验'), 'fixed external beam never presents a write-table action')
})

test('RTC stale wave preview and an in-flight old write cannot restore verification after input changes', async () => {
  const firstPreview = deferred()
  const pendingWrite = deferred()
  let previewCount = 0
  const renderer = loadRenderer({ post: async (path, body) => {
    if (path.endsWith('/write')) return pendingWrite.promise
    if (path.endsWith('/preview') && ++previewCount === 1) return firstPreview.promise
    return waveTable(body)
  } })
  const h = renderer.mount('RunPage', rtcRunProps())
  changeField(h, '采集模式', 'RTC_STOP_AND_GO')
  changeField(h, '收发模式', 'RX'); await h.settle()
  const first = renderer.calls.find((call) => call.path.endsWith('/preview')).body
  firstPreview.resolve(waveTable(first)); await h.settle()
  assert.ok(!label(h.tree).includes('0:TX:H:'), 'old preview cannot overwrite current RX bytes')
  const writing = button(h.tree, '写入 RTC 并校验').props.onClick(); h.flush()
  assert.equal(button(h.tree, '写入 RTC 并校验').props.disabled, true)
  const request = renderer.calls.find((call) => call.path.endsWith('/write')).body
  h.props = { ...h.props, arrayId: 7 }; h.dirty = true; h.flush(); await h.settle()
  pendingWrite.resolve(waveTable(request, 'VERIFIED')); await writing; await h.settle()
  assert.ok(!label(h.tree).includes('已回读确认与当前计划一致'))
  assert.ok(label(h.tree).includes('7:RX:H:'))
  assert.ok(!label(h.tree).includes('0:RX:H:'))
})

test('RTC calibration prepares with VNA and RTC only and clears preparation on wave edits or manual writes', async () => {
  let prepared
  const renderer = loadRenderer({ bridge: { openDirectory: async () => '/results' }, post: async (path, body) => path.endsWith('/prepare')
    ? { run_id: 'rtc-cal', state: 'PREPARED', total: 2, plan: body, rtc_configuration: { tr: { mode: 'TX', period_us: 100, high_us: 20, delay_us: 1 }, wave_count: 2, waves_verified: true } }
    : waveTable(body, path.endsWith('/write') ? 'VERIFIED' : 'PREVIEW') })
  const h = renderer.mount('RunPage', rtcRunProps({ devices: ['rtc', 'vna'].map((device_id) => ({ device_id, state: 'READY' })), setRun(value) { prepared = value } }))
  changeField(h, '采集模式', 'RTC_STOP_AND_GO')
  assert.ok(label(h.tree).includes('每通道测完经 RTC 关闭并确认后再测下一通道'))
  assert.ok(!find(h.tree, (node) => node.type === 'option' && node.props.value === 'RTC_CONTINUOUS'))
  await button(h.tree, '浏览').props.onClick(); await h.settle()
  assert.equal(button(h.tree, '准备测试').props.disabled, false, 'RTC calibration does not require a turntable or direct beam serial')
  await button(h.tree, '准备测试').props.onClick(); await h.settle()
  const request = renderer.calls.find((call) => call.path.endsWith('/prepare')).body
  assert.equal(request.test_type, 'CALIBRATION')
  assert.equal(request.topology, 'RTC_STOP_AND_GO')
  assert.ok(label(h.tree).includes('已回读确认与当前计划一致'))
  h.props = { ...h.props, run: prepared }; h.dirty = true; h.flush(); await h.settle()
  changeField(h, '极化', 'V')
  assert.equal(prepared, null)
  assert.ok(!label(h.tree).includes('已回读确认与当前计划一致'))
  h.props = { ...h.props, run: null }; h.dirty = true; h.flush(); await h.settle()
  await button(h.tree, '准备测试').props.onClick(); await h.settle()
  h.props = { ...h.props, run: prepared }; h.dirty = true; h.flush(); await h.settle()
  await button(h.tree, '写入 RTC 并校验').props.onClick(); await h.settle()
  assert.equal(prepared, null, 'a manual table write requires preparation again')
})

test('RTC wave read detects mismatch and disconnects or active measurements clear or lock verification', async () => {
  const renderer = loadRenderer({ bridge: { openDirectory: async () => '/results' }, post: async (path, body) => path.endsWith('/prepare')
    ? { run_id: 'rtc-lock', state: 'PREPARED', total: 2, plan: body }
    : { ...waveTable(body, path.endsWith('/write') ? 'VERIFIED' : path.endsWith('/read') ? 'MISMATCH' : 'PREVIEW'), entries: waveTable(body, path.endsWith('/write') ? 'VERIFIED' : path.endsWith('/read') ? 'MISMATCH' : 'PREVIEW').entries.map((entry) => ({ ...entry, readback_hex: 'actual-bytes' })) } })
  let prepared
  const h = renderer.mount('RunPage', rtcRunProps({ devices: ['rtc', 'vna'].map((device_id) => ({ device_id, state: 'READY' })), setRun(value) { prepared = value } }))
  changeField(h, '采集模式', 'RTC_STOP_AND_GO'); await h.settle()
  await button(h.tree, '写入 RTC 并校验').props.onClick(); await h.settle()
  h.props = { ...h.props, devices: [{ device_id: 'rtc', state: 'DISCONNECTED' }, { device_id: 'vna', state: 'READY' }] }; h.dirty = true; h.flush()
  assert.ok(!label(h.tree).includes('校验一致'))
  assert.equal(button(h.tree, '写入 RTC 并校验').props.disabled, true)
  h.props = { ...h.props, devices: ['rtc', 'vna'].map((device_id) => ({ device_id, state: 'READY' })) }; h.dirty = true; h.flush()
  await button(h.tree, '浏览').props.onClick(); await h.settle()
  await button(h.tree, '准备测试').props.onClick(); await h.settle()
  const originalRun = prepared
  h.props = { ...h.props, run: originalRun }; h.dirty = true; h.flush(); await h.settle()
  await button(h.tree, '读取 RTC 波位').props.onClick(); await h.settle()
  assert.equal(prepared, null, 'a mismatching read invalidates the prepared plan')
  assert.ok(label(h.tree).includes('不一致'))
  assert.ok(label(h.tree).includes('actual-bytes'))
  h.props = { ...h.props, run: { ...originalRun, state: 'RUNNING' } }; h.dirty = true; h.flush()
  assert.equal(button(h.tree, '写入 RTC 并校验').props.disabled, true)
  assert.equal(button(h.tree, '读取 RTC 波位').props.disabled, true)
  const count = renderer.calls.length
  await button(h.tree, '写入 RTC 并校验').props.onClick(); await button(h.tree, '读取 RTC 波位').props.onClick(); await h.settle()
  assert.equal(renderer.calls.length, count, 'disabled handlers cannot contact RTC during a measurement')
})
