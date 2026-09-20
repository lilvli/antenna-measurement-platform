import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const debugPort = Number(process.env.ANTENNA_DEBUG_PORT ?? 9223)
const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')
const profilePath = resolve(repositoryRoot, '功能总结文档和必要协议', '当前模板与示例', 'x_radar_天线协议配置包_V1.0.xlsx')
const coordinatePath = resolve(repositoryRoot, '功能总结文档和必要协议', '原协议配置包与坐标表模板与示例', '天线通道坐标表', '天线通道坐标表_256通道_H极化_8SPIx8芯片x4通道_v2.1_SIMULATED.xlsx')
const targets = await fetch(`http://127.0.0.1:${debugPort}/json`).then((response) => response.json())
const target = targets.find((item) => item.type === 'page' && item.title === '天线标校与方向图测试平台')
if (!target?.webSocketDebuggerUrl) throw new Error('未找到天线桌面程序的调试页面')

const socket = new WebSocket(target.webSocketDebuggerUrl)
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true })
  socket.addEventListener('error', reject, { once: true })
})

let nextId = 1
const pending = new Map()
socket.addEventListener('message', (event) => {
  const message = JSON.parse(event.data)
  if (!message.id) return
  const callback = pending.get(message.id)
  if (!callback) return
  pending.delete(message.id)
  message.error ? callback.reject(new Error(message.error.message)) : callback.resolve(message.result)
})

function call(method, params = {}) {
  const id = nextId++
  socket.send(JSON.stringify({ id, method, params }))
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }))
}

async function evaluate(expression) {
  const result = await call('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text)
  return result.result.value
}

await call('Runtime.enable')
const bridgeType = await evaluate('typeof window.antennaDesktop')
const appPaths = await evaluate('window.antennaDesktop?.appPaths()')
const protocolSend = await evaluate(`(async () => {
  const port = new URLSearchParams(location.search).get('servicePort') ?? '18765'
  const base = 'http://127.0.0.1:' + port
  const request = (path, body) => fetch(base + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  }).then(async (response) => {
    const value = await response.json()
    if (!response.ok) throw new Error(JSON.stringify(value))
    return value
  })
  const profile = await request('/api/assets/profile/load', { path: ${JSON.stringify(profilePath)} })
  await request('/api/assets/coordinates/load', { path: ${JSON.stringify(coordinatePath)} })
  for (const deviceId of ['beam_controller', 'rtc', 'turntable', 'vna']) {
    await request('/api/devices/' + deviceId + '/connect', { source: 'SIMULATED', parameters: {} })
  }
  return request('/api/devices/beam_controller/send', {
    profile_id: profile.asset_id,
    command_id: 'QUERY_INIT_STATE',
    array_id: 0,
    parameters: {}
  })
})()`)
await new Promise((resolve) => setTimeout(resolve, 200))
const clicked = await evaluate(`(() => {
  const button = [...document.querySelectorAll('nav button')].find((item) => item.textContent?.includes('设备调试'))
  button?.click()
  return Boolean(button)
})()`)
for (let attempt = 0; attempt < 50; attempt += 1) {
  if (await evaluate(`Boolean(document.querySelector('#beam-baud-select'))`)) break
  await new Promise((resolve) => setTimeout(resolve, 100))
}
const page = await evaluate(`({
  title: document.querySelector('main h1')?.textContent,
  healthText: document.querySelector('.top-status')?.textContent,
  deviceCardCount: document.querySelectorAll('.device-cards .card').length,
  hasDeviceHeading: document.body.innerText.includes('真实设备与模拟器使用同一业务接口'),
  hasScale10000: document.body.innerText.includes('比例系数') && document.body.innerText.includes('10000'),
  hasInternalDriver: Boolean(document.querySelector('#turntable-internal-driver')),
  hasDllChooser: [...document.querySelectorAll('.field > span')].some((item) => item.textContent?.includes('ImacFxDll')),
  serialOptionCount: document.querySelector('#beam-port-select')?.options.length ?? 0,
  serialOptions: [...(document.querySelector('#beam-port-select')?.options ?? [])].map((option) => option.value),
  baudRate: document.querySelector('#beam-baud-select')?.value,
  hasProtocolDebug: document.body.innerText.includes('波控协议调试'),
  hasCommandSelect: Boolean(document.querySelector('#beam-command-select')),
  hasRawLog: Boolean(document.querySelector('#device-raw-log')),
  rawLogRows: document.querySelectorAll('#device-raw-log .device-log-row').length,
  disconnectButtonCount: document.querySelectorAll('.device-disconnect').length,
  beamPortDisabled: document.querySelector('#beam-port-select')?.disabled,
  beamBaudDisabled: document.querySelector('#beam-baud-select')?.disabled,
  turntableSpeedValue: document.querySelector('#turntable-speed-input')?.value,
  turntableSpeedMin: document.querySelector('#turntable-speed-input')?.min,
  turntableSpeedStep: document.querySelector('#turntable-speed-input')?.step,
  hasReadPosition: Boolean(document.querySelector('#turntable-read-position')),
  fatalScreen: document.body.innerText.includes('RENDERER ERROR'),
  rootChildren: document.querySelector('#root')?.childElementCount ?? 0
})`)
const resourceUrls = await evaluate(`performance.getEntriesByType('resource').map((entry) => entry.name).filter((name) => name.includes('127.0.0.1'))`)
const readPositionClicked = await evaluate(`(() => {
  const button = document.querySelector('#turntable-read-position')
  button?.click()
  return Boolean(button && !button.disabled)
})()`)
await new Promise((resolve) => setTimeout(resolve, 300))
const turntableReadbackRows = await evaluate(`document.querySelectorAll('#turntable-position-readback > div').length`)
const disconnectClicked = await evaluate(`(() => {
  const card = [...document.querySelectorAll('.device-cards .card')].find((item) => item.textContent?.includes('波控机'))
  const button = card?.querySelector('.device-disconnect')
  button?.click()
  return Boolean(button)
})()`)
await new Promise((resolve) => setTimeout(resolve, 400))
const unlockedAfterDisconnect = await evaluate(`({
  beamPort: !document.querySelector('#beam-port-select')?.disabled,
  beamBaud: !document.querySelector('#beam-baud-select')?.disabled,
  hasConnectButton: [...document.querySelectorAll('.device-cards .card')].find((item) => item.textContent?.includes('波控机'))?.textContent?.includes('连接真实设备')
})`)
const refreshClicked = await evaluate(`(() => {
  const refresh = document.querySelector('.select-with-action button')
  refresh?.click()
  return Boolean(refresh)
})()`)
await new Promise((resolve) => setTimeout(resolve, 5000))
const expiredToastCount = await evaluate(`document.querySelectorAll('.toast').length`)
const selectedPort = await evaluate(`(() => {
  const select = document.querySelector('#beam-port-select')
  if (!select) return ''
  select.value = [...select.options].some((option) => option.value === 'COM100') ? 'COM100' : select.value
  select.dispatchEvent(new Event('change', { bubbles: true }))
  return select.value
})()`)
await evaluate(`(() => {
  const button = [...document.querySelectorAll('nav button')].find((item) => item.textContent?.includes('测试执行'))
  button?.click()
  return Boolean(button)
})()`)
await new Promise((resolve) => setTimeout(resolve, 200))
await evaluate(`([...document.querySelectorAll('.page-view:not(.inactive) .segmented button')].find((item) => item.textContent?.includes('逐通道定点标校')))?.click()`)
await new Promise((resolve) => setTimeout(resolve, 100))
const calibrationHeatmap = await evaluate(`({
  channelCount: Number(document.querySelector('.channel-map')?.dataset.channelCount ?? 0),
  kind: document.querySelector('.channel-map')?.dataset.heatmapKind,
  height: document.querySelector('.channel-map') ? getComputedStyle(document.querySelector('.channel-map')).height : '',
  hasRelativeScale: Boolean(document.querySelector('.relative-scale'))
})`)
const stateSeeded = await evaluate(`(() => {
  const field = [...document.querySelectorAll('.page-view:not(.inactive) .field')].find((item) => item.querySelector(':scope > span')?.textContent === '文件名')
  const input = field?.querySelector('input')
  if (!input) return false
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set
  setter.call(input, 'state_keep')
  input.dispatchEvent(new Event('input', { bubbles: true }))
  const pattern = [...document.querySelectorAll('.page-view:not(.inactive) .segmented button')].find((item) => item.textContent?.includes('方向图扫描'))
  pattern?.click()
  return Boolean(pattern)
})()`)
await new Promise((resolve) => setTimeout(resolve, 200))
const patternHeatmap = await evaluate(`({
  channelCount: Number(document.querySelector('.page-view:not(.inactive) .channel-map')?.dataset.channelCount ?? 0),
  kind: document.querySelector('.page-view:not(.inactive) .channel-map')?.dataset.heatmapKind,
  hasHdfNameHint: document.querySelector('.page-view:not(.inactive)')?.innerText.includes('_方向图_YYYYMMDD_HHMMSS_mmm.hdf5')
})`)
await evaluate(`([...document.querySelectorAll('nav button')].find((item) => item.textContent?.includes('设备调试')))?.click()`)
await new Promise((resolve) => setTimeout(resolve, 100))
const persistedPort = await evaluate(`document.querySelector('#beam-port-select')?.value`)
await evaluate(`([...document.querySelectorAll('nav button')].find((item) => item.textContent?.includes('测试执行')))?.click()`)
await new Promise((resolve) => setTimeout(resolve, 100))
const persistedRunState = await evaluate(`({
  filename: [...document.querySelectorAll('.page-view:not(.inactive) .field')].find((item) => item.querySelector(':scope > span')?.textContent === '文件名')?.querySelector('input')?.value,
  heatmapKind: document.querySelector('.page-view:not(.inactive) .channel-map')?.dataset.heatmapKind
})`)
socket.close()

const report = { bridgeType, appPaths, protocolSend, clicked, page, resourceUrls, readPositionClicked, turntableReadbackRows, disconnectClicked, unlockedAfterDisconnect, refreshClicked, expiredToastCount, selectedPort, calibrationHeatmap, stateSeeded, patternHeatmap, persistedPort, persistedRunState }
console.log(JSON.stringify(report, null, 2))
if (
  bridgeType !== 'object'
  || !clicked
  || !page.hasDeviceHeading
  || !page.hasScale10000
  || !page.hasInternalDriver
  || page.hasDllChooser
  || !page.serialOptions.includes('COM99')
  || !page.serialOptions.includes('COM100')
  || page.baudRate !== '115200'
  || !page.hasProtocolDebug
  || !page.hasCommandSelect
  || !page.hasRawLog
  || page.rawLogRows < 1
  || page.disconnectButtonCount !== 4
  || !page.beamPortDisabled
  || !page.beamBaudDisabled
  || page.turntableSpeedValue !== '1.0000'
  || page.turntableSpeedMin !== '0.0001'
  || page.turntableSpeedStep !== '0.0001'
  || !page.hasReadPosition
  || !readPositionClicked
  || turntableReadbackRows !== 5
  || !disconnectClicked
  || !unlockedAfterDisconnect.beamPort
  || !unlockedAfterDisconnect.beamBaud
  || !unlockedAfterDisconnect.hasConnectButton
  || !refreshClicked
  || page.fatalScreen
  || page.rootChildren === 0
  || expiredToastCount !== 0
  || calibrationHeatmap.channelCount !== 256
  || calibrationHeatmap.kind !== 'CALIBRATION'
  || calibrationHeatmap.height !== '265px'
  || !calibrationHeatmap.hasRelativeScale
  || !stateSeeded
  || patternHeatmap.channelCount !== 21
  || patternHeatmap.kind !== 'PATTERN'
  || !patternHeatmap.hasHdfNameHint
  || persistedPort !== selectedPort
  || persistedRunState.filename !== 'state_keep'
  || persistedRunState.heatmapKind !== 'PATTERN'
) {
  process.exitCode = 1
}
