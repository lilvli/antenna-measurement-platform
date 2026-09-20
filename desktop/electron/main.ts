import { app, BrowserWindow, dialog, ipcMain, Menu } from 'electron'
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'

// Keep development isolated from an installed production service that may already
// be running on 18765. An explicit environment override still takes precedence.
const preferredServicePort = Number(process.env.ANTENNA_SERVICE_PORT ?? (app.isPackaged ? 18765 : 18766))
let servicePort = preferredServicePort
let mainWindow: BrowserWindow | null = null
let serviceProcess: ChildProcessWithoutNullStreams | null = null
let closeApproved = false
let closeInProgress = false

type DialogLocations = Record<string, string>

function readDialogLocations(): DialogLocations {
  try {
    return JSON.parse(readFileSync(join(app.getPath('userData'), 'dialog-locations.json'), 'utf8')) as DialogLocations
  } catch {
    return {}
  }
}

function rememberDialogLocation(key: string, selectedPath: string, isDirectory = false): void {
  const locations = readDialogLocations()
  locations[key] = isDirectory ? selectedPath : dirname(selectedPath)
  writeFileSync(join(app.getPath('userData'), 'dialog-locations.json'), JSON.stringify(locations, null, 2), 'utf8')
}

type ServiceProbe = 'COMPATIBLE' | 'OCCUPIED' | 'UNAVAILABLE'

async function probeService(port: number): Promise<ServiceProbe> {
  try {
    const response = await fetch(`http://127.0.0.1:${port}/api/health`, {
      signal: AbortSignal.timeout(800)
    })
    if (!response.ok) return 'OCCUPIED'
    // A previous development window may have left an older service on the default port.
    // Health alone is insufficient. Require current protocol debugging and HDF slice-view
    // endpoints before reusing a previous development service on this port.
    const schema = await fetch(`http://127.0.0.1:${port}/openapi.json`, {
      signal: AbortSignal.timeout(800)
    }).then((result) => result.json()) as { paths?: Record<string, unknown> }
    const paths = schema.paths ?? {}
    return ['/api/devices/beam_controller/send', '/api/data/view', '/api/runs/current', '/api/shutdown/prepare', '/api/shutdown/cancel']
      .every((path) => paths[path]) ? 'COMPATIBLE' : 'OCCUPIED'
  } catch {
    return 'UNAVAILABLE'
  }
}

function startService(): void {
  if (process.env.ANTENNA_SERVICE_EXTERNAL === '1' || serviceProcess) return

  const packagedExecutable = join(process.resourcesPath, 'service', 'antenna-control-service.exe')
  let executable: string
  let args: string[]
  let cwd: string
  if (app.isPackaged && existsSync(packagedExecutable)) {
    executable = packagedExecutable
    args = ['--port', String(servicePort)]
    cwd = join(process.resourcesPath, 'service')
  } else {
    const workspaceRoot = resolve(app.getAppPath(), '..')
    const workspacePython = join(workspaceRoot, '.venv', 'Scripts', 'python.exe')
    // VS Code development starts use the repository virtual environment by default.
    // ANTENNA_SERVICE_PYTHON remains available for an explicit override.
    executable = process.env.ANTENNA_SERVICE_PYTHON || (existsSync(workspacePython) ? workspacePython : 'python')
    cwd = join(workspaceRoot, 'service')
    args = ['-m', 'antenna_service.main', '--port', String(servicePort)]
  }
  serviceProcess = spawn(executable, args, {
    cwd,
    windowsHide: true,
    env: { ...process.env, PYTHONUTF8: '1' }
  })
  serviceProcess.stdout.on('data', (chunk) => console.log(`[service] ${chunk}`))
  serviceProcess.stderr.on('data', (chunk) => console.error(`[service] ${chunk}`))
  serviceProcess.on('exit', () => {
    serviceProcess = null
  })
  serviceProcess.on('error', (error) => {
    console.error('[service] 启动失败:', error)
    serviceProcess = null
  })
}

type ShutdownState = {
  ready?: boolean
  control_owner: string | null
  run: { state: string; cleanup_pending: boolean } | null
}

async function serviceRequest(path: string, method = 'GET'): Promise<ShutdownState> {
  const response = await fetch(`http://127.0.0.1:${servicePort}${path}`, {
    method, signal: AbortSignal.timeout(10_000)
  })
  if (!response.ok) throw new Error(`测控服务返回 ${response.status}`)
  return await response.json() as ShutdownState
}

async function terminateOwnedService(): Promise<void> {
  const owned = serviceProcess
  if (!owned?.pid) return
  // The packaged Python onefile executable has a bootloader parent and service child.
  // Only after safe shutdown is confirmed, terminate this exact owned PID tree so
  // neither process survives the launcher. Never target unrelated services by name.
  if (process.platform === 'win32') {
    await new Promise<void>((resolvePromise, reject) => {
      const termination = spawn('taskkill', ['/PID', String(owned.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' })
      termination.once('error', reject)
      termination.once('exit', (code) => {
        if (code === 0 || owned.exitCode !== null) resolvePromise()
        else reject(new Error(`已完成设备收尾，但结束本窗口服务进程失败 (${code})`))
      })
    })
  } else {
    owned.kill()
  }
  serviceProcess = null
}

async function requestSafeClose(): Promise<void> {
  if (closeApproved || closeInProgress || !mainWindow) return
  closeInProgress = true
  let shutdownRequested = false
  try {
    const current = await serviceRequest('/api/runs/current')
    if (current.control_owner || current.run?.cleanup_pending || ['RUNNING', 'PAUSED', 'STOPPING'].includes(current.run?.state ?? '')) {
      const choice = await dialog.showMessageBox(mainWindow, {
        type: 'question', title: '设备操作尚未结束',
        message: '是否在当前操作安全结束后退出？',
        detail: '自动测试将正常停止并保存已完成数据；FLASH 写入、手动运动及设备收尾将等待完成。',
        buttons: ['正常停止后退出', '取消'], defaultId: 0, cancelId: 1
      })
      if (choice.response !== 0) return
    }
    let nextPrompt = Date.now() + 15_000
    while (true) {
      // This handshake blocks new writes, requests a normal run stop, and only reports
      // ready after the active operation has released control and devices disconnected.
      shutdownRequested = true
      const state = await serviceRequest('/api/shutdown/prepare', 'POST')
      if (state.ready) {
        // A reused service belongs to another launcher. Leave its process usable.
        if (!serviceProcess) await serviceRequest('/api/shutdown/cancel', 'POST')
        else await terminateOwnedService()
        closeApproved = true
        app.quit()
        return
      }
      if (Date.now() >= nextPrompt) {
        const choice = await dialog.showMessageBox(mainWindow, {
          type: 'info', title: '等待设备安全结束',
          message: '设备操作仍在进行，尚未关闭程序。',
          detail: '取消退出会保留窗口；已请求的正常停止仍会完成。',
          buttons: ['继续等待', '取消退出'], defaultId: 0, cancelId: 1
        })
        if (choice.response !== 0) return
        nextPrompt = Date.now() + 15_000
      }
      await new Promise((resolvePromise) => setTimeout(resolvePromise, 500))
    }
  } catch (error) {
    await dialog.showMessageBox(mainWindow, {
      type: 'error', title: '暂未退出', message: '无法确认设备操作已安全结束，窗口保持打开。',
      detail: error instanceof Error ? error.message : String(error)
    })
  } finally {
    if (shutdownRequested && !closeApproved) {
      try {
        await serviceRequest('/api/shutdown/cancel', 'POST')
      } catch (error) {
        console.error('[service] 取消退出握手失败:', error)
      }
    }
    closeInProgress = false
  }
}

async function waitForService(): Promise<void> {
  let status = await probeService(servicePort)
  if (status === 'COMPATIBLE') return
  if (status === 'OCCUPIED') {
    if (process.env.ANTENNA_SERVICE_PORT) {
      throw new Error(`指定测控服务端口 ${servicePort} 已被不兼容的服务占用`)
    }
    let selected = false
    for (let candidate = preferredServicePort + 1; candidate <= preferredServicePort + 20; candidate += 1) {
      status = await probeService(candidate)
      if (status === 'COMPATIBLE' || status === 'UNAVAILABLE') {
        servicePort = candidate
        selected = true
        break
      }
    }
    if (!selected) throw new Error('未找到可用的本机测控服务端口')
    if (status === 'COMPATIBLE') return
  }
  startService()
  const deadline = Date.now() + 20_000
  while (Date.now() < deadline) {
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 250))
    if (await probeService(servicePort) === 'COMPATIBLE') return
  }
  throw new Error('本机测控服务未能在 20 秒内启动')
}

async function createWindow(): Promise<void> {
  // The application uses its own left-side navigation and does not expose Electron's
  // default File/Edit/View/Window menu. Removing the application menu also prevents it
  // from reappearing when Alt is pressed on Windows.
  Menu.setApplicationMenu(null)
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1120,
    minHeight: 720,
    backgroundColor: '#0b1020',
    autoHideMenuBar: true,
    show: false,
    webPreferences: {
      preload: join(__dirname, '../preload/preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  })
  mainWindow.webContents.on('preload-error', (_event, preloadPath, error) => {
    console.error(`[preload] 加载失败 ${preloadPath}:`, error)
  })
  mainWindow.on('close', (event) => {
    if (closeApproved) return
    event.preventDefault()
    void requestSafeClose()
  })
  mainWindow.once('ready-to-show', () => mainWindow?.show())
  if (process.env.ELECTRON_RENDERER_URL) {
    const rendererUrl = new URL(process.env.ELECTRON_RENDERER_URL)
    rendererUrl.searchParams.set('servicePort', String(servicePort))
    await mainWindow.loadURL(rendererUrl.toString())
  } else {
    await mainWindow.loadFile(join(__dirname, '../renderer/index.html'), {
      query: { servicePort: String(servicePort) }
    })
  }
}

function registerIpc(): void {
  ipcMain.handle('dialog:open-file', async (_event, kind: string) => {
    const locations = readDialogLocations()
    const filters = kind === 'profile' || kind === 'coordinates'
      ? [{ name: 'Excel 工作簿', extensions: ['xlsx'] }]
      : kind === 'hdf'
        ? [{ name: 'HDF5 数据文件', extensions: ['hdf5', 'h5', 'hdf'] }]
        : [{ name: '天线 HDF5 数据文件', extensions: ['hdf5', 'h5', 'hdf'] }]
    const result = await dialog.showOpenDialog(mainWindow!, {
      properties: ['openFile'],
      filters,
      defaultPath: locations[kind]
    })
    if (result.canceled) return null
    rememberDialogLocation(kind, result.filePaths[0])
    return result.filePaths[0]
  })
  ipcMain.handle('dialog:open-directory', async () => {
    const result = await dialog.showOpenDialog(mainWindow!, {
      properties: ['openDirectory', 'createDirectory'],
      defaultPath: readDialogLocations().directory
    })
    if (result.canceled) return null
    rememberDialogLocation('directory', result.filePaths[0], true)
    return result.filePaths[0]
  })
  ipcMain.handle('dialog:save-file', async (_event, extension: string, defaultName: string) => {
    const locations = readDialogLocations()
    const result = await dialog.showSaveDialog(mainWindow!, {
      defaultPath: locations.save ? join(locations.save, defaultName) : defaultName,
      filters: [{ name: `天线 ${extension} 文件`, extensions: [extension] }]
    })
    if (result.canceled || !result.filePath) return null
    rememberDialogLocation('save', result.filePath)
    return result.filePath
  })
  ipcMain.handle('app:paths', () => ({
    turntableDll: app.isPackaged
      ? join(process.resourcesPath, 'drivers', 'turntable', 'ImacFxDll.dll')
      : resolve(app.getAppPath(), 'resources', 'turntable', 'ImacFxDll.dll')
  }))
}

app.whenReady().then(async () => {
  registerIpc()
  try {
    await waitForService()
    await createWindow()
  } catch (error) {
    dialog.showErrorBox('启动失败', error instanceof Error ? error.message : String(error))
    app.quit()
  }
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', (event) => {
  if (mainWindow && !mainWindow.isDestroyed() && !closeApproved) {
    event.preventDefault()
    void requestSafeClose()
    return
  }
  // The owned service tree was terminated only after the shutdown handshake.
})
