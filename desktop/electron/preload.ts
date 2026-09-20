import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('antennaDesktop', {
  openFile: (kind: 'profile' | 'coordinates' | 'data' | 'hdf'): Promise<string | null> => ipcRenderer.invoke('dialog:open-file', kind),
  openDirectory: (): Promise<string | null> => ipcRenderer.invoke('dialog:open-directory'),
  saveFile: (extension: string, defaultName: string): Promise<string | null> =>
    ipcRenderer.invoke('dialog:save-file', extension, defaultName),
  appPaths: (): Promise<{ turntableDll: string }> => ipcRenderer.invoke('app:paths')
})
