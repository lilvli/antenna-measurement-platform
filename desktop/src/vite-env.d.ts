/// <reference types="vite/client" />

interface Window {
  antennaDesktop: {
    openFile(kind: 'profile' | 'coordinates' | 'data' | 'hdf'): Promise<string | null>
    openDirectory(): Promise<string | null>
    saveFile(extension: string, defaultName: string): Promise<string | null>
    appPaths(): Promise<{ turntableDll: string }>
  }
}
