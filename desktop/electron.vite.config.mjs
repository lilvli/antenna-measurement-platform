import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig, externalizeDepsPlugin } from 'electron-vite'

const projectRoot = fileURLToPath(new URL('.', import.meta.url))

export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin()],
    build: { rollupOptions: { input: resolve(projectRoot, 'electron/main.ts'), external: ['electron'] } }
  },
  preload: {
    plugins: [externalizeDepsPlugin()],
    build: {
      rollupOptions: {
        input: resolve(projectRoot, 'electron/preload.ts'),
        external: ['electron'],
        // Sandboxed Electron preload scripts must be CommonJS. A .js file inherits
        // this package's "type: module" and silently fails before exposing the bridge.
        output: { format: 'cjs', entryFileNames: 'preload.cjs' }
      }
    }
  },
  renderer: {
    root: resolve(projectRoot, 'src'),
    plugins: [react()],
    build: { rollupOptions: { input: resolve(projectRoot, 'src/index.html') } }
  }
})
