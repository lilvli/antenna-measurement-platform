import { Component, type ErrorInfo, type ReactNode } from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles.css'

class RendererErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, details: ErrorInfo) {
    console.error('[renderer] 页面渲染失败', error, details.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="fatal-screen">
          <span>RENDERER ERROR</span>
          <h1>界面发生异常，但程序没有黑屏退出</h1>
          <p>{this.state.error.message}</p>
          <button onClick={() => window.location.reload()}>重新加载界面</button>
        </div>
      )
    }
    return this.props.children
  }
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <RendererErrorBoundary><App /></RendererErrorBoundary>
)
