import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/triage': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/queue': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/beds': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
