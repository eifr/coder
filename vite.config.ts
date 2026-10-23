import { defineConfig } from 'vite'
import preact from '@preact/preset-vite'
import tailwindcss from '@tailwindcss/vite'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [preact(), tailwindcss(), VitePWA({
    registerType: 'autoUpdate',
    includeAssets: ['icon.svg', 'icon-192.png', 'icon-512.png', 'icon-maskable-192.png', 'icon-maskable-512.png'],
    manifest: {
      name: 'Coder',
      short_name: 'Coder',
      description: 'AI coding assistant that runs entirely in your browser',
      theme_color: '#0e1015',
      background_color: '#0e1015',
      display: 'standalone',
      orientation: 'any',
      prefer_related_applications: false,
      icons: [
        { src: 'icon.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'any' },
        { src: 'icon-192.png', sizes: '192x192', type: 'image/png', purpose: 'any' },
        { src: 'icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any' },
        { src: 'icon-maskable-192.png', sizes: '192x192', type: 'image/png', purpose: 'maskable' },
        { src: 'icon-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' }
      ]
    },
    workbox: {
      globPatterns: ['**/*.{js,css,html,svg,png}'],
      maximumFileSizeToCacheInBytes: 15 * 1024 * 1024,
      globIgnores: ['**/ort-wasm-simd-threaded*.wasm', '**/*.wasm*'],
      runtimeCaching: [{
        urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/.*/i,
        handler: 'CacheFirst',
        options: { cacheName: 'cdn-cache', expiration: { maxEntries: 50, maxAgeSeconds: 86400 * 30 } }
      }]
    }
  })],
  optimizeDeps: {
    exclude: ['htm'],
    include: ['ai', '@browser-ai/core', '@browser-ai/web-llm', '@browser-ai/transformers-js', '@huggingface/transformers']
  },
  worker: {
    format: 'es'
  }
})
