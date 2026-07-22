import { defineConfig } from 'vite'
import preact from '@preact/preset-vite'
import tailwindcss from '@tailwindcss/vite'
import { VitePWA } from 'vite-plugin-pwa'

import { cloudflare } from "@cloudflare/vite-plugin";

export default defineConfig({
  plugins: [preact(), tailwindcss(), VitePWA({
    registerType: 'autoUpdate',
    includeAssets: ['icon.svg', 'icon-192.png', 'icon-512.png'],
    manifest: {
      name: 'WebCoder',
      short_name: 'WebCoder',
      description: 'AI coding assistant that runs entirely in your browser',
      theme_color: '#1e1e1e',
      background_color: '#1e1e1e',
      display: 'standalone',
      orientation: 'any',
      prefer_related_applications: false,
      icons: [
        { src: 'icon.svg', sizes: '192x192', type: 'image/svg+xml', purpose: 'any' },
        { src: 'icon.svg', sizes: '512x512', type: 'image/svg+xml', purpose: 'any' },
        { src: 'icon-192.png', sizes: '192x192', type: 'image/png', purpose: 'any maskable' },
        { src: 'icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any maskable' }
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
  }), cloudflare()],
  optimizeDeps: {
    exclude: ['htm'],
    include: ['ai', '@browser-ai/core', '@browser-ai/web-llm', '@browser-ai/transformers-js', '@huggingface/transformers']
  },
  worker: {
    format: 'es'
  }
})