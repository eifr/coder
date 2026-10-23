import { defineConfig, type Plugin } from 'vite'
import preact from '@preact/preset-vite'
import tailwindcss from '@tailwindcss/vite'
import { VitePWA } from 'vite-plugin-pwa'

// Cloudflare Workers rejects static assets larger than 25 MiB.
const CF_ASSET_LIMIT = 25 * 1024 * 1024

// `onnxruntime-web` contains `new URL('ort-wasm-simd-threaded.asyncify.wasm',
// import.meta.url)`, so Vite copies the ~26 MB binary into the bundle. Nothing
// ever requests that copy: Transformers.js sets
// `env.backends.onnx.wasm.wasmPaths` to the jsDelivr-hosted files for the exact
// `onnxruntime-web` version at import time (cdn.jsdelivr.net/npm/onnxruntime-web@<version>/dist/).
// Drop it so `wrangler deploy` stays under the per-asset limit.
function dropBundledOrtWasm(): Plugin {
  return {
    name: 'drop-bundled-ort-wasm',
    apply: 'build',
    generateBundle(_options, bundle) {
      for (const [fileName, output] of Object.entries(bundle)) {
        if (output.type !== 'asset') continue
        if (/ort-wasm-.*\.wasm$/.test(fileName)) {
          delete bundle[fileName]
          continue
        }
        const size = typeof output.source === 'string' ? output.source.length : output.source.byteLength
        if (size > CF_ASSET_LIMIT) {
          this.warn(`Asset ${fileName} is ${(size / 1024 / 1024).toFixed(2)} MiB, over Cloudflare's 25 MiB limit`)
        }
      }
    },
  }
}

export default defineConfig({
  plugins: [preact(), tailwindcss(), dropBundledOrtWasm(), VitePWA({
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
