import { defineConfig } from 'vite'
import preact from '@preact/preset-vite'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [preact(), tailwindcss()],
  optimizeDeps: {
    exclude: ['htm'],
    include: ['ai', '@browser-ai/core', '@browser-ai/web-llm', '@browser-ai/transformers-js', '@huggingface/transformers']
  },
})
