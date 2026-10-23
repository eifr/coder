import { streamText, jsonSchema, type ModelMessage, type LanguageModel } from 'ai'
import { browserAI, doesBrowserSupportBrowserAI } from '@browser-ai/core'
import { webLLM, doesBrowserSupportWebLLM } from '@browser-ai/web-llm'
import { transformersJS, doesBrowserSupportTransformersJS } from '@browser-ai/transformers-js'
import { prebuiltAppConfig } from '@mlc-ai/web-llm'

export const PROVIDERS = {
  BROWSER_AI: 'browser-ai',
  WEBLLM: 'webllm',
  TRANSFORMERS_JS: 'transformers-js',
} as const

export type Provider = (typeof PROVIDERS)[keyof typeof PROVIDERS]

export interface ModelDescriptor {
  id: string
  vram: number
  provider: Provider
  label?: string
  preferred?: number
}

interface InitProgress {
  text: string
  progress: number
}

// Curated from `prebuiltAppConfig.model_list` (q4f16_1 builds). `vram` is the
// catalog's `vram_required_MB`, rounded. Verify IDs still exist in the bundle
// when bumping @mlc-ai/web-llm.
const WEBLLM_MODELS = [
  { id: 'Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC', vram: 5107, label: 'Qwen2.5 Coder 7B', preferred: 1 },
  { id: 'Qwen2.5-Coder-3B-Instruct-q4f16_1-MLC', vram: 2505, label: 'Qwen2.5 Coder 3B', preferred: 2 },
  { id: 'Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC', vram: 1630, label: 'Qwen2.5 Coder 1.5B', preferred: 3 },
  { id: 'Qwen3.5-2B-q4f16_1-MLC', vram: 2245, label: 'Qwen3.5 2B', preferred: 4 },
  { id: 'Llama-3.2-1B-Instruct-q4f16_1-MLC', vram: 879, label: 'Llama 3.2 1B', preferred: 5 },
  { id: 'gemma3-1b-it-q4f16_1-MLC', vram: 711, label: 'Gemma 3 1B', preferred: 6 },
  { id: 'Qwen3.5-0.8B-q4f16_1-MLC', vram: 1629, label: 'Qwen3.5 0.8B' },
  { id: 'Qwen3.5-4B-q4f16_1-MLC', vram: 3868, label: 'Qwen3.5 4B' },
  { id: 'Qwen3-4B-q4f16_1-MLC', vram: 3432, label: 'Qwen3 4B' },
  { id: 'Phi-4-mini-instruct-q4f16_1-MLC', vram: 3438, label: 'Phi-4 Mini' },
  { id: 'Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC', vram: 2864, label: 'Ministral 3 3B' },
  { id: 'gemma-4-E2B-it-q4f16_1-MLC', vram: 3387, label: 'Gemma 4 E2B' },
  { id: 'Qwen3-8B-q4f16_1-MLC', vram: 5696, label: 'Qwen3 8B' },
  { id: 'Qwen3.5-9B-q4f16_1-MLC', vram: 6433, label: 'Qwen3.5 9B' },
  { id: 'DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC', vram: 5107, label: 'DeepSeek R1 Distill 7B' },
]

// ONNX builds loadable by @huggingface/transformers (q4f16 by default). Values
// are the q4f16 WebLLM catalog sizes where available, otherwise weight-size
// estimates; Transformers.js runs on CPU/WASM too, so these only gate what the
// VRAM probe exposes.
const TRANSFORMERS_MODELS = [
  { id: 'HuggingFaceTB/SmolLM2-360M-Instruct', vram: 512, label: 'SmolLM2 360M' },
  { id: 'onnx-community/Qwen2.5-Coder-0.5B-Instruct', vram: 1024, label: 'Qwen2.5 Coder 0.5B' },
  { id: 'onnx-community/Llama-3.2-1B-Instruct-ONNX', vram: 1024, label: 'Llama 3.2 1B' },
  { id: 'onnx-community/Qwen2.5-Coder-1.5B-Instruct', vram: 1792, label: 'Qwen2.5 Coder 1.5B' },
  { id: 'HuggingFaceTB/SmolLM2-1.7B-Instruct', vram: 2048, label: 'SmolLM2 1.7B' },
  { id: 'onnx-community/Qwen3-1.7B-ONNX', vram: 2048, label: 'Qwen3 1.7B' },
  { id: 'onnx-community/Qwen2.5-Coder-3B-Instruct', vram: 2560, label: 'Qwen2.5 Coder 3B' },
  { id: 'onnx-community/Llama-3.2-3B-Instruct-ONNX', vram: 2560, label: 'Llama 3.2 3B' },
  { id: 'onnx-community/Qwen3-4B-Instruct-2507-ONNX', vram: 4096, label: 'Qwen3 4B Instruct' },
]

const BROWSER_AI_MODELS = [
  { id: 'chrome-built-in', vram: 0, label: 'Chrome Built-in AI' },
]

let modelsCache: ModelDescriptor[] | null = null

export function getAvailableModels(vramMB?: number): ModelDescriptor[] {
  if (modelsCache) return modelsCache
  const limit = vramMB || 4096
  const all: ModelDescriptor[] = [
    ...WEBLLM_MODELS.filter(m => m.vram <= limit).map(m => ({ ...m, provider: PROVIDERS.WEBLLM })),
    ...TRANSFORMERS_MODELS.filter(m => m.vram <= limit).map(m => ({ ...m, provider: PROVIDERS.TRANSFORMERS_JS })),
    ...BROWSER_AI_MODELS.filter(m => m.vram <= limit).map(m => ({ ...m, provider: PROVIDERS.BROWSER_AI })),
  ]
  modelsCache = all
  return all
}

export async function estimateVRAM(): Promise<number> {
  const isMobile = /Android|iPhone|iPad|iPod|webOS/i.test(navigator.userAgent)
  if (!navigator.gpu) return isMobile ? 1024 : 2048

  try {
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) return isMobile ? 1024 : 2048

    const props = adapter.limits
    const maxStorageBuffer = props?.maxStorageBufferBindingSize || 0
    const maxBuffer = props?.maxBufferSize || 0

    let vramGuess: number
    if (maxStorageBuffer > 2_000_000_000 || maxBuffer > 2_000_000_000) {
      vramGuess = 8192
    } else if (maxStorageBuffer > 1_000_000_000 || maxBuffer > 1_000_000_000) {
      vramGuess = 4096
    } else {
      vramGuess = 2048
    }

    if (isMobile) vramGuess = Math.min(vramGuess, 4096)
    return vramGuess
  } catch {
    return isMobile ? 1024 : 2048
  }
}

export function getPreferredModel(models: ModelDescriptor[]): ModelDescriptor | null {
  const sorted = [...models].sort((a, b) => (a.preferred || 99) - (b.preferred || 99))
  return sorted[0] || null
}

export async function checkProviderAvailability(provider: Provider): Promise<boolean> {
  switch (provider) {
    case PROVIDERS.BROWSER_AI: {
      return doesBrowserSupportBrowserAI()
    }
    case PROVIDERS.WEBLLM:
      return doesBrowserSupportWebLLM()
    case PROVIDERS.TRANSFORMERS_JS:
      // Transformers.js falls back to WASM/CPU when WebGPU is missing, so it
      // is usable on machines (e.g. Chromium on Linux without WebGPU) where
      // WebLLM and Chrome Built-in AI are not.
      return doesBrowserSupportTransformersJS()
    default:
      return false
  }
}

export type BrowserAIStatus =
  | 'available'
  | 'needs-download'
  | 'unavailable'
  | 'unsupported'
  | 'insecure-context'

export interface BrowserAIDetail {
  status: BrowserAIStatus
  /** Short label for the (possibly disabled) provider dropdown option. */
  short: string
  /** Actionable explanation shown when loading or chatting fails. */
  message: string
}

function isChromiumWithoutChrome(): boolean {
  try {
    const nav = navigator as any
    const brands: Array<{ brand: string }> | undefined = nav?.userAgentData?.brands
    if (Array.isArray(brands) && brands.length > 0) {
      const names = brands.map(b => b.brand)
      const hasChromium = names.some(n => n.toLowerCase().includes('chromium'))
      const hasGoogleChrome = names.some(n => n === 'Google Chrome')
      if (hasChromium && !hasGoogleChrome) return true
      if (hasGoogleChrome) return false
    }
    const ua = navigator.userAgent || ''
    // Open-source Chromium UAs carry a Chromium/ token without a Chrome/ token.
    return /Chromium\//.test(ua) && !/ Chrome\//.test(ua)
  } catch {
    return false
  }
}

function isLinux(): boolean {
  try {
    const platform = (navigator as any).userAgentData?.platform || (navigator as any).platform || ''
    return /linux/i.test(platform) || /linux/i.test(navigator.userAgent || '')
  } catch {
    return false
  }
}

function chromiumMessage(): string {
  return 'This looks like open-source Chromium, which does not ship Google\u2019s proprietary on-device model (Gemini Nano). Chrome Built-in AI only works in official Google Chrome (138+ on desktop) with the model downloaded and the Prompt API enabled. On this browser use WebLLM (needs WebGPU) or Transformers.js (runs on CPU) instead.'
}

/**
 * Thorough Chrome Built-in AI check. `doesBrowserSupportBrowserAI()` only
 * tests for the `LanguageModel` global, which says nothing about whether the
 * on-device model is actually present — on Chromium/Linux the global is
 * usually missing entirely, or `availability()` reports `unavailable`.
 */
export async function getBrowserAIStatus(): Promise<BrowserAIDetail> {
  if (typeof window !== 'undefined' && (window as any).isSecureContext === false) {
    return {
      status: 'insecure-context',
      short: 'Chrome AI (needs HTTPS)',
      message: 'Chrome Built-in AI is only exposed in secure contexts. Serve the app over HTTPS or from localhost (`vite dev` on localhost is fine) — it is not available over plain http:// LAN addresses or file://.',
    }
  }
  if (!doesBrowserSupportBrowserAI()) {
    if (isChromiumWithoutChrome()) {
      return { status: 'unsupported', short: 'Chrome AI (needs Google Chrome)', message: chromiumMessage() }
    }
    return {
      status: 'unsupported',
      short: 'Chrome AI (not available)',
      message: 'This browser does not expose the Prompt API (`LanguageModel` global is missing). Chrome Built-in AI needs official Google Chrome 138+ or Edge 138+ on desktop, a secure context (HTTPS/localhost), and the on-device model enabled under chrome://flags. Use WebLLM or Transformers.js instead.',
    }
  }
  try {
    const availability = await (globalThis as any).LanguageModel.availability()
    if (availability === 'available') {
      return { status: 'available', short: 'Chrome Built-in AI', message: 'Ready.' }
    }
    if (availability === 'downloadable' || availability === 'downloading' || availability === 'available-after-download') {
      const linuxNote = isLinux() ? ' On Linux the download is several GB and can take a long time — keep the tab open.' : ''
      return {
        status: 'needs-download',
        short: 'Chrome AI (model download needed)',
        message: `The browser exposes the Prompt API but the on-device model is not ready yet (status: ${availability}). Click Load Model and wait for the download to finish.${linuxNote} Make sure the Prompt API flags are enabled and there is enough free disk space.`,
      }
    }
    if (isChromiumWithoutChrome()) {
      return { status: 'unavailable', short: 'Chrome AI (needs Google Chrome)', message: chromiumMessage() }
    }
    return {
      status: 'unavailable',
      short: 'Chrome AI (unavailable)',
      message: 'The browser exposes the Prompt API but reports the on-device model as unavailable. In official Chrome: enable the Prompt API flags under chrome://flags, then open chrome://components and make sure “Optimization Guide On Device Model” is downloaded (this component does not exist in open-source Chromium). On Linux also check you have several GB of free disk space.',
    }
  } catch {
    return {
      status: 'unsupported',
      short: 'Chrome AI (not available)',
      message: 'Could not query the Prompt API status. Chrome Built-in AI needs official Google Chrome 138+ on desktop with the on-device model enabled. Use WebLLM or Transformers.js instead.',
    }
  }
}

export function getProviderLabel(provider: Provider): string {
  switch (provider) {
    case PROVIDERS.BROWSER_AI: return 'Chrome Built-in AI'
    case PROVIDERS.WEBLLM: return 'WebLLM'
    case PROVIDERS.TRANSFORMERS_JS: return 'Transformers.js'
    default: return provider
  }
}

export async function initProviderModel(
  provider: Provider,
  modelId: string,
  onProgress?: (progress: InitProgress) => void
): Promise<LanguageModel> {
  switch (provider) {
    case PROVIDERS.BROWSER_AI: {
      if (!doesBrowserSupportBrowserAI()) {
        throw new Error('Browser AI is not supported in this browser. Use Chrome 127+ or Edge 127+ with AI features enabled.')
      }
      // Do NOT keep a pre-created session: the underlying SessionManager
      // caches the first session and ignores the system prompt on later
      // getSession() calls, so keeping it would permanently drop the
      // SYSTEM_PROMPT for all future chats. Warm up (triggers model
      // download with progress) then destroy the session so the first
      // streamText() call creates a fresh one with the correct system prompt.
      const model = browserAI()
      try {
        await model.createSessionWithProgress((p) => {
          onProgress?.({ text: 'Downloading Chrome AI model...', progress: p })
        })
      } catch (e) {
        console.warn('Chrome AI warm-up failed, will retry on first message:', e)
      }
      try {
        ;(model as any).sessionManager?.destroySession?.()
        ;(model as any).sessionManager?.destroy?.()
      } catch {}
      return model
    }
    case PROVIDERS.WEBLLM: {
      const model = webLLM(modelId, {
        engineConfig: {
          appConfig: { ...prebuiltAppConfig, cacheBackend: 'opfs' },
        },
        initProgressCallback: (r) => {
          onProgress?.({ text: r.text || `Loading ${modelId}...`, progress: r.progress || 0 })
        },
      })
      await new Promise(resolve => setTimeout(resolve, 100))
      return model
    }
    case PROVIDERS.TRANSFORMERS_JS: {
      return transformersJS(modelId, {
        initProgressCallback: (p) => {
          onProgress?.({ text: `Loading ${modelId}...`, progress: typeof p === 'number' ? p : 0 })
        },
      })
    }
    default:
      throw new Error(`Unknown provider: ${provider}`)
  }
}

interface CreateChatStreamOptions {
  provider: Provider
  model: LanguageModel | null
  messages: ModelMessage[]
  system?: string
}

export function createTools(
  readFileFn: (path: string) => Promise<string | null>,
  editFileFn: (path: string, content: string) => Promise<boolean>,
  createFileFn: (path: string, content: string) => Promise<boolean>,
  deleteFileFn: (path: string) => Promise<boolean>
): any {
  return {
    readFile: {
      description: 'Read the contents of a file in the project',
      inputSchema: jsonSchema({
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path relative to project root' },
        },
        required: ['path'],
      }),
      execute: async ({ path }: { path: string }) => {
        const content = await readFileFn(path)
        return content ? { path, content } : { error: `File not found: ${path}` }
      },
    },
    editFile: {
      description: 'Write content to an existing file. Overwrites the entire file.',
      inputSchema: jsonSchema({
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path relative to project root' },
          content: { type: 'string', description: 'Full file content to write' },
        },
        required: ['path', 'content'],
      }),
      execute: async ({ path, content }: { path: string; content: string }) => {
        const ok = await editFileFn(path, content)
        return ok ? { path, success: true } : { error: `Failed to write ${path}` }
      },
    },
    createFile: {
      description: 'Create a new file with the given content. Fails if the file already exists.',
      inputSchema: jsonSchema({
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path relative to project root' },
          content: { type: 'string', description: 'File content' },
        },
        required: ['path', 'content'],
      }),
      execute: async ({ path, content }: { path: string; content: string }) => {
        const ok = await createFileFn(path, content)
        return ok ? { path, success: true } : { error: `Failed to create ${path} (may already exist)` }
      },
    },
    deleteFile: {
      description: 'Delete a file from the project',
      inputSchema: jsonSchema({
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path relative to project root' },
        },
        required: ['path'],
      }),
      execute: async ({ path }: { path: string }) => {
        const ok = await deleteFileFn(path)
        return ok ? { path, success: true } : { error: `Failed to delete ${path}` }
      },
    },
  }
}

export function createChatStream({ provider, model, messages, system, tools, abortSignal }: CreateChatStreamOptions & { tools?: Record<string, any>; abortSignal?: AbortSignal }) {
  const chatModel = model || (provider === PROVIDERS.BROWSER_AI ? browserAI() : null)
  if (!chatModel) throw new Error(`Model not initialized for provider: ${provider}`)

  // Chrome Built-in AI normally runs through the native Prompt API harness in
  // chrome-harness.ts (native tool use). This AI SDK path remains as the
  // generic fallback for providers that implement LanguageModelV3.
  return streamText({
    model: chatModel,
    messages,
    system,
    temperature: 0.3,
    maxOutputTokens: 4096,
    topP: 0.95,
    tools: tools as any,
    allowSystemInMessages: true,
    abortSignal,
  })
}
