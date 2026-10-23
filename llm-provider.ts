import { streamText, jsonSchema, stepCountIs, type ModelMessage, type LanguageModel } from 'ai'
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

const CHROMIUM_MESSAGE = 'This looks like open-source Chromium, which does not ship Google\u2019s proprietary on-device model (Gemini Nano). Built-in AI only exists in official Google Chrome on desktop \u2014 the Prompt API is stable on the web in Chrome 148+ (Chrome 138+ in extensions), while Summarizer, Translator, and Language Detector are stable in Chrome 138+. Use WebLLM (needs WebGPU) or Transformers.js (runs on CPU) on this browser instead.'

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
      return { status: 'unsupported', short: 'Chrome AI (needs Google Chrome)', message: CHROMIUM_MESSAGE }
    }
    return {
      status: 'unsupported',
      short: 'Chrome AI (not available)',
      message: 'This browser does not expose the Prompt API (`LanguageModel` global is missing). On the web the Prompt API is stable in Chrome 148+ (Chrome 138+ in extensions); Summarizer, Translator, and Language Detector are stable in Chrome 138+. Built-in AI also requires a secure context (HTTPS/localhost) and the on-device model. Use WebLLM or Transformers.js instead.',
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
        message: `The browser exposes the Prompt API but the on-device model is not ready yet (status: ${availability}). Click Load Model and keep the tab open while Chrome downloads it.${linuxNote} Check that Chrome has enough free disk space.`,
      }
    }
    if (isChromiumWithoutChrome()) {
      return { status: 'unavailable', short: 'Chrome AI (needs Google Chrome)', message: CHROMIUM_MESSAGE }
    }
    return {
      status: 'unavailable',
      short: 'Chrome AI (unavailable)',
      message: 'The browser exposes the Prompt API but reports the on-device model as unavailable. Update Chrome and check chrome://components → “Optimization Guide On Device Model” (this component does not exist in open-source Chromium). On Linux also check you have several GB of free disk space.',
    }
  } catch {
    return {
      status: 'unsupported',
      short: 'Chrome AI (not available)',
      message: 'Could not query the Prompt API status. On the web the Prompt API is stable in Chrome 148+ on desktop. Use WebLLM or Transformers.js instead.',
    }
  }
}

export type BuiltInApiStatus = 'available' | 'downloadable' | 'downloading' | 'unavailable' | 'unsupported'

export interface BuiltInApiInfo {
  id: string
  label: string
  description: string
  status: BuiltInApiStatus
  note?: string
}

/**
 * Probes one built-in AI API by its global constructor. APIs are shipped at
 * different stages and Chromium builds omit them entirely, so the only honest
 * signal is the global plus `availability()`; a missing `availability()` is
 * treated as available (older revisions).
 */
async function probeBuiltInApi(globalName: string, options?: Record<string, string>): Promise<BuiltInApiStatus> {
  // `globalThis` has no index signature; names come from the fixed list below.
  const globals = globalThis as unknown as Record<string, unknown>
  const ctor = globals[globalName]
  if (typeof ctor !== 'function') return 'unsupported'
  if (!('availability' in ctor)) return 'available'
  const probe: unknown = ctor.availability
  if (typeof probe !== 'function') return 'available'
  try {
    const status: unknown = await Reflect.apply(probe, ctor, [options])
    if (status === 'available') return 'available'
    if (status === 'downloadable' || status === 'available-after-download') return 'downloadable'
    if (status === 'downloading') return 'downloading'
    return 'unavailable'
  } catch {
    return 'unavailable'
  }
}

/**
 * Live matrix of the browser's built-in AI APIs. The Translator probe uses an
 * English → Spanish pair because its availability is per language pair.
 */
export async function getBuiltInAICapabilities(): Promise<BuiltInApiInfo[]> {
  const [prompt, summarizer, translator, languageDetector, writer, rewriter, proofreader] = await Promise.all([
    probeBuiltInApi('LanguageModel'),
    probeBuiltInApi('Summarizer'),
    probeBuiltInApi('Translator', { sourceLanguage: 'en', targetLanguage: 'es' }),
    probeBuiltInApi('LanguageDetector'),
    probeBuiltInApi('Writer'),
    probeBuiltInApi('Rewriter'),
    probeBuiltInApi('Proofreader'),
  ])

  return [
    {
      id: 'prompt',
      label: 'Prompt API',
      description: 'Free-form prompts, tool use, and image input. Powers chat and file edits.',
      status: prompt,
      note: 'Stable on the web in Chrome 148+; Chrome 138+ in extensions.',
    },
    {
      id: 'summarizer',
      label: 'Summarizer',
      description: 'Condense long files or conversations into key points. Also available by right-clicking a selection.',
      status: summarizer,
      note: 'Stable in Chrome 138+.',
    },
    {
      id: 'translator',
      label: 'Translator',
      description: 'Translate a selection on-device. Availability is per language pair; also in the editor context menu.',
      status: translator,
      note: 'Stable in Chrome 138+.',
    },
    {
      id: 'language-detector',
      label: 'Language Detector',
      description: 'Identify the language of a text, for example before translating it.',
      status: languageDetector,
      note: 'Stable in Chrome 138+.',
    },
    {
      id: 'writer',
      label: 'Writer',
      description: 'Draft new text such as commit messages, docstrings, or release notes.',
      status: writer,
      note: 'Developer trial.',
    },
    {
      id: 'rewriter',
      label: 'Rewriter',
      description: 'Shorten, expand, or change the tone of selected text.',
      status: rewriter,
      note: 'Developer trial.',
    },
    {
      id: 'proofreader',
      label: 'Proofreader',
      description: 'Grammar and spelling corrections with explanations for docs and comments.',
      status: proofreader,
      note: 'Origin trial.',
    },
  ]
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
        throw new Error('Browser AI is not supported in this browser. On the web the Prompt API is stable in Chrome 148+ on desktop (Chrome 138+ in extensions); identical APIs are not available in open-source Chromium. Use WebLLM or Transformers.js instead.')
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
      // webLLM() returns a lazy model: the engine (download + WebGPU init) is
      // only created on the first generate call. Force it here so "Load model"
      // actually finishes the load and surfaces progress. With no callback,
      // createSessionWithProgress falls back to initProgressCallback above,
      // which keeps the engine's descriptive progress text.
      await model.createSessionWithProgress()
      return model
    }
    case PROVIDERS.TRANSFORMERS_JS: {
      const model = transformersJS(modelId, {
        initProgressCallback: (p) => {
          onProgress?.({ text: `Loading ${modelId}...`, progress: typeof p === 'number' ? p : 0 })
        },
      })
      // Same lazy behavior: transformersJS() only constructs the wrapper, the
      // tokenizer/weights load on first use. Do it during Load model.
      await model.createSessionWithProgress((p) => {
        onProgress?.({ text: `Loading ${modelId}...`, progress: typeof p === 'number' ? p : 0 })
      })
      return model
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

  const hasTools = !!tools && Object.keys(tools).length > 0

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
    // Without this the SDK stops after the first tool call and never produces
    // the follow-up answer that consumes the tool result.
    stopWhen: hasTools ? stepCountIs(8) : undefined,
    allowSystemInMessages: true,
    abortSignal,
  })
}
