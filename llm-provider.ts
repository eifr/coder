import { streamText, type ModelMessage, type LanguageModel } from 'ai'
import { browserAI, doesBrowserSupportBrowserAI } from '@browser-ai/core'
import { webLLM, doesBrowserSupportWebLLM } from '@browser-ai/web-llm'
import { transformersJS } from '@browser-ai/transformers-js'
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

const WEBLLM_MODELS = [
  { id: 'Llama-3.2-1B-Instruct-q4f16_1-MLC', vram: 1024, preferred: 1 },
  { id: 'Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC', vram: 2048, preferred: 2 },
  { id: 'Llama-3.2-3B-Instruct-q4f16_1-MLC', vram: 3072, preferred: 3 },
  { id: 'Qwen3-4B-q4f16_1-MLC', vram: 4096, preferred: 4 },
  { id: 'Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC', vram: 8192, preferred: 5 },
  { id: 'Qwen3-8B-q4f16_1-MLC', vram: 8192, preferred: 6 },
  { id: 'Qwen2-7B-Instruct-q4f16_1-MLC', vram: 6144 },
  { id: 'Phi-3-mini-4k-instruct-q4f16_1-MLC', vram: 3072 },
]

const TRANSFORMERS_MODELS = [
  { id: 'Xenova/tiny-random-LlamaForCausalLM', vram: 128, label: 'Tiny Test Model' },
  { id: 'Xenova/TinyLlama-1.1B-Chat-v1.0', vram: 1024 },
  { id: 'HuggingFaceTB/SmolLM2-360M-Instruct', vram: 512 },
  { id: 'HuggingFaceTB/SmolLM2-1.7B-Instruct', vram: 2048 },
  { id: 'microsoft/Phi-3-mini-4k-instruct', vram: 4096 },
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
      return !!(navigator as any).gpu || !!(navigator as any).ml
    default:
      return false
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
      const model = browserAI()
      try {
        await model.createSessionWithProgress((p) => {
          onProgress?.({ text: `Downloading Chrome AI model...`, progress: p })
        })
      } catch (e) {
        console.warn('Session creation failed, will retry on first message:', e)
      }
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
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path relative to project root' },
        },
        required: ['path'],
      },
      execute: async ({ path }: { path: string }) => {
        const content = await readFileFn(path)
        return content ? { path, content } : { error: `File not found: ${path}` }
      },
    },
    editFile: {
      description: 'Write content to an existing file. Overwrites the entire file.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path relative to project root' },
          content: { type: 'string', description: 'Full file content to write' },
        },
        required: ['path', 'content'],
      },
      execute: async ({ path, content }: { path: string; content: string }) => {
        const ok = await editFileFn(path, content)
        return ok ? { path, success: true } : { error: `Failed to write ${path}` }
      },
    },
    createFile: {
      description: 'Create a new file with the given content. Fails if the file already exists.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path relative to project root' },
          content: { type: 'string', description: 'File content' },
        },
        required: ['path', 'content'],
      },
      execute: async ({ path, content }: { path: string; content: string }) => {
        const ok = await createFileFn(path, content)
        return ok ? { path, success: true } : { error: `Failed to create ${path} (may already exist)` }
      },
    },
    deleteFile: {
      description: 'Delete a file from the project',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path relative to project root' },
        },
        required: ['path'],
      },
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
