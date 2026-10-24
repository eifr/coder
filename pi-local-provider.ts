import type { LanguageModel, ModelMessage } from 'ai'
import {
  createAssistantMessageEventStream,
  createModels,
  createProvider,
  getCurrentSystemPrompt,
  getCurrentTools,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Model,
  type ProviderStreams,
  type SimpleStreamOptions,
  type TranscriptContext,
} from '@earendil-works/pi-ai'
import { createChatStream, type Provider } from './llm-provider'
import {
  buildToolPrompt,
  consumeStructuredStream,
  countToolCalls,
  currentTurnMessages,
  hasRepeatedToolCall,
  isContextOverflow,
  toPlainMessage,
  zeroUsage,
  type TranscriptMessage,
} from './pi-chrome-provider'

const LOCAL_API = 'local-agent'
const LOCAL_PROVIDER = 'local'
const DEFAULT_CONTEXT_WINDOW = 8192
const MAX_AGENT_TOOL_CALLS = 6

export interface LocalModelOptions {
  /** The app provider that produced the AI SDK model (webllm, transformers-js). */
  provider: Provider
  /** The AI SDK model returned by initProviderModel. */
  model: LanguageModel
  /** Used for display and model references. */
  modelId: string
  contextWindow?: number
}

function streamLocal(
  model: Model<typeof LOCAL_API>,
  context: TranscriptContext,
  options: SimpleStreamOptions | undefined,
  source: LocalModelOptions
): AssistantMessageEventStream {
  const out = createAssistantMessageEventStream()
  const message: AssistantMessage = {
    role: 'assistant',
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: zeroUsage(),
    stopReason: 'pending',
    timestamp: Date.now(),
  }
  const signal = options?.signal

  void (async () => {
    try {
      const system = getCurrentSystemPrompt(context.messages)
      const tools = getCurrentTools(context.messages)
      const all = context.messages as TranscriptMessage[]
      const last = all[all.length - 1]
      if (!last) throw new Error('Local provider received an empty transcript.')
      const history = all.slice(0, -1).filter(item => item.role !== 'system')
      const turn = currentTurnMessages(all)
      const repeated = hasRepeatedToolCall(turn)
      const toolCallCount = countToolCalls(turn)
      const forceText = tools.length > 0 && last.role === 'toolResult' && (repeated || toolCallCount >= MAX_AGENT_TOOL_CALLS)
      if (forceText) {
        console.warn(`[pi-local] forcing a text-only answer (${toolCallCount} tool calls, repeated=${repeated}).`)
      }

      const prompt = tools.length > 0 ? buildToolPrompt(system, tools, forceText) : system
      const messages: ModelMessage[] = [...history, last].map(toPlainMessage)
      const result = createChatStream({
        provider: source.provider,
        model: source.model,
        messages,
        system: prompt,
        abortSignal: signal,
      })

      out.push({ type: 'start', partial: { ...message } })

      const chunks = (async function* () {
        for await (const part of result.fullStream) {
          if (part.type === 'text-delta') {
            yield (part as any).text ?? (part as any).delta ?? ''
          } else if (part.type === 'error') {
            const raw = (part as any).error ?? (part as any).errorText ?? (part as any).message
            throw new Error(raw instanceof Error ? raw.message : typeof raw === 'string' ? raw : JSON.stringify(raw ?? part))
          }
        }
      })()

      await consumeStructuredStream(chunks, tools, message, out, signal, forceText)

      if (signal?.aborted) {
        message.stopReason = 'aborted'
        message.errorMessage = 'Request aborted.'
        out.push({ type: 'error', reason: 'aborted', error: message })
        out.end(message)
        return
      }

      message.usage = zeroUsage()
      message.stopReason = message.content.some(block => block.type === 'toolCall') ? 'toolUse' : 'stop'
      out.push({ type: 'done', reason: message.stopReason, message })
      out.end(message)
    } catch (error) {
      const aborted = signal?.aborted || (error instanceof Error && error.name === 'AbortError')
      message.stopReason = aborted ? 'aborted' : 'error'
      const reason = error instanceof Error ? error.message : String(error)
      message.errorMessage = isContextOverflow(error) ? `context_length_exceeded: ${reason}` : reason
      out.push({ type: 'error', reason: message.stopReason, error: message })
      out.end(message)
    }
  })()

  return out
}

export function createLocalModels(options: LocalModelOptions) {
  const model: Model<typeof LOCAL_API> = {
    id: options.modelId,
    name: options.modelId,
    api: LOCAL_API,
    provider: LOCAL_PROVIDER,
    baseUrl: 'local://',
    input: ['text'],
    reasoning: false,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: options.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: 4096,
  }

  const streams: ProviderStreams = {
    stream: (streamModel, context, streamOptions) => streamLocal(
      streamModel as Model<typeof LOCAL_API>,
      context,
      streamOptions as SimpleStreamOptions,
      options
    ),
    streamSimple: (streamModel, context, streamOptions) => streamLocal(
      streamModel as Model<typeof LOCAL_API>,
      context,
      streamOptions,
      options
    ),
  }

  const models = createModels()
  models.setProvider(createProvider<typeof LOCAL_API>({
    id: LOCAL_PROVIDER,
    name: 'Local model',
    auth: {
      apiKey: {
        name: 'Local model',
        resolve: async () => ({ auth: {} }),
      },
    },
    models: [model],
    api: streams,
  }))
  return models
}

export { LOCAL_API, LOCAL_PROVIDER }
