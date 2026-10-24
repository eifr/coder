import {
  createAssistantMessageEventStream,
  createModels,
  createProvider,
  getCurrentSystemPrompt,
  getCurrentTools,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type JsonObject,
  type Model,
  type ProviderStreams,
  type SimpleStreamOptions,
  type Tool,
  type TranscriptContext,
  type Usage,
} from '@earendil-works/pi-ai'

const CHROME_API = 'chrome-prompt-api'
const PROVIDER_ID = 'chrome-ai'
const MODEL_ID = 'gemini-nano'
const DEFAULT_CONTEXT_WINDOW = 9216

interface PromptStreamOptions {
  signal?: AbortSignal
  responseConstraint?: unknown
}

interface PromptSession {
  readonly contextUsage?: number
  readonly contextWindow?: number
  promptStreaming(input: unknown, options?: PromptStreamOptions): AsyncIterable<unknown>
  destroy?: () => void
}

interface LanguageModelGlobal {
  availability(options?: Record<string, unknown>): Promise<string>
  create(options: Record<string, unknown>): Promise<PromptSession>
}

export interface StructuredToolCall {
  tool: string
  arguments: JsonObject
}

export interface NativeToolCall {
  callId?: string
  callID?: string
  name: string
  arguments?: Record<string, unknown>
}

export type TranscriptMessage = TranscriptContext['messages'][number]

function getLanguageModel(): LanguageModelGlobal | undefined {
  const candidate = (globalThis as Record<string, unknown>).LanguageModel
  if (candidate === null || (typeof candidate !== 'object' && typeof candidate !== 'function')) return undefined
  if (!('create' in candidate) || typeof candidate.create !== 'function') return undefined
  if (!('availability' in candidate) || typeof candidate.availability !== 'function') return undefined
  return candidate as LanguageModelGlobal
}

export async function getChromeAvailability(): Promise<{
  available: boolean
  availability: string
}> {
  const languageModel = getLanguageModel()
  if (!languageModel) return { available: false, availability: 'missing' }
  try {
    const availability = await languageModel.availability()
    return { available: availability === 'available' || availability === 'downloading', availability }
  } catch (error) {
    return { available: false, availability: error instanceof Error ? error.message : String(error) }
  }
}

export function zeroUsage(): Usage {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  }
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content
    .map(part => {
      if (!part || typeof part !== 'object') return ''
      if ('type' in part && part.type === 'text' && 'text' in part && typeof part.text === 'string') return part.text
      return ''
    })
    .join('')
}

/** Chrome's Prompt API text chunk: a plain string in older builds, a content object in newer ones. */
export function chunkText(chunk: unknown): string | undefined {
  if (typeof chunk === 'string') return chunk
  if (!chunk || typeof chunk !== 'object' || !('type' in chunk) || chunk.type !== 'text') return undefined
  if ('value' in chunk && typeof chunk.value === 'string') return chunk.value
  if ('text' in chunk && typeof chunk.text === 'string') return chunk.text
  return undefined
}

export function isContextOverflow(error: unknown): boolean {
  if (!(error instanceof Error)) return false
  return error.name === 'QuotaExceededError' || /context|too large|quota/i.test(error.message)
}

/**
 * Native `tool-call` chunks are emitted by builds whose on-device model uses
 * the Prompt API tool protocol. Current Linux models emit markdown instead, so
 * the structured fallback below is the default path.
 */
export function nativeToolCall(chunk: unknown): NativeToolCall | undefined {
  if (!chunk || typeof chunk !== 'object' || !('type' in chunk) || chunk.type !== 'tool-call') return undefined
  if (!('value' in chunk) || !chunk.value || typeof chunk.value !== 'object') return undefined
  const value = chunk.value as Record<string, unknown>
  if (typeof value.name !== 'string') return undefined
  const callId = typeof value.callId === 'string' ? value.callId : typeof value.callID === 'string' ? value.callID : undefined
  if (!callId) return undefined
  const args = value.arguments && typeof value.arguments === 'object'
    ? value.arguments as Record<string, unknown>
    : {}
  return { callId, name: value.name, arguments: args }
}

/**
 * Chrome's on-device model does not emit native `tool-call` content in current
 * builds, but it does honor `responseConstraint`. Tool calls are therefore
 * encoded as constrained JSON and translated back into pi tool calls.
 */
export function toPlainMessage(message: TranscriptMessage): { role: 'user' | 'assistant'; content: string } {
  if (message.role === 'system') {
    throw new Error('System messages are sent through initialPrompts, not toPlainMessage().')
  }
  if (message.role === 'user') return { role: 'user', content: textOf(message.content) }
  if (message.role === 'assistant') {
    const text = message.content.map(block => {
      if (block.type === 'text') return block.text
      if (block.type === 'toolCall') return `[tool call ${block.name} ${JSON.stringify(block.arguments)}]`
      return ''
    }).filter(Boolean).join('\n')
    return { role: 'assistant', content: text }
  }
  return {
    role: 'user',
    content: `Tool "${message.toolName}" returned: ${textOf(message.content)}\nThe tool has already run. Do not call it again; answer the user's request with {"text":"..."}.`,
  }
}

export function buildToolPrompt(system: string, tools: Tool[], forceText = false): string {
  const declarations = tools.map(tool => [
    `- ${tool.name}: ${tool.description}`,
    `  arguments: ${JSON.stringify(tool.parameters)}`,
  ].join('\n')).join('\n')
  return [
    system,
    'You can call tools. Reply with exactly one JSON object and no markdown:',
    '{"tool":"<name>","arguments":{...}} to call a tool, or {"text":"<answer>"} to answer.',
    'Call a tool only when the request needs project files or an edit. For greetings, explanations, or questions you can answer directly, use {"text":"<answer>"} and do not call a tool.',
    'After a tool result, answer with {"text":"<answer>"} and do not call that tool again.',
    'Omit optional arguments you do not need; never send empty strings or the text "undefined".',
    forceText ? 'The previous tool call already ran. You must answer with {"text":"<answer>"} now; no more tools are available this turn.' : '',
    'Tools:',
    declarations,
  ].filter(Boolean).join('\n')
}

export function buildTextSchema(): Record<string, unknown> {
  return {
    type: 'object',
    properties: { text: { type: 'string' } },
    required: ['text'],
  }
}

/**
 * True when the model repeated the exact same tool call back to back. Current
 * on-device models can get stuck in that loop instead of answering.
 */
export function hasRepeatedToolCall(messages: TranscriptMessage[]): boolean {
  let previous: string | undefined
  for (const message of messages) {
    if (message.role !== 'assistant') continue
    for (const block of message.content) {
      if (block.type !== 'toolCall') continue
      const key = `${block.name}\u0000${JSON.stringify(block.arguments)}`
      if (key === previous) return true
      previous = key
    }
  }
  return false
}

export function countToolCalls(messages: TranscriptMessage[]): number {
  let count = 0
  for (const message of messages) {
    if (message.role !== 'assistant') continue
    for (const block of message.content) {
      if (block.type === 'toolCall') count++
    }
  }
  return count
}

export function buildToolSchema(tools: Tool[]): Record<string, unknown> {
  return {
    anyOf: [
      {
        type: 'object',
        properties: { text: { type: 'string' } },
        required: ['text'],
      },
      ...tools.map(tool => ({
        type: 'object',
        properties: {
          tool: { enum: [tool.name] },
          arguments: tool.parameters,
        },
        required: ['tool', 'arguments'],
      })),
    ],
  }
}

export function parseStructuredToolCall(raw: string, tools: Tool[]): StructuredToolCall | undefined {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return undefined
  }
  if (!parsed || typeof parsed !== 'object') return undefined
  if (!('tool' in parsed) || typeof parsed.tool !== 'string') return undefined
  if (!tools.some(tool => tool.name === parsed.tool)) return undefined
  if (!('arguments' in parsed) || !parsed.arguments || typeof parsed.arguments !== 'object') return undefined
  // responseConstraint validated the arguments against the tool's JSON Schema.
  return { tool: parsed.tool, arguments: parsed.arguments as JsonObject }
}

export function structuredText(raw: string): string | undefined {
  try {
    const parsed: unknown = JSON.parse(raw)
    if (parsed && typeof parsed === 'object' && 'text' in parsed && typeof parsed.text === 'string') {
      return parsed.text
    }
  } catch {
    // Fall through to the raw response.
  }
  return undefined
}

async function createSession(
  languageModel: LanguageModelGlobal,
  system: string,
  history: TranscriptMessage[],
  tools: Tool[],
  signal: AbortSignal | undefined,
  forceText: boolean
): Promise<PromptSession> {
  const initialPrompts: unknown[] = []
  const prompt = tools.length > 0 ? buildToolPrompt(system, tools, forceText) : system
  // The Prompt API requires the system prompt to be the first message and
  // allows exactly one. History follows it; the current turn is sent with
  // promptStreaming(), never as another system message.
  if (prompt) initialPrompts.push({ role: 'system', content: prompt })
  initialPrompts.push(...history.map(toPlainMessage))

  const options: Record<string, unknown> = {
    initialPrompts,
    expectedInputs: [{ type: 'text' }],
    expectedOutputs: [{ type: 'text' }],
  }
  if (signal) options.signal = signal
  return languageModel.create(options)
}

export function newToolCallId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `tool-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

/**
 * Consumes text/native-tool chunks from any provider and emits pi events.
 * Shared by the Prompt API and the AI SDK (WebLLM/Transformers) adapters.
 */
export async function consumeStructuredStream(
  source: AsyncIterable<unknown>,
  tools: Tool[],
  message: AssistantMessage,
  out: AssistantMessageEventStream,
  signal?: AbortSignal,
  forceText = false
): Promise<void> {
  const structured = tools.length > 0
  let raw = ''
  let textIndex = -1
  let text = ''
  const nativeCalls: NativeToolCall[] = []
  let chunkCount = 0
  const sample: string[] = []

  for await (const chunk of source) {
    chunkCount++
    if (sample.length < 5) {
      if (typeof chunk === 'string') sample.push(`string:${chunk.slice(0, 80)}`)
      else if (chunk && typeof chunk === 'object') {
        const type = 'type' in chunk ? String(chunk.type) : ''
        sample.push(`object:${Object.keys(chunk).join(',')}${type ? ` type=${type}` : ''}`)
      } else sample.push(String(chunk))
    }
    if (signal?.aborted) break
    const native = nativeToolCall(chunk)
    if (native) {
      nativeCalls.push(native)
      continue
    }
    const delta = chunkText(chunk)
    if (delta === undefined) continue
    if (structured) {
      raw += delta
      continue
    }
    if (textIndex < 0) {
      textIndex = message.content.length
      message.content.push({ type: 'text', text: '' })
      out.push({ type: 'text_start', contentIndex: textIndex, partial: { ...message } })
    }
    text += delta
    const block = message.content[textIndex]
    if (block.type === 'text') block.text = text
    out.push({ type: 'text_delta', contentIndex: textIndex, delta, partial: { ...message } })
  }

  if (signal?.aborted) return

  if (nativeCalls.length > 0) {
    for (const native of nativeCalls) {
      const index = message.content.length
      // Native providers validate tool arguments against inputSchema.
      const args = (native.arguments ?? {}) as JsonObject
      const call = {
        type: 'toolCall' as const,
        id: native.callId ?? newToolCallId(),
        name: native.name,
        arguments: args,
      }
      message.content.push(call)
      out.push({ type: 'toolcall_start', contentIndex: index, partial: { ...message } })
      out.push({
        type: 'toolcall_delta',
        contentIndex: index,
        delta: JSON.stringify(call.arguments),
        partial: { ...message },
      })
      out.push({ type: 'toolcall_end', contentIndex: index, toolCall: call, partial: { ...message } })
    }
    return
  }

  if (structured) {
    const toolCall = forceText ? undefined : parseStructuredToolCall(raw, tools)
    if (toolCall) {
      const index = message.content.length
      const call = {
        type: 'toolCall' as const,
        id: newToolCallId(),
        name: toolCall.tool,
        arguments: toolCall.arguments,
      }
      message.content.push(call)
      out.push({ type: 'toolcall_start', contentIndex: index, partial: { ...message } })
      out.push({
        type: 'toolcall_delta',
        contentIndex: index,
        delta: JSON.stringify(call.arguments),
        partial: { ...message },
      })
      out.push({ type: 'toolcall_end', contentIndex: index, toolCall: call, partial: { ...message } })
      return
    }
    const answer = structuredText(raw) ?? raw
    if (answer) {
      textIndex = message.content.length
      message.content.push({ type: 'text', text: answer })
      out.push({ type: 'text_start', contentIndex: textIndex, partial: { ...message } })
      out.push({ type: 'text_delta', contentIndex: textIndex, delta: answer, partial: { ...message } })
      out.push({ type: 'text_end', contentIndex: textIndex, content: answer, partial: { ...message } })
    } else {
      console.warn('[pi-structured] model returned an empty structured response.', raw)
    }
    return
  }

  if (textIndex >= 0) {
    out.push({ type: 'text_end', contentIndex: textIndex, content: text, partial: { ...message } })
  }

  if (message.content.length === 0) {
    console.warn('[pi-structured] model produced no content', { chunkCount, sample, rawLength: raw.length })
  }
}

function streamSimple(
  model: Model<typeof CHROME_API>,
  context: TranscriptContext,
  options?: SimpleStreamOptions
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream()
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
    const languageModel = getLanguageModel()
    if (!languageModel) {
      message.stopReason = 'error'
      message.errorMessage = 'Chrome Built-in AI (Prompt API) is not available in this browser.'
      stream.push({ type: 'error', reason: 'error', error: message })
      stream.end(message)
      return
    }

    let session: PromptSession | undefined
    try {
      const system = getCurrentSystemPrompt(context.messages)
      const tools = getCurrentTools(context.messages)
      const all = context.messages as TranscriptMessage[]
      const last = all[all.length - 1]
      if (!last) throw new Error('Chrome provider received an empty transcript.')
      const history = all.slice(0, -1).filter(item => item.role !== 'system')
      const repeated = hasRepeatedToolCall(context.messages)
      const toolCallCount = countToolCalls(context.messages)
      // pi's agent stops at 8 turns; force an answer before that so a model
      // that keeps calling tools still produces a summary.
      const forceText = tools.length > 0 && last.role === 'toolResult' && (repeated || toolCallCount >= 6)
      if (forceText) {
        console.warn(`[pi-chrome] forcing a text-only answer (${toolCallCount} tool calls, repeated=${repeated}).`)
      }

      session = await createSession(languageModel, system, history, tools, signal, forceText)
      const inputTokens = session.contextUsage ?? 0
      const structured = tools.length > 0
      const promptInput = [toPlainMessage(last)]
      const promptOptions: PromptStreamOptions = {}
      if (signal) promptOptions.signal = signal
      if (structured) {
        promptOptions.responseConstraint = forceText ? buildTextSchema() : buildToolSchema(tools)
      }

      stream.push({ type: 'start', partial: { ...message } })

      await consumeStructuredStream(session.promptStreaming(promptInput, promptOptions), tools, message, stream, signal, forceText)

      // Some Chrome builds ignore responseConstraint and return no text at all.
      // Retry once as a plain text session so the user still gets an answer.
      if (structured && message.content.length === 0 && !signal?.aborted) {
        console.warn('[pi-chrome] structured response was empty; retrying as plain text.')
        session.destroy?.()
        session = await createSession(languageModel, system, history, [], signal, false)
        const retryInput = session.contextUsage ?? 0
        await consumeStructuredStream(
          session.promptStreaming(promptInput, signal ? { signal } : undefined),
          [],
          message,
          stream,
          signal,
          false
        )
        const afterRetry = session.contextUsage ?? retryInput
        const outputTokens = Math.max(0, afterRetry - retryInput)
        message.usage = {
          ...zeroUsage(),
          input: retryInput,
          output: outputTokens,
          totalTokens: retryInput + outputTokens,
        }
      } else {
        const after = session.contextUsage ?? inputTokens
        const outputTokens = Math.max(0, after - inputTokens)
        message.usage = {
          ...zeroUsage(),
          input: inputTokens,
          output: outputTokens,
          totalTokens: inputTokens + outputTokens,
        }
      }
      message.stopReason = message.content.some(block => block.type === 'toolCall') ? 'toolUse' : 'stop'
      stream.push({ type: 'done', reason: message.stopReason, message })
      stream.end(message)
    } catch (error) {
      const aborted = signal?.aborted || (error instanceof Error && error.name === 'AbortError')
      message.stopReason = aborted ? 'aborted' : 'error'
      const reason = error instanceof Error ? error.message : String(error)
      message.errorMessage = isContextOverflow(error) ? `context_length_exceeded: ${reason}` : reason
      stream.push({ type: 'error', reason: message.stopReason, error: message })
      stream.end(message)
    } finally {
      session?.destroy?.()
    }
  })()

  return stream
}

const chromeStreams: ProviderStreams = {
  stream: (model, context, options) => streamSimple(
    model as Model<typeof CHROME_API>,
    context,
    options as SimpleStreamOptions
  ),
  streamSimple,
}

export function createChromeModels(contextWindow = DEFAULT_CONTEXT_WINDOW) {
  const model: Model<typeof CHROME_API> = {
    id: MODEL_ID,
    name: 'Chrome Gemini Nano',
    api: CHROME_API,
    provider: PROVIDER_ID,
    baseUrl: 'chrome://built-in-ai',
    input: ['text'],
    reasoning: false,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow,
    maxTokens: 2048,
  }

  const models = createModels()
  models.setProvider(createProvider<typeof CHROME_API>({
    id: PROVIDER_ID,
    name: 'Chrome Built-in AI',
    auth: {
      apiKey: {
        name: 'Chrome Built-in AI',
        resolve: async () => ({ auth: {} }),
      },
    },
    models: [model],
    api: chromeStreams,
  }))
  return models
}

export { CHROME_API, MODEL_ID, PROVIDER_ID }
