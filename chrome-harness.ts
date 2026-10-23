/**
 * Coding harness on top of the Chrome/Edge Prompt API (`LanguageModel`).
 *
 * The Prompt API supports tool use in open-loop mode: the model emits
 * `LanguageModelToolCall`s, the app executes them and sends
 * `LanguageModelToolSuccess`/`LanguageModelToolError` results back in a
 * follow-up prompt. We drive that loop here so the on-device model can inspect
 * and edit the user's project like the larger WebLLM/Transformers models.
 *
 * Tool use is still experimental in Chrome. When a session refuses to accept
 * `tools`/tool modalities we fall back to a plain text session (and tell the
 * caller) instead of failing the chat turn.
 *
 * Spec: https://github.com/webmachinelearning/prompt-api#tool-use
 */

export interface HarnessTool {
  name: string
  description: string
  inputSchema: {
    type: 'object'
    properties: Record<string, unknown>
    required?: string[]
  }
  execute: (args: Record<string, unknown>) => Promise<unknown> | unknown
}

export type ChromeHarnessEvent =
  | { type: 'text'; text: string; delta: string }
  | { type: 'tool-call'; name: string; args: Record<string, unknown> }
  | { type: 'tool-result'; name: string; ok: boolean; preview: string }
  | { type: 'fallback'; reason: string }
  | { type: 'done'; text: string; toolCalls: number }

export interface ChromeHarnessOptions {
  system: string
  /** System prompt used if native tool use is unavailable. */
  plainSystem: string
  messages: Array<{ role: string; content: unknown }>
  tools?: HarnessTool[]
  signal?: AbortSignal
}

interface NativeMessage {
  role: 'system' | 'user' | 'assistant'
  content: unknown
}

interface NativeToolCall {
  callId: string
  name: string
  arguments?: Record<string, unknown>
}

interface NativeContent {
  type: string
  value?: unknown
}

interface PromptSession {
  promptStreaming(input: unknown, options?: { signal: AbortSignal }): AsyncIterable<unknown>
  destroy?: () => void
}

interface LanguageModelGlobal {
  create(options: Record<string, unknown>): Promise<PromptSession>
}

type ToolResponseConstructor = new (init: Record<string, unknown>) => unknown

const MAX_TOOL_STEPS = 8
const MAX_RESULT_CHARS = 24000

function truncate(value: string, max = MAX_RESULT_CHARS): string {
  if (value.length <= max) return value
  return `${value.slice(0, max)}\n…[truncated ${value.length - max} characters]`
}

function getLanguageModel(): LanguageModelGlobal | undefined {
  const candidate = (globalThis as Record<string, unknown>).LanguageModel
  if (!candidate || typeof candidate !== 'object' || !('create' in candidate)) return undefined
  if (typeof candidate.create !== 'function') return undefined
  // Reason: the Prompt API has no TypeScript lib here; the shape was checked
  // above and every use goes through this interface.
  const model = candidate as LanguageModelGlobal
  return model
}

function isNativeContent(chunk: unknown): chunk is NativeContent {
  if (!chunk || typeof chunk !== 'object' || !('type' in chunk)) return false
  return typeof chunk.type === 'string'
}

function isToolCall(value: unknown): value is NativeToolCall {
  if (!value || typeof value !== 'object') return false
  if (!('callId' in value) || typeof value.callId !== 'string') return false
  if (!('name' in value) || typeof value.name !== 'string') return false
  if ('arguments' in value && value.arguments !== undefined) {
    if (!value.arguments || typeof value.arguments !== 'object') return false
  }
  return true
}

/** Prompt API messages accept plain strings; the app only sends text content. */
function toNativeMessages(messages: ChromeHarnessOptions['messages']): NativeMessage[] {
  const out: NativeMessage[] = []
  for (const message of messages) {
    if (message.role !== 'user' && message.role !== 'assistant') continue
    const content = typeof message.content === 'string'
      ? message.content
      : Array.isArray(message.content)
        ? message.content
            .filter((part): part is { type: 'text'; text: string } => {
              if (!part || typeof part !== 'object') return false
              return 'type' in part && part.type === 'text' && 'text' in part && typeof part.text === 'string'
            })
            .map(part => part.text)
            .join('')
        : String(message.content ?? '')
    out.push({ role: message.role, content })
  }
  return out
}

function createSessionOptions(
  system: string,
  tools: HarnessTool[],
  signal?: AbortSignal
): Record<string, unknown> {
  const options: Record<string, unknown> = {
    initialPrompts: system ? [{ role: 'system', content: system }] : [],
  }
  if (signal) options.signal = signal
  if (tools.length > 0) {
    options.expectedInputs = [{ type: 'text' }, { type: 'tool-response' }]
    options.expectedOutputs = [{ type: 'text' }, { type: 'tool-call' }]
    options.tools = tools.map(tool => ({
      name: tool.name,
      description: tool.description,
      inputSchema: tool.inputSchema,
    }))
  }
  return options
}

function makeToolResponse(call: NativeToolCall, ok: boolean, text: string): unknown {
  const globals = globalThis as Record<string, unknown>
  const Success = globals.LanguageModelToolSuccess as ToolResponseConstructor | undefined
  const Failure = globals.LanguageModelToolError as ToolResponseConstructor | undefined
  if (ok) {
    const init = { callId: call.callId, name: call.name, result: [{ type: 'text', value: text }] }
    return Success ? new Success(init) : init
  }
  const init = { callId: call.callId, name: call.name, errorMessage: text }
  return Failure ? new Failure(init) : init
}

/**
 * Runs one assistant turn. Yields incremental text plus tool activity and
 * resolves the open-loop tool calls against `tools` until the model answers
 * without requesting more tools (or the step budget is exhausted).
 */
export async function* streamChromeHarness(options: ChromeHarnessOptions): AsyncGenerator<ChromeHarnessEvent> {
  const LanguageModel = getLanguageModel()
  if (!LanguageModel) {
    throw new Error('Chrome Built-in AI (Prompt API) is not available in this browser.')
  }

  let system = options.system
  let tools = options.tools || []
  let session: PromptSession

  if (tools.length > 0) {
    try {
      session = await LanguageModel.create(createSessionOptions(system, tools, options.signal))
    } catch (err) {
      // Older Chrome builds expose the Prompt API but not tool use. Retry as a
      // plain text session so the chat still works (without project access).
      const reason = err instanceof Error ? err.message : String(err)
      tools = []
      system = options.plainSystem
      yield { type: 'fallback', reason }
      session = await LanguageModel.create(createSessionOptions(system, tools, options.signal))
    }
  } else {
    session = await LanguageModel.create(createSessionOptions(system, tools, options.signal))
  }

  const messages = toNativeMessages(options.messages)
  let input: unknown = messages.length > 0 ? messages : ''
  const textParts: string[] = []
  let stepText = ''
  let totalToolCalls = 0

  try {
    for (let step = 0; step <= MAX_TOOL_STEPS; step++) {
      if (options.signal?.aborted) break

      const toolCalls: NativeToolCall[] = []
      try {
        const stream = session.promptStreaming(input, options.signal ? { signal: options.signal } : undefined)
        for await (const chunk of stream) {
          if (options.signal?.aborted) break
          if (typeof chunk === 'string') {
            stepText += chunk
            yield { type: 'text', text: [...textParts, stepText].filter(part => part !== '').join('\n\n'), delta: chunk }
          } else if (isNativeContent(chunk) && chunk.type === 'text' && typeof chunk.value === 'string') {
            stepText += chunk.value
            yield { type: 'text', text: [...textParts, stepText].filter(part => part !== '').join('\n\n'), delta: chunk.value }
          } else if (isNativeContent(chunk) && chunk.type === 'tool-call' && isToolCall(chunk.value)) {
            toolCalls.push(chunk.value)
          }
          // Other content types (e.g. "thought") are intentionally ignored.
        }
      } catch (err) {
        if (options.signal?.aborted || (err instanceof Error && err.name === 'AbortError')) break
        throw err
      }

      if (toolCalls.length === 0) break
      if (stepText !== '') {
        textParts.push(stepText)
        stepText = ''
      }

      const responses: unknown[] = []
      for (const call of toolCalls) {
        totalToolCalls++
        const args = call.arguments || {}
        yield { type: 'tool-call', name: call.name, args }
        const tool = tools.find(candidate => candidate.name === call.name)
        let result: { ok: boolean; text: string }
        if (!tool) {
          result = { ok: false, text: `Unknown tool: ${call.name}` }
        } else {
          try {
            const value = await tool.execute(args)
            const serialized = typeof value === 'string' ? value : JSON.stringify(value, null, 2) ?? 'null'
            result = { ok: true, text: truncate(serialized) }
          } catch (err) {
            result = { ok: false, text: err instanceof Error ? err.message : String(err) }
          }
        }
        yield { type: 'tool-result', name: call.name, ok: result.ok, preview: truncate(result.text, 300) }
        responses.push({ type: 'tool-response', value: makeToolResponse(call, result.ok, result.text) })
      }

      input = [{ role: 'user', content: responses }]
    }
  } finally {
    try {
      session.destroy?.()
    } catch {}
  }

  const visibleText = [...textParts, stepText].filter(part => part !== '').join('\n\n').replace(/<think>[\s\S]*?<\/think>/g, '').trim()
  const finalText = visibleText || (totalToolCalls > 0
    ? `Completed ${totalToolCalls} tool call${totalToolCalls === 1 ? '' : 's'} but the model returned no summary.`
    : '')
  yield { type: 'done', text: finalText, toolCalls: totalToolCalls }
}

// ---------------------------------------------------------------------------
// Project tools
// ---------------------------------------------------------------------------

export interface HarnessFs {
  /** Project file paths relative to the opened root. */
  listFiles: () => Promise<string[]> | string[]
  readFile: (path: string) => Promise<string | null>
  /** Create or overwrite a file (parent directories are created). */
  writeFile: (path: string, content: string) => Promise<boolean>
  /** Create a file, failing if it already exists. */
  createFile: (path: string, content: string) => Promise<boolean>
  deleteFile: (path: string) => Promise<boolean>
}

const MAX_READ_LINES = 600
const MAX_READ_CHARS = 16000
const MAX_LISTED_FILES = 500
const MAX_SEARCH_FILES = 400
const MAX_SEARCH_BYTES = 2_000_000
const MAX_SEARCH_MATCHES = 40

function normalizePath(path: unknown): string {
  return String(path ?? '').replace(/\\/g, '/').replace(/^\.?\//, '').replace(/\/+$/, '')
}

export function createHarnessTools(fs: HarnessFs): HarnessTool[] {
  const filesUnder = async (prefix: unknown) => {
    const base = normalizePath(prefix)
    const all = (await fs.listFiles()).slice().sort()
    if (!base) return all
    return all.filter(path => path === base || path.startsWith(`${base}/`))
  }

  return [
    {
      name: 'listFiles',
      description: 'List project files as paths relative to the project root. Use it to discover files. Optionally filter by directory and/or a filename substring.',
      inputSchema: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Directory to list, e.g. "src". Omit for the whole project.' },
          query: { type: 'string', description: 'Case-insensitive substring that must appear in the file path.' },
        },
        required: [],
      },
      execute: async ({ path, query }) => {
        const needle = String(query ?? '').toLowerCase()
        const matches = (await filesUnder(path)).filter(file => !needle || file.toLowerCase().includes(needle))
        return {
          count: matches.length,
          files: matches.slice(0, MAX_LISTED_FILES),
          truncated: matches.length > MAX_LISTED_FILES,
        }
      },
    },
    {
      name: 'searchFiles',
      description: 'Search project file contents for a case-insensitive substring. Returns file paths, 1-based line numbers and matching lines. Use it to find symbols, imports or usages.',
      inputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Text to search for.' },
          path: { type: 'string', description: 'Optional directory limit, e.g. "src".' },
        },
        required: ['query'],
      },
      execute: async ({ query, path }) => {
        const needle = String(query ?? '').toLowerCase()
        if (!needle) return { error: 'query is required' }
        const matches: Array<{ path: string; line: number; text: string }> = []
        let scanned = 0
        let scannedFiles = 0
        let truncated = false
        for (const file of await filesUnder(path)) {
          if (scannedFiles >= MAX_SEARCH_FILES || scanned >= MAX_SEARCH_BYTES) {
            truncated = true
            break
          }
          const content = await fs.readFile(file)
          if (content == null || content.includes('\u0000')) continue
          scannedFiles++
          scanned += content.length
          const lines = content.split('\n')
          for (let i = 0; i < lines.length; i++) {
            if (!lines[i].toLowerCase().includes(needle)) continue
            matches.push({ path: file, line: i + 1, text: truncate(lines[i].trim(), 300) })
            if (matches.length >= MAX_SEARCH_MATCHES) {
              truncated = true
              break
            }
          }
          if (matches.length >= MAX_SEARCH_MATCHES) break
        }
        return { query, count: matches.length, matches, truncated }
      },
    },
    {
      name: 'readFile',
      description: 'Read a project file. Always read before editing. Optionally pass a 1-based line range for large files (max 600 lines / 24000 characters per call).',
      inputSchema: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path relative to the project root.' },
          startLine: { type: 'number', description: 'First line to return (1-based).' },
          endLine: { type: 'number', description: 'Last line to return (inclusive).' },
        },
        required: ['path'],
      },
      execute: async ({ path, startLine, endLine }) => {
        const file = normalizePath(path)
        const content = await fs.readFile(file)
        if (content == null) return { error: `File not found: ${file}` }
        const lines = content.split('\n')
        const totalLines = lines.length
        let start = typeof startLine === 'number' && Number.isFinite(startLine) ? Math.max(1, Math.floor(startLine)) : 1
        let end = typeof endLine === 'number' && Number.isFinite(endLine) ? Math.floor(endLine) : start + MAX_READ_LINES - 1
        start = Math.min(start, Math.max(1, totalLines))
        end = Math.max(start, Math.min(end, start + MAX_READ_LINES - 1, totalLines))
        let body = lines.slice(start - 1, end).join('\n')
        let truncated = end < totalLines
        if (body.length > MAX_READ_CHARS) {
          body = body.slice(0, MAX_READ_CHARS)
          truncated = true
        }
        return { path: file, startLine: start, endLine: end, totalLines, content: body, truncated }
      },
    },
    {
      name: 'createFile',
      description: 'Create a new file. Fails if the file already exists. Parent directories are created automatically.',
      inputSchema: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path relative to the project root.' },
          content: { type: 'string', description: 'Full file content.' },
        },
        required: ['path', 'content'],
      },
      execute: async ({ path, content }) => {
        const file = normalizePath(path)
        if (!file) return { error: 'path is required' }
        const ok = await fs.createFile(file, String(content ?? ''))
        return ok ? { path: file, created: true } : { error: `Failed to create ${file} (it may already exist)` }
      },
    },
    {
      name: 'editFile',
      description: 'Replace the entire contents of an existing file. Read the file first. Creates the file (and parent directories) if it does not exist.',
      inputSchema: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path relative to the project root.' },
          content: { type: 'string', description: 'Full new file content.' },
        },
        required: ['path', 'content'],
      },
      execute: async ({ path, content }) => {
        const file = normalizePath(path)
        if (!file) return { error: 'path is required' }
        const ok = await fs.writeFile(file, String(content ?? ''))
        return ok ? { path: file, written: true } : { error: `Failed to write ${file}` }
      },
    },
    {
      name: 'deleteFile',
      description: 'Delete a file from the project.',
      inputSchema: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path relative to the project root.' },
        },
        required: ['path'],
      },
      execute: async ({ path }) => {
        const file = normalizePath(path)
        if (!file) return { error: 'path is required' }
        const ok = await fs.deleteFile(file)
        return ok ? { path: file, deleted: true } : { error: `Failed to delete ${file}` }
      },
    },
  ]
}
