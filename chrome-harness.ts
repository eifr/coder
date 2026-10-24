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
  | { type: 'status'; text: string }
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
  // Chrome is renaming `callID` to `callId`; supported builds send one of them.
  callId?: string
  callID?: string
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

interface DownloadProgressEvent {
  loaded: number
  total?: number
}

interface DownloadMonitor {
  addEventListener(type: 'downloadprogress', listener: (event: DownloadProgressEvent) => void): void
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
  // `LanguageModel` is an interface object in Chrome (`typeof === 'function'`),
  // not a plain object; older polyfills/mocks may expose an object. `typeof` +
  // `in` both narrow `unknown` without an unchecked assertion.
  if (candidate === null || (typeof candidate !== 'object' && typeof candidate !== 'function')) return undefined
  if (!('create' in candidate)) return undefined
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
  if (!('name' in value) || typeof value.name !== 'string') return false
  const hasCallId = 'callId' in value && typeof value.callId === 'string'
  const hasLegacyCallId = 'callID' in value && typeof value.callID === 'string'
  if (!hasCallId && !hasLegacyCallId) return false
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

interface StatusChannel {
  push(text: string): void
  close(): void
  next(): Promise<string | null>
}

/**
 * Bridges `monitor` callbacks (fired while `LanguageModel.create` is pending)
 * to an async generator, so callers can render a loading state during the
 * wait — including the multi-gigabyte on-device model download.
 */
function createStatusChannel(): StatusChannel {
  const buffered: string[] = []
  let wake: ((value: string | null) => void) | null = null
  let closed = false
  return {
    push(text) {
      if (closed) return
      if (wake) {
        const resolve = wake
        wake = null
        resolve(text)
      } else {
        buffered.push(text)
      }
    },
    close() {
      closed = true
      if (wake) {
        const resolve = wake
        wake = null
        resolve(null)
      }
    },
    next() {
      const text = buffered.shift()
      if (text !== undefined) return Promise.resolve(text)
      if (closed) return Promise.resolve(null)
      return new Promise(resolve => { wake = resolve })
    },
  }
}

/**
 * Creates a session while yielding `status` events: one while Chrome prepares
 * the model, then download progress reported through the Prompt API `monitor`.
 */
async function* createSession(
  model: LanguageModelGlobal,
  options: Record<string, unknown>
): AsyncGenerator<ChromeHarnessEvent, PromptSession> {
  const status = createStatusChannel()
  let lastPercent = -1
  options.monitor = (monitor: DownloadMonitor) => {
    monitor.addEventListener('downloadprogress', event => {
      const total = typeof event.total === 'number' && event.total > 0 ? event.total : 0
      const percent = total > 0 ? Math.round((event.loaded / total) * 100) : -1
      if (percent === lastPercent) return
      lastPercent = percent
      status.push(percent >= 0
        ? `Downloading the on-device model… ${percent}%`
        : 'Downloading the on-device model…')
    })
  }
  yield { type: 'status', text: 'Preparing the on-device model…' }
  const creating = model.create(options).then(
    session => ({ ok: true as const, session }),
    (error: unknown) => ({ ok: false as const, error })
  ).finally(() => status.close())
  while (true) {
    const text = await status.next()
    if (text === null) break
    yield { type: 'status', text }
  }
  const outcome = await creating
  if (!outcome.ok) throw outcome.error
  return outcome.session
}

function makeToolResponse(call: NativeToolCall, ok: boolean, text: string): unknown {
  const globals = globalThis as Record<string, unknown>
  const Success = globals.LanguageModelToolSuccess as ToolResponseConstructor | undefined
  const Failure = globals.LanguageModelToolError as ToolResponseConstructor | undefined
  // Constructors ignore members they do not know, so passing both spellings
  // works on builds before and after the `callID` → `callId` rename.
  const callId = call.callId ?? call.callID ?? ''
  if (ok) {
    const init = { callId, callID: callId, name: call.name, result: [{ type: 'text', value: text }] }
    return Success ? new Success(init) : init
  }
  const init = { callId, callID: callId, name: call.name, errorMessage: text }
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
      session = yield* createSession(LanguageModel, createSessionOptions(system, tools, options.signal))
    } catch (err) {
      // Older Chrome builds expose the Prompt API but not tool use. Retry as a
      // plain text session so the chat still works (without project access).
      const reason = err instanceof Error ? err.message : String(err)
      tools = []
      system = options.plainSystem
      yield { type: 'fallback', reason }
      session = yield* createSession(LanguageModel, createSessionOptions(system, tools, options.signal))
    }
  } else {
    session = yield* createSession(LanguageModel, createSessionOptions(system, tools, options.signal))
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
  /** Current editor markers, including unsaved files. */
  getDiagnostics?: () => HarnessDiagnostic[]
  /** The file currently open in the editor, including unsaved changes. */
  getOpenFile?: () => HarnessOpenFile | null
  /** The current editor selection. */
  getSelection?: () => HarnessSelection | null
}

export type HarnessSeverity = 'error' | 'warning' | 'info' | 'hint'

export interface HarnessDiagnostic {
  path: string
  severity: HarnessSeverity
  message: string
  line: number
  column: number
}

export interface HarnessOpenFile {
  path: string
  content: string
}

export interface HarnessSelection {
  path: string | null
  text: string
  startLine: number
  endLine: number
}

const MAX_READ_LINES = 600
const MAX_READ_CHARS = 16000
const MAX_LISTED_FILES = 500
const MAX_SEARCH_FILES = 400
const MAX_SEARCH_BYTES = 2_000_000
const MAX_SEARCH_MATCHES = 40
const MAX_FIND_MATCHES = 200
const MAX_EDITOR_TEXT = 16000

function normalizePath(path: unknown): string {
  return String(path ?? '').replace(/\\/g, '/').replace(/^\.?\//, '').replace(/\/+$/, '')
}

const MAX_WRITE_CHARS = 512_000
const PROTECTED_DIRS: Record<string, true> = { '.git': true, node_modules: true }

type PathCheck = { path: string } | { error: string }

/**
 * Validates a model-supplied path. The tool layer is the trust boundary: a
 * small on-device model will happily emit `../`, absolute paths, or `.git/…`,
 * and the File System Access API must never see them.
 */
function checkProjectPath(value: unknown): PathCheck {
  const raw = String(value ?? '').trim()
  if (!raw) return { error: 'path is required' }
  if (raw.includes('\u0000')) return { error: 'Invalid path.' }
  const slashed = raw.replace(/\\/g, '/')
  if (slashed.startsWith('/') || /^[a-zA-Z]:\//.test(slashed)) {
    return { error: `Use a path relative to the project root, not an absolute path: ${raw}` }
  }
  const segments = normalizePath(slashed).split('/').filter(segment => segment !== '.' && segment !== '')
  if (segments.length === 0) return { error: 'path is required' }
  if (segments.includes('..')) {
    return { error: `Refusing a path outside the project root: ${raw}` }
  }
  if (segments.some(segment => PROTECTED_DIRS[segment] === true)) {
    return { error: `Refusing to touch protected directories (${Object.keys(PROTECTED_DIRS).join(', ')}): ${raw}` }
  }
  return { path: segments.join('/') }
}

/** Minimal glob matcher: `*`, `**`, `?`, and literal path segments. */
function globToRegExp(pattern: string): RegExp {
  let source = '^'
  for (let i = 0; i < pattern.length; i++) {
    const char = pattern[i]
    if (char === '*') {
      if (pattern[i + 1] === '*') {
        i++
        if (pattern[i + 1] === '/') {
          i++
          source += '(?:.*/)?'
        } else {
          source += '.*'
        }
      } else {
        source += '[^/]*'
      }
      continue
    }
    if (char === '?') {
      source += '[^/]'
      continue
    }
    source += char.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  }
  return new RegExp(`${source}$`)
}

export function createHarnessTools(fs: HarnessFs): HarnessTool[] {
  // Files the model has read (or written) this turn. Mutations require a prior
  // read so the model cannot rewrite code it has never seen.
  const clearedForEdit = new Set<string>()
  const ensureRead = (path: string) => clearedForEdit.has(path)
    ? null
    : `Read "${path}" with readFile before editing or deleting it.`
  const filesUnder = async (prefix: string) => {
    const all = (await fs.listFiles()).slice().sort()
    if (!prefix) return all
    return all.filter(path => path === prefix || path.startsWith(`${prefix}/`))
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
        const dir = String(path ?? '').trim()
        const check = dir ? checkProjectPath(dir) : { path: '' }
        if ('error' in check) return { error: check.error }
        const needle = String(query ?? '').toLowerCase()
        const matches = (await filesUnder(check.path)).filter(file => !needle || file.toLowerCase().includes(needle))
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
        const dir = String(path ?? '').trim()
        const check = dir ? checkProjectPath(dir) : { path: '' }
        if ('error' in check) return { error: check.error }
        const needle = String(query ?? '').toLowerCase()
        if (!needle) return { error: 'query is required' }
        const matches: Array<{ path: string; line: number; text: string }> = []
        let scanned = 0
        let scannedFiles = 0
        let truncated = false
        for (const file of await filesUnder(check.path)) {
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
      name: 'findFiles',
      description: 'Find files by glob pattern. Use it to locate files by name or extension, e.g. "src/**/*.ts" or "*.json".',
      inputSchema: {
        type: 'object',
        properties: {
          pattern: { type: 'string', description: 'Glob pattern. Supports *, ** and ?.' },
          path: { type: 'string', description: 'Optional directory limit, e.g. "src".' },
        },
        required: ['pattern'],
      },
      execute: async ({ pattern, path }) => {
        const dir = String(path ?? '').trim()
        const check = dir ? checkProjectPath(dir) : { path: '' }
        if ('error' in check) return { error: check.error }
        const glob = String(pattern ?? '').trim().replace(/\\/g, '/').replace(/^\.?\//, '')
        if (!glob) return { error: 'pattern is required' }
        if (glob.split('/').includes('..')) return { error: 'Patterns cannot traverse outside the project.' }
        const matcher = globToRegExp(glob)
        const matchFullPath = glob.includes('/')
        const matches = (await filesUnder(check.path)).filter(file => {
          const candidate = matchFullPath ? file : file.split('/').pop() || file
          return matcher.test(candidate)
        })
        return {
          pattern: glob,
          count: matches.length,
          files: matches.slice(0, MAX_FIND_MATCHES),
          truncated: matches.length > MAX_FIND_MATCHES,
        }
      },
    },
    {
      name: 'readFile',
      description: 'Read a project file. Required before replaceInFile, editFile, or deleteFile. Optionally pass a 1-based line range for large files (max 600 lines / 24000 characters per call).',
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
        const check = checkProjectPath(path)
        if ('error' in check) return { error: check.error }
        const file = check.path
        const content = await fs.readFile(file)
        if (content == null) return { error: `File not found: ${file}` }
        clearedForEdit.add(file)
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
      name: 'getDiagnostics',
      description: 'Return current editor diagnostics (errors and warnings) for open files. Use it after edits to check for broken imports, types or syntax.',
      inputSchema: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Optional file path filter.' },
        },
        required: [],
      },
      execute: async ({ path }) => {
        if (!fs.getDiagnostics) return { error: 'Editor diagnostics are not available.' }
        const filter = String(path ?? '').trim()
        const check = filter ? checkProjectPath(filter) : null
        if (check && 'error' in check) return { error: check.error }
        const diagnostics = fs.getDiagnostics().filter(item => !check || item.path === check.path)
        return {
          count: diagnostics.length,
          diagnostics: diagnostics.slice(0, 100),
          truncated: diagnostics.length > 100,
        }
      },
    },
    {
      name: 'getOpenFile',
      description: 'Read the file currently open in the editor, including unsaved changes.',
      inputSchema: { type: 'object', properties: {}, required: [] },
      execute: async () => {
        if (!fs.getOpenFile) return { error: 'No editor is available.' }
        const open = fs.getOpenFile()
        if (!open) return { error: 'No file is open in the editor.' }
        return {
          path: open.path,
          content: truncate(open.content, MAX_EDITOR_TEXT),
          truncated: open.content.length > MAX_EDITOR_TEXT,
        }
      },
    },
    {
      name: 'getSelection',
      description: 'Read the current editor selection and its line range.',
      inputSchema: { type: 'object', properties: {}, required: [] },
      execute: async () => {
        if (!fs.getSelection) return { error: 'No editor is available.' }
        const selection = fs.getSelection()
        if (!selection || !selection.text.trim()) return { error: 'No text is selected in the editor.' }
        return {
          ...selection,
          text: truncate(selection.text, MAX_EDITOR_TEXT),
          truncated: selection.text.length > MAX_EDITOR_TEXT,
        }
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
        const check = checkProjectPath(path)
        if ('error' in check) return { error: check.error }
        const file = check.path
        const body = String(content ?? '')
        if (body.length > MAX_WRITE_CHARS) {
          return { error: `Refusing to write ${body.length} characters (limit ${MAX_WRITE_CHARS}). Split the change into smaller edits.` }
        }
        const ok = await fs.createFile(file, body)
        if (ok) clearedForEdit.add(file)
        return ok ? { path: file, created: true } : { error: `Failed to create ${file} (it may already exist)` }
      },
    },
    {
      name: 'replaceInFile',
      description: 'Preferred edit tool. Replaces one exact snippet in a file you have already read this turn. Fails when oldText is missing or matches more than once unless replaceAll is true. Keep oldText large enough to be unique and include surrounding context.',
      inputSchema: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path relative to the project root.' },
          oldText: { type: 'string', description: 'Exact text to replace, including surrounding context to make it unique.' },
          newText: { type: 'string', description: 'Replacement text.' },
          replaceAll: { type: 'boolean', description: 'Replace every occurrence. Defaults to false.' },
        },
        required: ['path', 'oldText', 'newText'],
      },
      execute: async ({ path, oldText, newText, replaceAll }) => {
        const check = checkProjectPath(path)
        if ('error' in check) return { error: check.error }
        const file = check.path
        const guard = ensureRead(file)
        if (guard) return { error: guard }
        const needle = String(oldText ?? '')
        if (!needle) return { error: 'oldText is required' }
        const content = await fs.readFile(file)
        if (content == null) return { error: `File not found: ${file}` }
        const replacement = String(newText ?? '')
        const count = content.split(needle).length - 1
        if (count === 0) return { error: `oldText was not found in ${file}` }
        if (count > 1 && replaceAll !== true) {
          return { error: `oldText matches ${count} times in ${file}; add surrounding context or set replaceAll to true` }
        }
        const next = replaceAll === true
          ? content.split(needle).join(replacement)
          : content.slice(0, content.indexOf(needle)) + replacement + content.slice(content.indexOf(needle) + needle.length)
        if (next.length > MAX_WRITE_CHARS) {
          return { error: `Refusing to write ${next.length} characters (limit ${MAX_WRITE_CHARS}). Split the change into smaller edits.` }
        }
        const ok = await fs.writeFile(file, next)
        return ok ? { path: file, replacements: replaceAll === true ? count : 1 } : { error: `Failed to write ${file}` }
      },
    },
    {
      name: 'editFile',
      description: 'Last resort: replaces the entire contents of an existing file. Requires a prior readFile this turn and refuses empty content. Use replaceInFile whenever possible; use createFile for new files.',
      inputSchema: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path relative to the project root.' },
          content: { type: 'string', description: 'Full new file content.' },
        },
        required: ['path', 'content'],
      },
      execute: async ({ path, content }) => {
        const check = checkProjectPath(path)
        if ('error' in check) return { error: check.error }
        const file = check.path
        const guard = ensureRead(file)
        if (guard) return { error: guard }
        const body = String(content ?? '')
        if (!body.trim()) {
          return { error: `Refusing to replace ${file} with empty content; use deleteFile if the file should be removed.` }
        }
        if (body.length > MAX_WRITE_CHARS) {
          return { error: `Refusing to write ${body.length} characters (limit ${MAX_WRITE_CHARS}). Split the change into smaller edits.` }
        }
        if (await fs.readFile(file) == null) return { error: `File not found: ${file}. Use createFile for new files.` }
        const ok = await fs.writeFile(file, body)
        return ok ? { path: file, written: true } : { error: `Failed to write ${file}` }
      },
    },
    {
      name: 'deleteFile',
      description: 'Delete a file. Requires a prior readFile this turn, and should only be used when the user explicitly asked to delete that file.',
      inputSchema: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path relative to the project root.' },
        },
        required: ['path'],
      },
      execute: async ({ path }) => {
        const check = checkProjectPath(path)
        if ('error' in check) return { error: check.error }
        const file = check.path
        const guard = ensureRead(file)
        if (guard) return { error: guard }
        const ok = await fs.deleteFile(file)
        return ok ? { path: file, deleted: true } : { error: `Failed to delete ${file}` }
      },
    },
  ]
}
