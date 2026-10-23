import { html, render, useState, useEffect, useRef, useCallback, useMemo } from 'htm/preact/standalone.mjs'
import { PROVIDERS, getAvailableModels, estimateVRAM, getPreferredModel, checkProviderAvailability, getBrowserAIStatus, initProviderModel, createChatStream, createTools, type Provider, type ModelDescriptor } from './llm-provider'
import { createHarnessTools, streamChromeHarness, type HarnessTool } from './chrome-harness'
import type { LanguageModel, ModelMessage } from 'ai'

interface Notification {
  id: number
  text: string
  level: 'info' | 'warning' | 'error'
}

let notificationId = 0
const notifications: Notification[] = []
const notificationListeners: Array<() => void> = []

function addNotification(text: string, level: Notification['level'] = 'info') {
  notifications.push({ id: ++notificationId, text, level })
  notificationListeners.forEach(fn => fn())
  setTimeout(() => {
    const idx = notifications.findIndex(n => n.id === notificationId)
    if (idx >= 0) {
      notifications.splice(idx, 1)
      notificationListeners.forEach(fn => fn())
    }
  }, 8000)
}

// Hook into AI SDK warnings
const origWarn = console.warn
console.warn = (...args: any[]) => {
  const msg = args.join(' ')
  if (msg.includes('AI SDK Warning')) {
    const match = msg.match(/AI SDK Warning[^:]*:\s*(.+)/)
    if (match) addNotification(match[1], 'warning')
  }
  origWarn.apply(console, args)
}

interface FileEntry {
  name: string
  path: string
  type: 'dir' | 'file'
  children?: FileEntry[]
  /** Directory children have been enumerated. */
  loaded?: boolean
  handle: FileSystemHandle
}

interface CodeBlock {
  lang: string
  path: string
  code: string
  applied: boolean
}

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  isStreaming?: boolean
}

interface InitProgressState {
  text: string
  progress: number
}

interface ChatChunk {
  text: string
  done: boolean
  delta: string
  usage: unknown
}

declare global {
  interface Window {
    __gpuAvailable: boolean
    __transformersAvailable: boolean
    __browserAiAvailable: boolean
    __browserAiNote: string
    MonacoEnvironment: {
      getWorker: (id: string, label: string) => Worker
    }
    require?: {
      config: (cfg: { paths: { vs: string } }) => void
    } & ((deps: string[], callback: () => void) => void)
    monaco?: any
  }
}

const STORAGE_KEY = 'coder-state'

interface PersistedState {
  selectedProvider: Provider
  selectedModel: string
  customModelUrl: string
  attachContext: boolean
  messages: Message[]
}

function loadState(): Partial<PersistedState> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return {}
    return JSON.parse(raw)
  } catch {
    return {}
  }
}

function saveState(state: Partial<PersistedState>) {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    const existing = raw ? JSON.parse(raw) : {}
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...existing, ...state }))
  } catch {}
}

function clearState() {
  try { localStorage.removeItem(STORAGE_KEY) } catch {}
}

const EXCLUDED = new Set(['node_modules','.git','dist','.DS_Store','__pycache__','.next','.vscode','build','.cache','.venv','venv','env','.idea','coverage','.nyc_output'])
const MAX_DEPTH = 8
const MAX_CONTEXT_FILES = 8
const MAX_CONTEXT_FILE_SIZE = 1500
// Suppress AI SDK warnings (toolChoice not supported by WebLLM is expected)
globalThis.AI_SDK_LOG_WARNINGS = false

// Suppress AI SDK warnings (toolChoice not supported by WebLLM is expected)
globalThis.AI_SDK_LOG_WARNINGS = false

// Suppress AI SDK warnings (toolChoice not supported by WebLLM is expected)
globalThis.AI_SDK_LOG_WARNINGS = false

import * as monaco from 'monaco-editor'

self.MonacoEnvironment = {
  getWorker() {
    const blob = new Blob(['self.onmessage = () => {}'], { type: 'application/javascript' })
    return new Worker(URL.createObjectURL(blob), { type: 'module' })
  }
}

monaco.languages.register({ id: 'plaintext' })
monaco.languages.register({ id: 'javascript' })
monaco.languages.register({ id: 'typescript' })
monaco.languages.register({ id: 'python' })
monaco.languages.register({ id: 'html' })
monaco.languages.register({ id: 'css' })
monaco.languages.register({ id: 'json' })
monaco.languages.register({ id: 'markdown' })
monaco.languages.register({ id: 'xml' })
monaco.languages.register({ id: 'yaml' })
monaco.languages.register({ id: 'shell' })
monaco.languages.register({ id: 'sql' })
monaco.languages.register({ id: 'rust' })
monaco.languages.register({ id: 'go' })
monaco.languages.register({ id: 'java' })
monaco.languages.register({ id: 'ruby' })
monaco.languages.register({ id: 'php' })
monaco.languages.register({ id: 'c' })
monaco.languages.register({ id: 'cpp' })
monaco.languages.register({ id: 'graphql' })
const LANG_MAP: Record<string, string> = {
  '.js':'javascript','.jsx':'javascript','.mjs':'javascript','.cjs':'javascript',
  '.ts':'typescript','.tsx':'typescript',
  '.py':'python','.pyw':'python',
  '.html':'html','.htm':'html',
  '.css':'css','.scss':'scss','.sass':'scss','.less':'less',
  '.json':'json','.jsonc':'json','.json5':'json',
  '.md':'markdown','.mdx':'markdown',
  '.xml':'xml','.svg':'xml',
  '.yaml':'yaml','.yml':'yaml',
  '.rs':'rust','.go':'go','.java':'java','.rb':'ruby','.php':'php',
  '.c':'c','.cpp':'cpp','.h':'c','.hpp':'cpp',
  '.sh':'shell','.bash':'shell','.zsh':'shell',
  '.sql':'sql','.graphql':'graphql','.gql':'graphql',
  '.toml':'plaintext','.ini':'plaintext','.cfg':'plaintext',
  '.wasm':'plaintext','.lock':'plaintext',
}

const SYSTEM_PROMPT = `You are Coder, an AI coding assistant running entirely inside the user's browser. All computation is local and private.

When file tools are not available, suggest code changes with a code block that includes the file path header like this:
\`\`\`language:path/to/file.js
// your code here
\`\`\`

Rules:
1. Always specify the exact file path after the language name, separated by a colon.
2. When creating new files, specify the full path relative to the project root.
3. Explain your changes concisely before the code block.
4. If multiple files need changes, use separate code blocks for each.
5. Be precise and minimal in your code changes.
6. When asked to review code, point out specific issues with line numbers.`

const TOOL_SYSTEM_PROMPT = `${SYSTEM_PROMPT}

You can inspect and modify the user's local project with tools:
- listFiles: list project file paths (optionally filter by directory or filename substring)
- searchFiles: case-insensitive search in file contents
- readFile: read a file, optionally a 1-based line range
- createFile: create a new file
- editFile: overwrite an existing file (creates it if missing)
- deleteFile: delete a file

Rules for tool use:
1. Find relevant files first (listFiles/searchFiles), then read them before editing.
2. Paths are relative to the project root. Never invent file contents.
3. Make minimal, complete edits; verify the result when practical.
4. If a tool returns an error, adapt instead of repeating the same call.
5. When done, summarize exactly what changed and where.`

let editor: any = null
let currentFilePath: string | null = null
let currentFileHandle: FileSystemFileHandle | null = null
let rootDirHandle: FileSystemDirectoryHandle | null = null
let rootDirName = ''
const fileHandles = new Map<string, FileEntry>()
const monacoModels = new Map<string, any>()
let editorContentVersion = 0
let autoSaveTimer: ReturnType<typeof setTimeout> | null = null
let currentModel: LanguageModel | null = null
let abortController: AbortController | null = null
let modelLoadPromise: Promise<LanguageModel> | null = null

async function verifyPermission(fileHandle: FileSystemHandle, readWrite = false): Promise<boolean> {
  const options: FileSystemHandlePermissionDescriptor = {}
  if (readWrite) options.mode = 'readwrite'
  if ((await (fileHandle as any).queryPermission(options)) === 'granted') return true
  if ((await (fileHandle as any).requestPermission(options)) === 'granted') return true
  return false
}

async function readFileContent(fileHandle: FileSystemFileHandle): Promise<string> {
  const file = await fileHandle.getFile()
  return await file.text()
}

async function writeFileContent(fileHandle: FileSystemFileHandle, content: string): Promise<void> {
  const writable = await fileHandle.createWritable()
  await writable.write(content)
  await writable.close()
}

/** Enumerates one directory level. Child directories stay unloaded until expanded. */
async function readDirEntries(dirHandle: FileSystemDirectoryHandle, path = '', generation = folderGeneration): Promise<FileEntry[]> {
  const entries: FileEntry[] = []
  for await (const [name, handle] of dirHandle.entries()) {
    if (EXCLUDED.has(name)) continue
    const entryPath = path ? `${path}/${name}` : name
    if (handle.kind === 'directory') {
      entries.push({ name, path: entryPath, type: 'dir', handle, loaded: false })
    } else {
      const entry: FileEntry = { name, path: entryPath, type: 'file', handle }
      entries.push(entry)
      if (generation === folderGeneration) fileHandles.set(entryPath, entry)
    }
  }
  entries.sort((a, b) => {
    if (a.type !== b.type) return a.type === 'dir' ? -1 : 1
    return a.name.localeCompare(b.name)
  })
  return entries
}

const MAX_INDEX_ENTRIES = 50000
let folderGeneration = 0
let indexPromise: Promise<void> | null = null
const loadingDirs = new Set<string>()

/**
 * Builds the path index used by the AI tools without ever blocking the tree
 * UI. Runs in the background, yields to the event loop periodically and stops
 * when the folder changes or the entry budget is reached.
 */
async function indexFolder(
  dirHandle: FileSystemDirectoryHandle,
  generation: number,
  onProgress?: (indexedFiles: number) => void
): Promise<void> {
  const queue: Array<{ handle: FileSystemDirectoryHandle; path: string; depth: number }> = [
    { handle: dirHandle, path: '', depth: 0 },
  ]
  let seen = 0
  try {
    for (let cursor = 0; cursor < queue.length; cursor++) {
      if (generation !== folderGeneration) return
      const current = queue[cursor]
      if (current.depth > MAX_DEPTH) continue
      try {
        for await (const [name, child] of current.handle.entries()) {
          if (generation !== folderGeneration) return
          if (EXCLUDED.has(name)) continue
          const path = current.path ? `${current.path}/${name}` : name
          if (child.kind === 'directory') {
            queue.push({ handle: child, path, depth: current.depth + 1 })
          } else if (!fileHandles.has(path)) {
            fileHandles.set(path, { name, path, type: 'file', handle: child })
          }
          if (++seen >= MAX_INDEX_ENTRIES) return
          if (seen % 250 === 0) {
            onProgress?.(fileHandles.size)
            const { promise, resolve } = Promise.withResolvers<void>()
            setTimeout(resolve, 0)
            await promise
          }
        }
      } catch (err) {
        console.warn('indexFolder:', current.path, err)
      }
    }
  } finally {
    if (generation === folderGeneration) onProgress?.(fileHandles.size)
  }
}

async function ensureIndexed(): Promise<void> {
  if (indexPromise) await indexPromise
}

async function resolveFileHandle(path: string): Promise<FileSystemFileHandle | null> {
  const cached = fileHandles.get(path)
  if (cached) return cached.handle as FileSystemFileHandle
  if (!rootDirHandle) return null
  try {
    const parts = path.split('/')
    let dir: FileSystemDirectoryHandle = rootDirHandle
    for (let i = 0; i < parts.length - 1; i++) dir = await dir.getDirectoryHandle(parts[i])
    const handle = await dir.getFileHandle(parts[parts.length - 1])
    fileHandles.set(path, { name: parts[parts.length - 1], path, type: 'file', handle })
    return handle
  } catch {
    return null
  }
}

function getLanguage(filePath: string): string {
  const ext = filePath.match(/(\.[^.]+)$/)?.[1] || ''
  return LANG_MAP[ext] || 'plaintext'
}

async function openFolder(onIndexProgress?: (indexedFiles: number) => void): Promise<FileEntry[] | null> {
  try {
    const handle = await (window as any).showDirectoryPicker()
    const generation = ++folderGeneration
    rootDirHandle = handle
    rootDirName = handle.name
    fileHandles.clear()
    loadingDirs.clear()
    // Render only the top level immediately; nested directories load on
    // expand, while a background index feeds the AI tools.
    const entries = await readDirEntries(handle)
    if (generation !== folderGeneration) return null
    indexPromise = indexFolder(handle, generation, onIndexProgress)
    return entries
  } catch (err) {
    if ((err as Error).name !== 'AbortError' && (err as Error).name !== 'SecurityError') throw err
    return null
  }
}

async function loadFile(path: string): Promise<string | null> {
  if (!editor) { console.warn('Editor not ready'); return null }
  const handle = await resolveFileHandle(path)
  if (!handle) return null
  if (!await verifyPermission(handle, false)) { alert(`Cannot read ${path}`); return null }
  try {
    const content = await readFileContent(handle)
    currentFilePath = path
    currentFileHandle = handle
    if (monacoModels.has(path)) {
      const m = monacoModels.get(path)
      if (m.getValue() !== content) m.setValue(content)
      editor.setModel(m)
    } else {
      const model = monaco.editor.createModel(content, getLanguage(path))
      monacoModels.set(path, model)
      editor.setModel(model)
    }
    return path
  } catch (err) {
    console.error('loadFile:', err)
    return null
  }
}

async function saveFile(path?: string): Promise<boolean> {
  const targetPath = path || currentFilePath
  if (!targetPath || !editor) return false
  const handle = await resolveFileHandle(targetPath)
  if (!handle) return false
  if (!await verifyPermission(handle, true)) { alert(`Cannot write ${targetPath}`); return false }
  try {
    await writeFileContent(handle, editor.getValue())
    return true
  } catch (err) { console.error('saveFile:', err); return false }
}

async function applyFileChanges(path: string, code: string): Promise<boolean> {
  let entry = fileHandles.get(path)
  if (!entry) {
    if (!rootDirHandle) return false
    const parts = path.split('/')
    let dirHandle = rootDirHandle
    for (let i = 0; i < parts.length - 1; i++) {
      try { dirHandle = await dirHandle.getDirectoryHandle(parts[i], { create: true }) } catch { return false }
    }
    try {
      const fh = await dirHandle.getFileHandle(parts[parts.length - 1], { create: true })
      fileHandles.set(path, { handle: fh, name: parts[parts.length - 1], path, type: 'file' })
      entry = fileHandles.get(path)
    } catch { return false }
  }
  if (!entry || !await verifyPermission(entry.handle, true)) return false
  try {
    await writeFileContent(entry.handle as FileSystemFileHandle, code)
    if (monacoModels.has(path)) {
      const model = monacoModels.get(path)
      if (model !== editor.getModel()) model.setValue(code)
      else editor.setValue(code)
    } else if (path === currentFilePath) {
      editor.setValue(code)
    }
    return true
  } catch { return false }
}

function initMonaco(container: HTMLElement): Promise<monaco.editor.IStandaloneCodeEditor> {
  return new Promise((resolve) => {
    const ed = monaco.editor.create(container, {
      value: '', language: 'plaintext', theme: 'vs-dark',
      automaticLayout: true, minimap: { enabled: false }, fontSize: 14,
      lineNumbers: 'on', scrollBeyondLastLine: false, tabSize: 2,
      wordWrap: 'off', padding: { top: 8 },
      bracketPairColorization: { enabled: true },
    })
    ed.onDidChangeModelContent(() => { editorContentVersion++ })
    ed.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, async () => { await saveFile() })
    editor = ed
    resolve(ed)
  })
}

async function initLLM(provider: Provider, modelId: string, onProgress: (p: InitProgressState) => void): Promise<LanguageModel> {
  if (modelLoadPromise) {
    return modelLoadPromise
  }
  modelLoadPromise = initProviderModel(provider, modelId, onProgress)
  try {
    currentModel = await modelLoadPromise
    return currentModel
  } finally {
    modelLoadPromise = null
  }
}

function stopGeneration(): void {
  if (abortController) {
    abortController.abort()
    abortController = null
  }
}

async function* streamChat(messages: ModelMessage[], system?: string, plainSystem?: string, tools?: HarnessTool[] | Record<string, any>, signal?: AbortSignal): AsyncGenerator<ChatChunk> {
  if (!currentModel) throw new Error('Model not initialized')
  const provider = (currentModel as any).provider || PROVIDERS.WEBLLM

  if (provider === PROVIDERS.BROWSER_AI) {
    for await (const event of streamChromeHarness({
      system: system || '',
      plainSystem: plainSystem || system || '',
      messages,
      tools: Array.isArray(tools) ? tools : [],
      signal,
    })) {
      if (signal?.aborted) break
      if (event.type === 'text') {
        yield { text: event.text, done: false, delta: event.delta, usage: null }
      } else if (event.type === 'tool-call') {
        console.log('Tool call:', event.name, event.args)
      } else if (event.type === 'tool-result') {
        console.log('Tool result:', event.name, event.ok ? 'ok' : 'error', event.preview)
      } else if (event.type === 'fallback') {
        addNotification(`Chrome tool use unavailable (${event.reason}). Continuing without project tools.`, 'warning')
      } else if (event.type === 'done') {
        yield { text: event.text, done: true, delta: '', usage: null }
        return
      }
    }
    yield { text: '', done: true, delta: '', usage: null }
    return
  }

  let full = ''
  try {
    const result = createChatStream({ provider, model: currentModel, messages, system, tools: tools as Record<string, any>, abortSignal: signal })
    for await (const part of result.fullStream) {
      if (signal?.aborted) break
      if (part.type === 'text-delta') {
        full += (part as any).text ?? (part as any).delta ?? ''
        const cleaned = full.replace(/ thinking[\s\S]*?<\/think>/g, '').trim()
        yield { text: cleaned, done: false, delta: (part as any).text ?? (part as any).delta ?? '', usage: null }
      } else if ((part as any).type === 'reasoning-delta') {
        // Some providers stream thinking as reasoning deltas; surface it so
        // the bubble never stays empty while the model is working.
        full += (part as any).delta ?? (part as any).text ?? ''
        const cleaned = full.replace(/ thinking[\s\S]*?<\/think>/g, '').trim()
        yield { text: cleaned, done: false, delta: (part as any).delta ?? '', usage: null }
      } else if ((part as any).type === 'tool-call' || (part as any).type === 'tool-input-available' || (part as any).type === 'tool-input-start') {
        console.log('Tool call:', (part as any).toolName || 'unknown tool', (part as any).input ?? (part as any).args)
      } else if ((part as any).type === 'tool-result' || (part as any).type === 'tool-output-available') {
        console.log('Tool result:', (part as any).toolName, (part as any).output)
      } else if ((part as any).type === 'error') {
        const raw = (part as any).error ?? (part as any).errorText ?? (part as any).message
        const detail = raw instanceof Error ? raw.message : typeof raw === 'string' ? raw : JSON.stringify(raw ?? part)
        console.error('Chat stream error part:', part)
        throw new Error(detail || 'Stream error')
      }
    }
    const cleaned = full.replace(/<think>[\s\S]*?<\/think>/g, '').trim()
    yield { text: cleaned, done: true, delta: '', usage: null }
  } catch (err) {
    const msg = (err as Error).message || ''
    if (msg.includes('mapAsync') || msg.includes('Buffer was unmapped')) {
      console.warn('GPU buffer error, retrying...')
      await new Promise(r => setTimeout(r, 500))
      const cleaned = full.replace(/<think>[\s\S]*?<\/think>/g, '').trim()
      yield { text: cleaned, done: true, delta: '', usage: null }
      return
    }
    if (abortController?.signal.aborted) {
      const cleaned = full.replace(/<think>[\s\S]*?<\/think>/g, '').trim()
      yield { text: cleaned, done: true, delta: '', usage: null }
      return
    }
    throw err
  }
}

function parseCodeBlocks(text: string): CodeBlock[] {
  const blocks: CodeBlock[] = []
  const re = /```(\w+(?::[^\n]*)?)\n([\s\S]*?)```/g
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    const h = m[1]; const ci = h.indexOf(':')
    blocks.push({
      lang: ci >= 0 ? h.slice(0, ci) : h,
      path: ci >= 0 ? h.slice(ci + 1).trim() : '',
      code: m[2].trim(),
      applied: false,
    })
  }
  return blocks
}

function updateEntry(
  entries: FileEntry[] | null,
  path: string,
  update: (entry: FileEntry) => FileEntry
): FileEntry[] | null {
  if (!entries) return null
  return entries.map(entry => {
    if (entry.path === path) return update(entry)
    if (entry.children && path.startsWith(`${entry.path}/`)) {
      const children = updateEntry(entry.children, path, update)
      if (children !== entry.children) return { ...entry, children: children ?? [] }
    }
    return entry
  })
}

function buildFileTree(treeData: FileEntry[]): string {
  const lines = [`Project: ${rootDirName}`, '']
  function flat(entries: FileEntry[], p = '') {
    for (const e of entries) {
      if (e.type === 'dir') { lines.push(`${p}${e.name}/`); if (e.children) flat(e.children, p + '  ') }
      else lines.push(`${p}${e.name}`)
    }
  }
  flat(treeData)
  return lines.join('\n')
}

async function readFileForTool(path: string): Promise<string | null> {
  const handle = await resolveFileHandle(path)
  if (!handle) return null
  try {
    return await readFileContent(handle)
  } catch {
    return null
  }
}

async function createFileForTool(path: string, content: string): Promise<boolean> {
  if (fileHandles.has(path)) return false
  if (!rootDirHandle) return false
  const parts = path.split('/')
  let dirHandle = rootDirHandle
  for (let i = 0; i < parts.length - 1; i++) {
    try { dirHandle = await dirHandle.getDirectoryHandle(parts[i], { create: true }) } catch { return false }
  }
  try {
    const fh = await dirHandle.getFileHandle(parts[parts.length - 1], { create: true })
    await writeFileContent(fh, content)
    fileHandles.set(path, { handle: fh, name: parts[parts.length - 1], path, type: 'file' })
    return true
  } catch { return false }
}

async function deleteFileForTool(path: string): Promise<boolean> {
  if (!rootDirHandle) return false
  try {
    const parts = path.split('/')
    let dirHandle: FileSystemDirectoryHandle = rootDirHandle
    for (let i = 0; i < parts.length - 1; i++) {
      dirHandle = await dirHandle.getDirectoryHandle(parts[i])
    }
    await dirHandle.removeEntry(parts[parts.length - 1])
    fileHandles.delete(path)
    if (monacoModels.has(path)) {
      monacoModels.get(path).dispose()
      monacoModels.delete(path)
    }
    if (path === currentFilePath) {
      currentFilePath = null
      currentFileHandle = null
    }
    return true
  } catch { return false }
}

interface SidebarProps {
  treeData: FileEntry[] | null
  indexedCount: number
  expandedPaths: Set<string>
  onToggleDir: (entry: FileEntry) => void
  onFileClick: (path: string) => void
  selectedFile: string | null
  modelList: ModelDescriptor[]
  selectedModel: string
  onModelChange: (model: string) => void
  modelLoaded: boolean
  modelLoading: boolean
  initProgress: InitProgressState | null
  onLoadModel: () => void
  onOpenFolder: () => void
  attachContext: boolean
  onAttachContextChange: (checked: boolean) => void
  customModelUrl: string
  onCustomModel: (url: string) => void
  selectedProvider: Provider
  onProviderChange: (provider: Provider) => void
}

function Sidebar({ treeData, indexedCount, expandedPaths, onToggleDir, onFileClick, selectedFile, modelList, selectedModel, onModelChange, modelLoaded, modelLoading, initProgress, onLoadModel, onOpenFolder, attachContext, onAttachContextChange, customModelUrl, onCustomModel, selectedProvider, onProviderChange }: SidebarProps) {
  const visibleCount = useMemo(() => {
    if (!treeData) return 0
    let n = 0
    function w(e: FileEntry[]) { for (const x of e) { if (x.type === 'file') n++; if (x.children) w(x.children) } }
    w(treeData)
    return n
  }, [treeData])

  return html`
    <div class="flex flex-col h-full">
      <div class="p-3 border-b border-[#3c3c3c] space-y-2">
        <button class="w-full px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded flex items-center justify-center gap-1.5 transition-colors" onClick=${onOpenFolder}>
          <span>📂</span>
          <span>${treeData ? 'Change Folder' : 'Open Local Folder'}</span>
        </button>
        ${treeData ? html`<div class="text-xs text-gray-500 flex items-center justify-between"><span class="truncate">${rootDirName}</span><span>${Math.max(visibleCount, indexedCount)} files</span></div>` : ''}
      </div>
      <div class="flex-1 overflow-y-auto scrollbar-thin">
        ${treeData ? html`<${FileTree} entries=${treeData} expandedPaths=${expandedPaths} onToggle=${onToggleDir} onFileClick=${onFileClick} selectedFile=${selectedFile} depth=${0} />`
        : html`<div class="p-4 text-center text-gray-600 text-xs mt-8"><p class="mb-2">📁</p><p>No folder open</p></div>`}
      </div>
      <div class="p-3 border-t border-[#3c3c3c] space-y-2">
        <select class="w-full bg-[#3c3c3c] text-gray-200 text-xs rounded px-2 py-1.5 border border-[#4c4c4c] outline-none focus:border-blue-500" value=${selectedProvider} onChange=${(e: Event) => onProviderChange((e.target as HTMLSelectElement).value as Provider)} disabled=${modelLoading}>
          ${window.__gpuAvailable ? html`<option value=${PROVIDERS.WEBLLM}>WebLLM</option>` : ''}
          ${window.__transformersAvailable ? html`<option value=${PROVIDERS.TRANSFORMERS_JS}>Transformers.js</option>` : ''}
          ${window.__browserAiAvailable ? html`<option value=${PROVIDERS.BROWSER_AI}>Chrome Built-in AI</option>` : html`<option disabled>${window.__browserAiNote || 'Chrome AI (not available)'}</option>`}
        </select>
        <select class="w-full bg-[#3c3c3c] text-gray-200 text-xs rounded px-2 py-1.5 border border-[#4c4c4c] outline-none focus:border-blue-500" value=${selectedModel} onChange=${(e: Event) => onModelChange((e.target as HTMLSelectElement).value)} disabled=${modelLoading}>
          ${modelList.filter(m => m.provider === selectedProvider).map(m => html`<option value=${m.id}>${m.label || m.id}${m.vram ? ' ('+m.vram+'MB)' : ''}</option>`)}
        </select>
        <input type="text" class="w-full bg-[#3c3c3c] text-gray-200 text-xs rounded px-2 py-1.5 border border-[#4c4c4c] outline-none focus:border-blue-500 placeholder-gray-500" placeholder="Custom HF model URL..." value=${customModelUrl} onInput=${(e: Event) => onCustomModel((e.target as HTMLInputElement).value)} />
        ${initProgress ? html`
          <div class="space-y-1">
            <div class="flex justify-between text-xs text-gray-400"><span class="truncate">${initProgress.text}</span><span>${Math.round(initProgress.progress * 100)}%</span></div>
            <div class="h-1.5 bg-[#3c3c3c] rounded-full overflow-hidden"><div class="h-full bg-blue-500 rounded-full transition-all duration-300" style=${{ width: `${initProgress.progress * 100}%` }}></div></div>
          </div>
        ` : ''}
        <button class="w-full px-3 py-1.5 text-xs rounded transition-colors ${modelLoaded ? 'bg-green-700 hover:bg-green-600 text-white' : modelLoading ? 'bg-yellow-700 text-yellow-200 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 text-white'}" onClick=${onLoadModel} disabled=${modelLoading}>
          ${modelLoaded ? '✓ Model Ready' : modelLoading ? 'Loading...' : 'Load Model'}
        </button>
        <label class="flex items-center gap-2 text-xs text-gray-400 cursor-pointer"><input type="checkbox" checked=${attachContext} onChange=${(e: Event) => onAttachContextChange((e.target as HTMLInputElement).checked)} class="accent-blue-500" /><span>Attach Project Context</span></label>
      </div>
    </div>
  `
}

interface FileTreeProps {
  entries: FileEntry[]
  expandedPaths: Set<string>
  onToggle: (entry: FileEntry) => void
  onFileClick: (path: string) => void
  selectedFile: string | null
  depth: number
}

function FileTree({ entries, expandedPaths, onToggle, onFileClick, selectedFile, depth }: FileTreeProps) {
  return html`${entries.map(e => {
    if (e.type === 'dir') {
      const open = expandedPaths.has(e.path)
      return html`<div key=${e.path}>
        <div class="flex items-center cursor-pointer px-2 py-0.5 text-gray-300 hover:bg-[#2a2a2a] file-tree-item" style=${{ paddingLeft: `${depth*14+8}px` }} onClick=${() => onToggle(e)}>
          <span class="text-xs w-4 shrink-0">${open ? '▾' : '▸'}</span>
          <span class="text-xs mr-1 shrink-0">${open ? '📂' : '📁'}</span>
          <span class="text-xs truncate">${e.name}</span>
        </div>
        ${open
          ? e.loaded
            ? (e.children ? html`<${FileTree} entries=${e.children} expandedPaths=${expandedPaths} onToggle=${onToggle} onFileClick=${onFileClick} selectedFile=${selectedFile} depth=${depth+1} />` : '')
            : html`<div class="text-xs text-gray-600 py-0.5" style=${{ paddingLeft: `${(depth+1)*14+8}px` }}>Loading…</div>`
          : ''}
      </div>`
    }
    return html`<div key=${e.path} class="flex items-center cursor-pointer px-2 py-0.5 ${e.path===selectedFile?'bg-[#37373d] text-white':'text-gray-400 hover:text-gray-200'} file-tree-item" style=${{ paddingLeft: `${depth*14+8}px` }} onClick=${() => onFileClick(e.path)}>
      <span class="text-xs mr-1 shrink-0">📄</span>
      <span class="text-xs truncate">${e.name}</span>
    </div>`
  })}`
}

interface MessageBubbleProps {
  msg: Message
  onApply: (path: string, code: string) => void
  appliedBlocks: CodeBlock[]
}

function MessageBubble({ msg, onApply, appliedBlocks }: MessageBubbleProps) {
  const parts = useMemo(() => {
    const res: Array<{ t: string; v?: string; lang?: string; path?: string; code?: string }> = []
    let rem = msg.content
    const re = /```(\w+(?::[^\n]*)?)\n([\s\S]*?)```/g
    let li = 0, m: RegExpExecArray | null
    while ((m = re.exec(rem)) !== null) {
      if (m.index > li) res.push({ t: 'text', v: rem.slice(li, m.index) })
      const h = m[1]; const ci = h.indexOf(':')
      res.push({ t: 'code', lang: ci>=0?h.slice(0,ci):h, path: ci>=0?h.slice(ci+1).trim() : '', code: m[2].trim() })
      li = m.index + m[0].length
    }
    if (li < rem.length) res.push({ t: 'text', v: rem.slice(li) })
    return res
  }, [msg.content])

  return html`
    <div class="flex ${msg.role==='user'?'justify-end':'justify-start'}">
      <div class="max-w-[92%] ${msg.role==='user'?'bg-blue-700 text-white rounded-2xl rounded-br-sm':'bg-[#2d2d2d] text-gray-200 rounded-2xl rounded-bl-sm'} px-3 py-2">
        ${msg.role==='assistant' ? html`<div class="text-[11px] text-gray-500 mb-1 font-medium">Coder</div>` : ''}
        <div class="text-xs leading-relaxed whitespace-pre-wrap break-words">
          ${parts.map((p: any, i: number) => {
            if (p.t === 'text') return html`<span key=${i}>${p.v}</span>`
            const bi = appliedBlocks.findIndex((b: CodeBlock) => b.path===p.path && b.code===p.code)
            const applied = bi >= 0 && appliedBlocks[bi].applied
            return html`<div key=${i} class="my-2 bg-[#1e1e1e] rounded overflow-hidden border border-[#3c3c3c]">
              <div class="flex items-center justify-between px-2 py-1 bg-[#252526] border-b border-[#3c3c3c]">
                <span class="text-[10px] text-gray-400 truncate">${p.path || p.lang}</span>
                ${p.path ? html`<button class="text-[10px] px-2 py-0.5 rounded shrink-0 ml-2 transition-colors ${applied?'bg-green-800 text-green-300':'bg-blue-600 hover:bg-blue-500 text-white'}" onClick=${() => !applied && onApply(p.path!, p.code!)} disabled=${applied}>${applied?'✓ Applied':'Apply'}</button>` : ''}
              </div>
              <pre class="text-[11px] p-2 overflow-x-auto scrollbar-thin"><code>${p.code}</code></pre>
            </div>`
          })}
          ${msg.isStreaming ? html`<span class="inline-block w-1.5 h-4 bg-blue-400 animate-pulse ml-0.5 align-text-bottom"></span>` : ''}
        </div>
      </div>
    </div>
  `
}

interface ChatPanelProps {
  messages: Message[]
  streamingContent: string
  codeBlocks: CodeBlock[]
  isGenerating: boolean
  onSend: (text: string) => void
  onStop: () => void
  onApply: (path: string, code: string) => void
  onClear: () => void
  modelLoaded: boolean
}

function ChatPanel({ messages, streamingContent, codeBlocks, isGenerating, onSend, onStop, onApply, onClear, modelLoaded }: ChatPanelProps) {
  const endRef = useRef<HTMLDivElement | null>(null)
  const [input, setInput] = useState('')

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, streamingContent, codeBlocks])

  const send = useCallback(() => {
    const t = input.trim()
    if (!t || isGenerating) return
    setInput('')
    onSend(t)
  }, [input, isGenerating, onSend])

  const kd = useCallback((e: KeyboardEvent) => { if (e.key==='Enter'&&!e.shiftKey) { e.preventDefault(); send() } }, [send])

  const all = useMemo(() => {
    const r = [...messages]
    if (streamingContent && (messages.length===0 || messages[messages.length-1]?.role!=='assistant')) {
      r.push({ role:'assistant', content: streamingContent, isStreaming: true })
    } else if (streamingContent && messages.length>0 && messages[messages.length-1]?.role==='assistant') {
      r[r.length-1] = { ...r[r.length-1], content: streamingContent, isStreaming: true }
    }
    return r
  }, [messages, streamingContent])

  return html`
    <div class="flex flex-col h-full">
      <div class="h-9 border-b border-[#3c3c3c] flex items-center px-3 shrink-0">
        <span class="text-xs font-medium text-gray-300">AI Chat</span>
        <button class="ml-auto text-xs text-gray-500 hover:text-gray-300" onClick=${onClear}>Clear</button>
      </div>
      <div class="flex-1 overflow-y-auto scrollbar-thin p-3 space-y-3">
        ${all.length===0 ? html`
          <div class="text-center text-gray-600 text-xs mt-8">
            <p class="mb-1">🤖</p>
            <p>Ask the AI to help with your code</p>
            ${!modelLoaded ? html`<p class="mt-2 text-yellow-600">Load a model first</p>` : ''}
          </div>
        ` : all.map((m: Message, i: number) => html`<${MessageBubble} key=${i} msg=${m} onApply=${onApply} appliedBlocks=${codeBlocks} />`)}
        ${isGenerating && !streamingContent ? html`<div class="flex items-center gap-1 text-gray-500 text-xs px-3 py-2"><span class="dot-pulse">●</span><span class="dot-pulse">●</span><span class="dot-pulse">●</span></div>` : ''}
        <div ref=${endRef} />
      </div>
      <div class="p-3 border-t border-[#3c3c3c] space-y-2">
        ${codeBlocks.filter(b => !b.applied).length > 0 ? html`
          <div class="space-y-1 max-h-20 overflow-y-auto scrollbar-thin">
            ${codeBlocks.filter(b => !b.applied).map((b, i) => html`
              <div key=${i} class="flex items-center justify-between bg-[#2d2d2d] rounded px-2 py-1 border border-[#3c3c3c]">
                <span class="text-[10px] text-gray-400 truncate">${b.path || b.lang}</span>
                <button class="text-[10px] px-2 py-0.5 bg-blue-600 hover:bg-blue-500 text-white rounded shrink-0 ml-2" onClick=${() => onApply(b.path, b.code)}>Apply</button>
              </div>
            `)}
          </div>
        ` : ''}
        <div class="flex gap-2">
          <textarea class="flex-1 bg-[#3c3c3c] text-gray-200 text-xs rounded px-2.5 py-1.5 border border-[#4c4c4c] outline-none focus:border-blue-500 resize-none placeholder-gray-500" rows=${2} placeholder=${modelLoaded?"Ask about your code...":"Load a model first..."} value=${input} onInput=${(e: Event)=>setInput((e.target as HTMLTextAreaElement).value)} onKeyDown=${kd} disabled=${!modelLoaded}></textarea>
          <div class="flex flex-col gap-1">
            <button class="px-3 py-1.5 text-xs rounded ${modelLoaded&&!isGenerating?'bg-blue-600 hover:bg-blue-700 text-white':'bg-[#3c3c3c] text-gray-500 cursor-not-allowed'}" onClick=${send} disabled=${!modelLoaded||isGenerating}>Send</button>
            ${isGenerating ? html`<button class="px-3 py-1.5 text-xs rounded bg-red-700 hover:bg-red-600 text-white" onClick=${onStop}>Stop</button>` : ''}
          </div>
        </div>
      </div>
    </div>
  `
}

function App() {
  const saved = loadState()
  const [treeData, setTreeData] = useState<FileEntry[] | null>(null)
  const [indexedCount, setIndexedCount] = useState(0)
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set())
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [selectedProvider, setSelectedProvider] = useState<Provider>((saved.selectedProvider as Provider) || PROVIDERS.WEBLLM)
  const [modelList, setModelList] = useState<ModelDescriptor[]>([])
  const [selectedModel, setSelectedModel] = useState(saved.selectedModel || '')
  const [modelLoaded, setModelLoaded] = useState(false)
  const [modelLoading, setModelLoading] = useState(false)
  const [initProgress, setInitProgress] = useState<InitProgressState | null>(null)
  const [messages, setMessages] = useState<Message[]>(saved.messages || [])
  const [streamingContent, setStreamingContent] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [codeBlocks, setCodeBlocks] = useState<CodeBlock[]>([])
  const [attachContext, setAttachContext] = useState(saved.attachContext ?? true)
  const [customModelUrl, setCustomModelUrl] = useState(saved.customModelUrl || '')
  const [fileStatus, setFileStatus] = useState('')
  const [notifs, setNotifs] = useState<Notification[]>([])

  useEffect(() => {
    const update = () => setNotifs([...notifications])
    notificationListeners.push(update)
    return () => {
      const idx = notificationListeners.indexOf(update)
      if (idx >= 0) notificationListeners.splice(idx, 1)
    }
  }, [])

  const editorRef = useRef<HTMLDivElement | null>(null)
  const abortRef = useRef(false)
  const generatingRef = useRef(false)

  useEffect(() => {
    saveState({ selectedProvider, selectedModel, customModelUrl, attachContext, messages })
  }, [selectedProvider, selectedModel, customModelUrl, attachContext, messages])

  useEffect(() => {
    (async () => {
      try {
        const vram = await estimateVRAM()
        const providerUsable = (p: Provider) =>
          p === PROVIDERS.WEBLLM ? window.__gpuAvailable
          : p === PROVIDERS.TRANSFORMERS_JS ? window.__transformersAvailable
          : p === PROVIDERS.BROWSER_AI ? window.__browserAiAvailable : false
        const models = getAvailableModels(vram).filter(m => providerUsable(m.provider))
        setModelList(models)
        // Fall back to a usable provider when nothing is saved, or when the
        // saved provider (e.g. Chrome Built-in AI on Chromium/Linux) is not
        // available in this browser.
        const savedUsable = saved.selectedModel && saved.selectedProvider && providerUsable(saved.selectedProvider as Provider)
        if (!savedUsable) {
          const preferred = getPreferredModel(models)
          if (preferred) {
            setSelectedModel(preferred.id)
            setSelectedProvider(preferred.provider)
          }
        }
      } catch {
        const fallback: ModelDescriptor[] = [
          { id: 'HuggingFaceTB/SmolLM2-360M-Instruct', vram: 512, provider: PROVIDERS.TRANSFORMERS_JS, label: 'SmolLM2 360M' },
          { id: 'Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC', vram: 1630, provider: PROVIDERS.WEBLLM, label: 'Qwen2.5 Coder 1.5B' },
          { id: 'Qwen3.5-2B-q4f16_1-MLC', vram: 2245, provider: PROVIDERS.WEBLLM, label: 'Qwen3.5 2B' },
          { id: 'Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC', vram: 5107, provider: PROVIDERS.WEBLLM, label: 'Qwen2.5 Coder 7B' },
        ]
        const usable = fallback.filter(m =>
          m.provider === PROVIDERS.WEBLLM ? window.__gpuAvailable
          : m.provider === PROVIDERS.TRANSFORMERS_JS ? window.__transformersAvailable
          : m.provider === PROVIDERS.BROWSER_AI ? window.__browserAiAvailable : false)
        setModelList(usable.length ? usable : fallback)
        const savedUsable = (saved.selectedProvider as Provider) === PROVIDERS.WEBLLM ? window.__gpuAvailable
          : (saved.selectedProvider as Provider) === PROVIDERS.TRANSFORMERS_JS ? window.__transformersAvailable
          : (saved.selectedProvider as Provider) === PROVIDERS.BROWSER_AI ? window.__browserAiAvailable : false
        if (!saved.selectedModel || !savedUsable) {
          const first = (usable.length ? usable : fallback)[0]
          setSelectedModel(first.id)
          setSelectedProvider(first.provider)
        }
      }
    })()
  }, [])

  useEffect(() => {
    initMonaco(editorRef.current!).then(() => {})
  }, [])

  const handleOpenFolder = useCallback(async () => {
    const entries = await openFolder(setIndexedCount)
    if (!entries) return
    setIndexedCount(fileHandles.size)
    for (const m of monacoModels.values()) m.dispose()
    monacoModels.clear()
    setTreeData(entries)
    setExpandedPaths(new Set())
    setSelectedFile(null)
    currentFilePath = null
    currentFileHandle = null
    setFileStatus('')
  }, [])

  const handleToggleDir = useCallback((entry: FileEntry) => {
    const path = entry.path
    setExpandedPaths((p: Set<string>) => { const n = new Set(p); n.has(path) ? n.delete(path) : n.add(path); return n })
    if (entry.type !== 'dir' || entry.loaded || loadingDirs.has(path)) return

    const generation = folderGeneration
    loadingDirs.add(path)
    readDirEntries(entry.handle as FileSystemDirectoryHandle, path, generation)
      .then(children => {
        if (generation !== folderGeneration) return
        setTreeData((prev: FileEntry[] | null) => updateEntry(prev, path, item => ({ ...item, children, loaded: true })))
      })
      .catch(err => {
        console.error('load dir:', path, err)
        addNotification(`Could not load ${path}`, 'error')
        setExpandedPaths(prev => { const next = new Set(prev); next.delete(path); return next })
      })
      .finally(() => loadingDirs.delete(path))
  }, [])

  const handleFileClick = useCallback(async (path: string) => {
    setSelectedFile(path)
    const ok = await loadFile(path)
    if (ok) setFileStatus(path)
  }, [])

  const handleSend = useCallback(async (text: string) => {
    if (!currentModel || !modelLoaded || generatingRef.current) return
    generatingRef.current = true
    abortController = new AbortController()
    const userMsg: Message = { role: 'user', content: text }
    const newMsgs = [...messages, userMsg]
    setMessages(newMsgs)
    setIsGenerating(true)
    setStreamingContent('')
    setCodeBlocks([])
    abortRef.current = false

    const fileTree = attachContext && treeData ? buildFileTree(treeData) : null
    const isChromeAI = selectedProvider === PROVIDERS.BROWSER_AI
    let plainSystem = fileTree ? `${SYSTEM_PROMPT}\n\nProject file tree:\n${fileTree}` : SYSTEM_PROMPT
    // Send the full conversation history on every turn: the Chrome harness
    // creates a fresh Prompt API session per turn, so history is applied once.
    const chatMsgs: ModelMessage[] = [
      ...newMsgs
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => ({ role: m.role, content: m.content } as ModelMessage)),
    ]

    let tools: HarnessTool[] | Record<string, any> | undefined
    if (attachContext) {
      tools = isChromeAI
        ? createHarnessTools({
            listFiles: async () => { await ensureIndexed(); return [...fileHandles.keys()] },
            readFile: readFileForTool,
            writeFile: applyFileChanges,
            createFile: createFileForTool,
            deleteFile: deleteFileForTool,
          })
        : createTools(readFileForTool, applyFileChanges, createFileForTool, deleteFileForTool)
    }

    // Without tools Chrome cannot inspect files, so keep the open file in the
    // prompt (also used by the harness if native tool use is unavailable).
    if (isChromeAI && !attachContext && currentFilePath && editor) {
      const model = editor.getModel()
      if (model) {
        plainSystem += `\n\nCurrent open file: ${currentFilePath}\n\`\`\`\n${model.getValue()}\n\`\`\``
      }
    }

    const systemMsg = attachContext
      ? (fileTree ? `${TOOL_SYSTEM_PROMPT}\n\nProject file tree:\n${fileTree}` : TOOL_SYSTEM_PROMPT)
      : plainSystem

    try {
      let full = ''
      for await (const chunk of streamChat(chatMsgs, systemMsg, plainSystem, tools, abortController.signal)) {
        if (abortRef.current) break
        full = chunk.text
        setStreamingContent(full)
        const bl = parseCodeBlocks(full)
        if (bl.length) setCodeBlocks(bl)
      }
      if (!abortRef.current) {
        if (!full.trim()) {
          full = 'The model returned an empty response. Try asking again, or switch to WebLLM for larger models.'
        }
        setMessages((prev: Message[]) => [...prev, { role: 'assistant', content: full }])
        setStreamingContent('')
        setCodeBlocks(parseCodeBlocks(full))
      }
    } catch (err) {
      console.error('Generation error:', err)
      let detail = (err as Error).message || (err as Error).toString() || 'Generation failed'
      if (selectedProvider === PROVIDERS.BROWSER_AI) {
        try {
          const status = await getBrowserAIStatus()
          if (status.status !== 'available' && status.status !== 'needs-download') detail = status.message
        } catch {}
      }
      setMessages((prev: Message[]) => [...prev, { role: 'assistant', content: `Error: ${detail}` }])
    } finally {
      generatingRef.current = false
      setIsGenerating(false)
      setStreamingContent('')
    }
  }, [messages, modelLoaded, currentModel, attachContext, treeData, selectedProvider])

  const handleStop = useCallback(() => { abortRef.current = true; stopGeneration() }, [])
  const handleClear = useCallback(() => { setMessages([]); setStreamingContent(''); setCodeBlocks([]); saveState({ messages: [] }) }, [])

  const handleApply = useCallback(async (path: string, code: string) => {
    const ok = await applyFileChanges(path, code)
    if (ok) {
      setCodeBlocks((prev: CodeBlock[]) => prev.map((b: CodeBlock) => (b.path===path && b.code===code) ? { ...b, applied: true } : b))
      if (path === currentFilePath) setFileStatus(path)
    }
  }, [])

  const handleLoadModel = useCallback(async () => {
    let modelId = selectedModel
    const provider = selectedProvider
    if (customModelUrl.trim()) {
      const parts = customModelUrl.trim().split('/')
      modelId = parts[parts.length-1] || customModelUrl.trim()
    }
    setModelLoading(true)
    setInitProgress({ text: `Preparing ${modelId}...`, progress: 0 })
    try {
      const model = await initLLM(provider, modelId, (r) => setInitProgress({ text: r.text||`Preparing ${modelId}...`, progress: r.progress||0 }))
      ;(model as any).provider = provider
      currentModel = model
      setModelLoaded(true)
      setInitProgress({ text: 'Model ready', progress: 1 })
      setTimeout(() => setInitProgress(null), 3000)
    } catch (err) {
      const msg = (err as Error)?.message || (err as Error)?.toString() || 'Model failed to load (check console for details)'
      console.error('Model load error:', err)
      let userMsg = `Error: ${msg}`
      if (provider === PROVIDERS.BROWSER_AI) {
        // Report the live Prompt API status — on Chromium/Linux this explains
        // that the on-device model simply is not shipped, instead of pointing
        // at chrome://components entries that do not exist there.
        try {
          userMsg = (await getBrowserAIStatus()).message
        } catch {
          userMsg = `Error: ${msg}`
        }
      } else if (msg.includes('not enough space')) {
        userMsg = 'Not enough storage for the model. Try a smaller model.'
      }
      setInitProgress({ text: userMsg, progress: 0 })
      setTimeout(() => setInitProgress(null), 10000)
    } finally { setModelLoading(false) }
  }, [selectedModel, selectedProvider, customModelUrl])

  const sidebarProps: SidebarProps = { treeData, indexedCount, expandedPaths, onToggleDir: handleToggleDir, onFileClick: handleFileClick, selectedFile, modelList, selectedModel, onModelChange: m => { setSelectedModel(m); setModelLoaded(false) }, modelLoaded, modelLoading, initProgress, onLoadModel: handleLoadModel, onOpenFolder: handleOpenFolder, attachContext, onAttachContextChange: setAttachContext, customModelUrl, onCustomModel: setCustomModelUrl, selectedProvider, onProviderChange: p => { if (p === PROVIDERS.BROWSER_AI && !window.__browserAiAvailable) return; setSelectedProvider(p); setSelectedModel(''); setModelLoaded(false); const models = getAvailableModels(4096); const first = models.find(m => m.provider === p); if (first) setSelectedModel(first.id) } }

  const chatProps: ChatPanelProps = { messages, streamingContent, codeBlocks, isGenerating, onSend: handleSend, onStop: handleStop, onApply: handleApply, onClear: handleClear, modelLoaded }

  return html`
    <div class="h-screen flex flex-col bg-[#1e1e1e] text-gray-200 text-sm">
      ${notifs.length > 0 ? html`
        <div class="fixed top-12 right-4 z-50 space-y-2 max-w-sm">
          ${notifs.map(n => html`
            <div key=${n.id} class="px-3 py-2 rounded text-xs border ${n.level === 'warning' ? 'bg-yellow-900/90 border-yellow-700 text-yellow-200' : n.level === 'error' ? 'bg-red-900/90 border-red-700 text-red-200' : 'bg-blue-900/90 border-blue-700 text-blue-200'}">
              <div class="font-medium mb-0.5">${n.level === 'warning' ? '⚠️ Warning' : n.level === 'error' ? '❌ Error' : 'ℹ️ Info'}</div>
              <div class="opacity-90">${n.text}</div>
            </div>
          `)}
        </div>
      ` : ''}
      <header class="h-9 bg-[#1e1e1e] border-b border-[#3c3c3c] flex items-center px-4 shrink-0">
        <span class="font-semibold text-gray-200 text-sm tracking-wide">Coder</span>
        <span class="text-xs text-gray-500 ml-3 hidden sm:inline">Local-First AI Editor</span>
        <div class="ml-auto flex items-center gap-3 text-xs text-gray-500">
          ${modelLoaded ? html`<span class="text-green-500">● Model Ready</span>` : ''}
          ${fileStatus ? html`<span class="text-gray-400 truncate max-w-[200px]" title=${fileStatus}>${fileStatus}</span>` : ''}
        </div>
      </header>
      <div class="flex-1 flex overflow-hidden">
        <div class="w-64 shrink-0 border-r border-[#3c3c3c] flex flex-col bg-[#252526] overflow-hidden">
          <${Sidebar} ...${sidebarProps} />
        </div>
        <div ref=${editorRef} class="flex-1 overflow-hidden relative bg-[#1e1e1e]">
          ${!treeData ? html`
            <div class="absolute inset-0 flex items-center justify-center bg-[#1e1e1e] z-10">
              <div class="text-center p-8">
                <div class="text-4xl mb-4">⚡</div>
                <p class="text-gray-400 text-sm mb-4">Open a local folder to start editing</p>
                <button class="px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors" onClick=${handleOpenFolder}>📂 Open Local Folder</button>
              </div>
            </div>
          ` : ''}
        </div>
        <div class="w-[380px] shrink-0 border-l border-[#3c3c3c] flex flex-col bg-[#1e1e1e] overflow-hidden hidden md:flex">
          <${ChatPanel} ...${chatProps} />
        </div>
      </div>
    </div>
  `
}

function hasWebGPU(): boolean {
  return !!navigator.gpu
}

async function hasBrowserAI(): Promise<boolean> {
  try {
    const detail = await getBrowserAIStatus()
    window.__browserAiNote = detail.short
    // 'needs-download' still counts as available: loading will trigger (and
    // show progress for) the download.
    return detail.status === 'available' || detail.status === 'needs-download'
  } catch {
    window.__browserAiNote = 'Chrome AI (not available)'
    return false
  }
}

async function checkAvailability() {
  window.__gpuAvailable = hasWebGPU()
  try {
    window.__transformersAvailable = await checkProviderAvailability(PROVIDERS.TRANSFORMERS_JS)
  } catch {
    window.__transformersAvailable = typeof WebAssembly !== 'undefined'
  }
  window.__browserAiAvailable = await hasBrowserAI()
}

checkAvailability().then(() => {
  render(html`<${App} />`, document.getElementById('root')!)
})
