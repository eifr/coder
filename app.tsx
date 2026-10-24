import { html, render, Component, useState, useEffect, useRef, useCallback, useMemo } from 'htm/preact/standalone.mjs'
import { PROVIDERS, getAvailableModels, estimateVRAM, getPreferredModel, checkProviderAvailability, getBrowserAIStatus, getBuiltInAICapabilities, initProviderModel, type Provider, type ModelDescriptor, type BuiltInApiInfo, type BuiltInApiStatus } from './llm-provider'
import { createHarnessTools, type HarnessTool, type HarnessDiagnostic, type HarnessOpenFile, type HarnessSelection } from './chrome-harness'
import { readRecentCommits, type RecentCommit } from './git-log'
import { createLocalModels } from './pi-local-provider'
import { Agent, type AgentMessage, type AgentTool } from '@earendil-works/pi-agent-core'
import { Type, type Model, type Models } from '@earendil-works/pi-ai'
import { CHROME_API, MODEL_ID, PROVIDER_ID, createChromeModels } from './pi-chrome-provider'
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
  const id = ++notificationId
  notifications.push({ id, text, level })
  notificationListeners.forEach(fn => fn())
  setTimeout(() => {
    const idx = notifications.findIndex(n => n.id === id)
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
  /** Human-readable loading state, e.g. while the on-device model prepares. */
  status?: string
  tool?: { name: string; path?: string; from?: string; ok?: boolean }
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
// Bump this when debugging a stale PWA/dev bundle; shown in the Model tab.
const APP_BUILD = 'chrome-fallback-4'

interface PersistedState {
  selectedProvider: Provider
  selectedModel: string
  customModelUrl: string
  attachContext: boolean
  messages: Message[]
  openFiles: string[]
  selectedFile: string | null
  projectName: string
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

function loadWidth(key: string, fallback: number): number {
  try {
    const raw = localStorage.getItem(key)
    const value = raw ? Number(raw) : NaN
    return Number.isFinite(value) ? value : fallback
  } catch {
    return fallback
  }
}

const EXCLUDED = new Set(['node_modules','.git','dist','.DS_Store','__pycache__','.next','.vscode','build','.cache','.venv','venv','env','.idea','coverage','.nyc_output'])
const MAX_DEPTH = 8
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

/* Inline icon set. Stroke icons inherit `currentColor`; only the mark keeps
 * fixed brand colors (it mirrors public/icon.svg). */
function Logo({ class: cls = 'w-6 h-6' }: { class?: string }) {
  return html`<svg viewBox="0 0 32 32" class=${cls} aria-hidden="true">
    <rect x="0.75" y="0.75" width="30.5" height="30.5" rx="7.25" fill="#13161d" stroke="#343d4d" stroke-width="1.5" />
    <path d="M13 10.5 7.5 16 13 21.5" fill="none" stroke="#e4e8f0" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" />
    <path d="M19 10.5 24.5 16 19 21.5" fill="none" stroke="#e4e8f0" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" />
    <rect x="14.7" y="11" width="2.6" height="10" rx="1.3" fill="#f4b860" />
  </svg>`
}

function Chevron({ open, class: cls = 'w-3 h-3' }: { open: boolean; class?: string }) {
  return html`<svg viewBox="0 0 16 16" class=${`${cls} shrink-0 transition-transform duration-150 ${open ? 'rotate-90' : ''}`} fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 3.5 10.5 8 6 12.5" /></svg>`
}

function FolderIcon({ open = false, class: cls = 'w-3.5 h-3.5' }: { open?: boolean; class?: string }) {
  const d = open
    ? 'm6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2'
    : 'M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z'
  return html`<svg viewBox="0 0 24 24" class=${cls} fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d=${d} /></svg>`
}

function FileIcon({ class: cls = 'w-3.5 h-3.5' }: { class?: string }) {
  return html`<svg viewBox="0 0 24 24" class=${cls} fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6" /></svg>`
}

function FolderPlusIcon({ class: cls = 'w-3.5 h-3.5' }: { class?: string }) {
  return html`<svg viewBox="0 0 24 24" class=${cls} fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z" /><path d="M12 11v6M9 14h6" /></svg>`
}

function CheckIcon({ class: cls = 'w-3 h-3' }: { class?: string }) {
  return html`<svg viewBox="0 0 24 24" class=${cls} fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m5 12 4 4L19 6" /></svg>`
}

function ChatIcon({ class: cls = 'w-3.5 h-3.5' }: { class?: string }) {
  return html`<svg viewBox="0 0 24 24" class=${cls} fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></svg>`
}

function SparkIcon({ class: cls = 'w-3.5 h-3.5' }: { class?: string }) {
  return html`<svg viewBox="0 0 24 24" class=${cls} fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3.5 13.9 9l5.6 1.9-5.6 1.9L12 18.4l-1.9-5.6L4.5 10.9 10.1 9z" /><path d="M19 15.5v3M20.5 17h-3" /></svg>`
}

function SearchIcon({ class: cls = 'w-3.5 h-3.5' }: { class?: string }) {
  return html`<svg viewBox="0 0 24 24" class=${cls} fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="m20 20-3.6-3.6" /></svg>`
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
- listFiles / searchFiles: find files and symbols
- findFiles: find files by glob pattern, e.g. src/**/*.ts
- readFile: read a file (required before any edit)
- getOpenFile / getSelection: read the open editor file or selected text
- getDiagnostics: current editor errors and warnings
- createFile: create a new file (fails if it already exists)
- makeDirectory: create a directory
- moveFile / copyFile: rename or copy a file (read it first)
- replaceInFile: replace one exact, unique snippet (preferred edit)
- applyPatch: apply multiple replacements across files in one call (read each file first)
- editFile: rewrite a whole file (last resort)
- deleteFile: remove a file (only when explicitly asked)

Work carefully, in this order:
1. Understand the request. If it is ambiguous, or the change would remove data, ask a short question first instead of guessing.
2. Locate the relevant files with listFiles/searchFiles.
3. Read every file you intend to change with readFile. Never edit a file you have not read in this turn, never reconstruct code from memory, and never invent APIs.
4. Make the smallest change that satisfies the request. Prefer replaceInFile with a unique snippet. Keep unrelated code, formatting, imports, and comments exactly as they are.
5. After each edit, re-read the changed range to confirm it matches the intent.
6. Only call deleteFile when the user explicitly asked to delete that file.
7. If a tool returns an error, adapt; do not repeat the same call unchanged.

Rules:
- Paths are relative to the project root. Never use absolute paths or "..".
- Never modify .git or node_modules.
- Do not refactor, rename, or reformat code the request did not ask about.
- If you run low on context or are unsure, stop editing and explain what is left.
- When done, list exactly which files changed and what changed in each.`

let editor: any = null
let editorReadyPromise: Promise<void> | null = null
let currentFilePath: string | null = null
let currentFileHandle: FileSystemFileHandle | null = null
let rootDirHandle: FileSystemDirectoryHandle | null = null
let rootDirName = ''
const fileHandles = new Map<string, FileEntry>()
const monacoModels = new Map<string, any>()
let currentModel: LanguageModel | null = null
let abortController: AbortController | null = null
let modelLoadPromise: Promise<LanguageModel> | null = null
let onFileSaved: ((path: string) => void) | null = null
let onFileOpened: ((path: string) => void) | null = null
let onFileDirtyChanged: ((path: string, dirty: boolean) => void) | null = null
let onRecentCommits: ((commits: RecentCommit[]) => void) | null = null
const savedContents = new Map<string, string>()

interface EditorRange {
  startLineNumber: number
  startColumn: number
  endLineNumber: number
  endColumn: number
}

interface BuiltInActionRequest {
  kind: 'summarize' | 'summarize-file' | 'translate' | 'proofread' | 'rewrite'
  text: string
  range?: EditorRange
  path?: string
}

type SelectionActionKind = 'explain' | 'refactor' | 'tests'
let onSelectionAction: ((kind: SelectionActionKind, text: string, path: string | null) => void) | null = null

let onBuiltInAction: ((request: BuiltInActionRequest) => void) | null = null

function hasBuiltInApi(name: string): boolean {
  const globals = globalThis as unknown as Record<string, unknown>
  return typeof globals[name] === 'function'
}

function replaceEditorRange(range: EditorRange | undefined, text: string): void {
  if (!editor || !range) return
  editor.executeEdits('coder-built-in-ai', [{ range, text, forceMoveMarkers: true }])
  editor.focus()
}

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

const compareEntries = (a: FileEntry, b: FileEntry): number => {
  if (a.type !== b.type) return a.type === 'dir' ? -1 : 1
  return a.name.localeCompare(b.name)
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
  entries.sort(compareEntries)
  return entries
}

async function readExpandedTree(
  handle: FileSystemDirectoryHandle,
  expanded: Set<string>,
  generation: number
): Promise<FileEntry[]> {
  const entries = await readDirEntries(handle, '', generation)
  return expandTreeEntries(entries, expanded, generation)
}

async function expandTreeEntries(
  entries: FileEntry[],
  expanded: Set<string>,
  generation: number
): Promise<FileEntry[]> {
  const result: FileEntry[] = []
  for (const entry of entries) {
    if (entry.type !== 'dir' || !expanded.has(entry.path)) {
      result.push(entry)
      continue
    }
    try {
      // entry.type === 'dir' guarantees a directory handle.
      const dir = entry.handle as FileSystemDirectoryHandle
      const children = await readDirEntries(dir, entry.path, generation)
      result.push({ ...entry, children: await expandTreeEntries(children, expanded, generation), loaded: true })
    } catch {
      result.push(entry)
    }
  }
  return result
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

/** Rejects model-supplied paths that escape the opened project. */
function safeProjectPath(path: string): string | null {
  const slashed = String(path ?? '').replace(/\\/g, '/').trim()
  if (!slashed || slashed.startsWith('/') || /^[a-zA-Z]:\//.test(slashed)) return null
  const segments = slashed.split('/').filter(segment => segment !== '' && segment !== '.')
  if (segments.length === 0 || segments.includes('..')) return null
  if (segments.some(segment => segment === '.git' || segment === 'node_modules')) return null
  return segments.join('/')
}

async function resolveFileHandle(path: string): Promise<FileSystemFileHandle | null> {
  const safe = safeProjectPath(path)
  if (!safe) return null
  path = safe
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

const WORKSPACE_DB = 'coder-workspace'
const WORKSPACE_STORE = 'workspace'
const WORKSPACE_KEY = 'root'

/**
 * `FileSystemDirectoryHandle`s are structured-cloneable, so IndexedDB can keep
 * the last project across reloads. Permission itself is not persisted: Chrome
 * may require a click to re-grant it, which the sidebar prompts for.
 */
function openWorkspaceDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(WORKSPACE_DB, 1)
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(WORKSPACE_STORE)) {
        request.result.createObjectStore(WORKSPACE_STORE)
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

async function storeRootHandle(handle: FileSystemDirectoryHandle): Promise<void> {
  try {
    const db = await openWorkspaceDb()
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(WORKSPACE_STORE, 'readwrite')
      tx.objectStore(WORKSPACE_STORE).put(handle, WORKSPACE_KEY)
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
      tx.onabort = () => reject(tx.error)
    })
    db.close()
  } catch (err) {
    console.warn('storeRootHandle:', err)
  }
}

async function readStoredRootHandle(): Promise<FileSystemDirectoryHandle | null> {
  try {
    const db = await openWorkspaceDb()
    const handle = await new Promise<FileSystemDirectoryHandle | null>((resolve, reject) => {
      const tx = db.transaction(WORKSPACE_STORE, 'readonly')
      const request = tx.objectStore(WORKSPACE_STORE).get(WORKSPACE_KEY)
      request.onsuccess = () => resolve((request.result as FileSystemDirectoryHandle | undefined) ?? null)
      request.onerror = () => reject(request.error)
    })
    db.close()
    return handle
  } catch (err) {
    console.warn('readStoredRootHandle:', err)
    return null
  }
}

async function clearStoredRootHandle(): Promise<void> {
  try {
    const db = await openWorkspaceDb()
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(WORKSPACE_STORE, 'readwrite')
      tx.objectStore(WORKSPACE_STORE).delete(WORKSPACE_KEY)
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
      tx.onabort = () => reject(tx.error)
    })
    db.close()
  } catch (err) {
    console.warn('clearStoredRootHandle:', err)
  }
}

async function activateFolder(
  handle: FileSystemDirectoryHandle,
  onIndexProgress?: (indexedFiles: number) => void
): Promise<FileEntry[] | null> {
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
  onRecentCommits?.([])
  void readRecentCommits(handle).then(commits => {
    if (generation === folderGeneration) onRecentCommits?.(commits)
  })
  return entries
}

async function openFolder(onIndexProgress?: (indexedFiles: number) => void): Promise<FileEntry[] | null> {
  try {
    const handle = await (window as any).showDirectoryPicker()
    const entries = await activateFolder(handle, onIndexProgress)
    if (entries) void storeRootHandle(handle)
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
    savedContents.set(path, content)
    if (monacoModels.has(path)) {
      const m = monacoModels.get(path)
      if (m.getValue() !== content) m.setValue(content)
      editor.setModel(m)
    } else {
      const model = monaco.editor.createModel(content, getLanguage(path))
      model.onDidChangeContent(() => {
        onFileDirtyChanged?.(path, savedContents.get(path) !== model.getValue())
      })
      monacoModels.set(path, model)
      editor.setModel(model)
    }
    onFileOpened?.(path)
    onFileDirtyChanged?.(path, false)
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
    savedContents.set(targetPath, editor.getValue())
    onFileDirtyChanged?.(targetPath, false)
    onFileSaved?.(targetPath)
    return true
  } catch (err) { console.error('saveFile:', err); return false }
}

async function applyFileChanges(path: string, code: string): Promise<boolean> {
  const safe = safeProjectPath(path)
  if (!safe) return false
  path = safe
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
    savedContents.set(path, code)
    if (monacoModels.has(path)) {
      const model = monacoModels.get(path)
      if (model !== editor.getModel()) model.setValue(code)
      else editor.setValue(code)
    } else if (path === currentFilePath) {
      editor.setValue(code)
    }
    onFileDirtyChanged?.(path, false)
    return true
  } catch { return false }
}

function initMonaco(container: HTMLElement): Promise<monaco.editor.IStandaloneCodeEditor> {
  const { promise, resolve } = Promise.withResolvers<monaco.editor.IStandaloneCodeEditor>()
  monaco.editor.defineTheme('coder-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [{ token: '', foreground: 'e4e8f0' }],
    colors: {
      'editor.background': '#0e1015',
      'editor.foreground': '#e4e8f0',
      'editorLineNumber.foreground': '#4b5468',
      'editorLineNumber.activeForeground': '#9aa4b8',
      'editor.lineHighlightBackground': '#151922',
      'editor.selectionBackground': '#2a3550',
      'editorCursor.foreground': '#f4b860',
      'editorIndentGuide.background1': '#1d222c',
      'editorGutter.background': '#0e1015',
      'minimap.background': '#0e1015',
    },
  })
  const ed = monaco.editor.create(container, {
    value: '', language: 'plaintext', theme: 'coder-dark',
    automaticLayout: true, minimap: { enabled: false }, fontSize: 13,
    lineNumbers: 'on', scrollBeyondLastLine: false, tabSize: 2,
    wordWrap: 'off', padding: { top: 10 },
    bracketPairColorization: { enabled: true },
  })
  ed.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, async () => { await saveFile() })

  const readSelection = (): { text: string; range: EditorRange } | null => {
    const model = ed.getModel()
    const selection = ed.getSelection()
    if (!model || !selection || selection.isEmpty()) return null
    return {
      text: model.getValueInRange(selection),
      range: {
        startLineNumber: selection.startLineNumber,
        startColumn: selection.startColumn,
        endLineNumber: selection.endLineNumber,
        endColumn: selection.endColumn,
      },
    }
  }

  // These actions only exist when the browser exposes the task API, so the
  // context menu never advertises an action that cannot run.
  if (hasBuiltInApi('Summarizer')) {
    ed.addAction({
      id: 'coder.summarizeSelection',
      label: '✦ Summarize selection',
      precondition: 'editorHasSelection',
      contextMenuGroupId: '1_modification',
      contextMenuOrder: 1.5,
      run: () => {
        const selection = readSelection()
        if (selection) onBuiltInAction?.({ kind: 'summarize', ...selection })
      },
    })
  }
  if (hasBuiltInApi('Translator')) {
    ed.addAction({
      id: 'coder.translateSelection',
      label: '✦ Translate selection',
      precondition: 'editorHasSelection',
      contextMenuGroupId: '1_modification',
      contextMenuOrder: 1.6,
      run: () => {
        const selection = readSelection()
        if (selection) onBuiltInAction?.({ kind: 'translate', ...selection })
      },
    })
  }
  if (hasBuiltInApi('Proofreader')) {
    ed.addAction({
      id: 'coder.proofreadSelection',
      label: '✦ Proofread selection',
      precondition: 'editorHasSelection',
      contextMenuGroupId: '1_modification',
      contextMenuOrder: 1.7,
      run: () => {
        const selection = readSelection()
        if (selection) onBuiltInAction?.({ kind: 'proofread', ...selection })
      },
    })
  }
  if (hasBuiltInApi('Rewriter')) {
    ed.addAction({
      id: 'coder.rewriteSelection',
      label: '✦ Rewrite selection',
      precondition: 'editorHasSelection',
      contextMenuGroupId: '1_modification',
      contextMenuOrder: 1.8,
      run: () => {
        const selection = readSelection()
        if (selection) onBuiltInAction?.({ kind: 'rewrite', ...selection })
      },
    })
  }

  // Model-agnostic actions: they send the selection to the chat, so they work
  // with WebLLM, Transformers.js, and Chrome built-in AI alike.
  const selectionActions: Array<{ id: string; label: string; kind: SelectionActionKind; order: number }> = [
    { id: 'coder.explainSelection', label: '✦ Explain selection', kind: 'explain', order: 1.9 },
    { id: 'coder.refactorSelection', label: '✦ Refactor selection', kind: 'refactor', order: 2.0 },
    { id: 'coder.testSelection', label: '✦ Add tests for selection', kind: 'tests', order: 2.1 },
  ]
  for (const action of selectionActions) {
    ed.addAction({
      id: action.id,
      label: action.label,
      precondition: 'editorHasSelection',
      // A distinct group makes Monaco draw a separator before the
      // model-backed actions, after the built-in AI task actions.
      contextMenuGroupId: '2_coder',
      contextMenuOrder: action.order,
      run: () => {
        const selection = readSelection()
        if (selection) onSelectionAction?.(action.kind, selection.text, currentFilePath)
      },
    })
  }

  editor = ed
  resolve(ed)
  return promise
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

const CHROME_MODEL = createChromeModels()
const MAX_AGENT_TURNS = 8
const MAX_AGENT_TOOL_RESULT = 24000
const MUTATING_TOOLS = new Set(['createFile', 'editFile', 'replaceInFile', 'deleteFile', 'makeDirectory', 'moveFile', 'copyFile', 'applyPatch'])

function modelContentText(content: unknown): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content.map(part => {
    if (!part || typeof part !== 'object') return ''
    if ('type' in part && part.type === 'text' && 'text' in part && typeof part.text === 'string') return part.text
    return ''
  }).join('')
}

function toAgentHistory(messages: ModelMessage[]): AgentMessage[] {
  const history: AgentMessage[] = []
  for (const message of messages) {
    if (message.role !== 'user' && message.role !== 'assistant') continue
    const text = modelContentText(message.content)
    if (message.role === 'user') {
      history.push({ role: 'user', content: text, timestamp: Date.now() })
      continue
    }
    history.push({
      role: 'assistant',
      content: [{ type: 'text', text }],
      api: CHROME_API,
      provider: PROVIDER_ID,
      model: MODEL_ID,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: 'stop',
      timestamp: Date.now(),
    })
  }
  return history
}

function toAgentTool(tool: HarnessTool): AgentTool {
  return {
    name: tool.name,
    label: tool.name,
    description: tool.description,
    parameters: Type.Unsafe(tool.inputSchema),
    execute: async (_toolCallId, params) => {
      // pi validated the arguments against the same JSON Schema the app declares.
      const args = params as Record<string, unknown>
      const value = await tool.execute(args)
      const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2) ?? 'null'
      return {
        content: [{
          type: 'text',
          text: text.length > MAX_AGENT_TOOL_RESULT
            ? `${text.slice(0, MAX_AGENT_TOOL_RESULT)}\n…[truncated ${text.length - MAX_AGENT_TOOL_RESULT} characters]`
            : text,
        }],
        details: undefined,
      }
    },
  }
}

function toolArgPath(args: unknown, key: string): string | undefined {
  if (!args || typeof args !== 'object' || !(key in args)) return undefined
  // `key in args` narrows to object; the record view reads the dynamic key.
  const record = args as Record<string, unknown>
  const value = record[key]
  if (typeof value !== 'string') return undefined
  return value.replace(/\\/g, '/').replace(/^\.?\//, '').replace(/\/+$/, '')
}

interface ChunkQueue {
  push(chunk: ChatChunk): void
  close(): void
  drain(): AsyncGenerator<ChatChunk>
}

/**
 * Bridges pi's subscription callbacks to the async generator the chat UI
 * consumes. `push` is called from agent events; `drain` is called by streamChat.
 */
function createChunkQueue(): ChunkQueue {
  const chunks: ChatChunk[] = []
  let wake: (() => void) | null = null
  let closed = false
  return {
    push(chunk) {
      chunks.push(chunk)
      wake?.()
      wake = null
    },
    close() {
      closed = true
      wake?.()
      wake = null
    },
    async *drain() {
      while (true) {
        const chunk = chunks.shift()
        if (chunk) {
          yield chunk
          continue
        }
        if (closed) return
        // Promise.withResolvers() is ES2024; this app targets ES2020.
        await new Promise<void>(resolve => { wake = resolve })
      }
    },
  }
}

async function* streamPiAgent(
  models: Models,
  model: Model<any>,
  messages: ModelMessage[],
  system: string | undefined,
  tools: HarnessTool[],
  signal?: AbortSignal
): AsyncGenerator<ChatChunk> {

  const chat = messages.filter(message => message.role === 'user' || message.role === 'assistant')
  const last = chat[chat.length - 1]
  if (!last) throw new Error('No user message to send.')
  const prompt = modelContentText(last.content)

  const queue = createChunkQueue()
  const toolPaths = new Map<string, string>()
  const toolFroms = new Map<string, string>()
  let visible = ''
  let toolCalls = 0
  let turns = 0
  let lastToolResult = ''
  let finished = false
  let cleaned = false
  let failure: Error | null = null

  const agent = new Agent({
    initialState: {
      systemPrompt: system || '',
      model,
      tools: tools.map(toAgentTool),
      messages: toAgentHistory(chat.slice(0, -1)),
    },
    streamFn: models.streamSimple.bind(models),
    convertToLlm: messages => messages,
    toolExecution: 'sequential',
    finishTurn: async () => {
      turns++
      return turns >= MAX_AGENT_TURNS ? { action: 'end' as const } : undefined
    },
  })

  const unsubscribe = agent.subscribe(event => {
    if (event.type === 'message_update') {
      const update = event.assistantMessageEvent
      if (update.type === 'text_delta') {
        visible += update.delta
        queue.push({ text: visible, done: false, delta: update.delta, usage: null })
      } else if (update.type === 'toolcall_end') {
        toolCalls++
        console.log('Tool call:', update.toolCall.name, update.toolCall.arguments)
      }
      return
    }
    if (event.type === 'tool_execution_start') {
      const path = toolArgPath(event.args, 'path') ?? toolArgPath(event.args, 'to')
      const from = toolArgPath(event.args, 'from')
      toolPaths.set(event.toolCallId, path || '')
      if (from) toolFroms.set(event.toolCallId, from)
      queue.push({
        text: visible,
        done: false,
        delta: '',
        usage: null,
        status: `Running ${event.toolName}…`,
        tool: { name: event.toolName, path, from },
      })
      return
    }
    if (event.type === 'tool_execution_end') {
      const path = toolPaths.get(event.toolCallId)
      const from = toolFroms.get(event.toolCallId)
      lastToolResult = event.result.content
        .map((part: { type: string; text?: string }) => part.type === 'text' ? part.text ?? '' : '')
        .join('')
      console.log('Tool result:', event.toolName, event.isError ? 'error' : 'ok')
      queue.push({
        text: visible,
        done: false,
        delta: '',
        usage: null,
        tool: { name: event.toolName, path, from, ok: !event.isError },
      })
    }
  })

  const onAbort = () => agent.abort()
  signal?.addEventListener('abort', onAbort)
  const cleanup = () => {
    if (cleaned) return
    cleaned = true
    unsubscribe()
    signal?.removeEventListener('abort', onAbort)
  }

  const run = agent.prompt(prompt).catch(error => {
    failure = error instanceof Error ? error : new Error(String(error))
  }).finally(() => {
    finished = true
    const resultPreview = lastToolResult.length > 1200
      ? `${lastToolResult.slice(0, 1200)}\n…[truncated]`
      : lastToolResult
    const agentError = agent.state.errorMessage
    if (!visible && agentError) console.error('[coder] agent ended without content:', agentError)
    const fallback = agentError
      ? `Error: ${agentError}`
      : toolCalls > 0
        ? `The model completed ${toolCalls} tool call${toolCalls === 1 ? '' : 's'} without a final summary.${resultPreview ? `\n\nLast tool result:\n${resultPreview}` : ''}`
        : ''
    queue.push({ text: visible || fallback, done: true, delta: '', usage: null })
    queue.close()
    cleanup()
  })
  void run

  try {
    for await (const chunk of queue.drain()) yield chunk
    if (failure && !signal?.aborted) throw failure
  } finally {
    if (!finished) agent.abort()
    cleanup()
  }
}

async function* streamChat(messages: ModelMessage[], system?: string, tools?: HarnessTool[], signal?: AbortSignal): AsyncGenerator<ChatChunk> {
  if (!currentModel) throw new Error('Model not initialized')
  const provider = (currentModel as any).provider || PROVIDERS.WEBLLM
  console.info('[coder] streamChat', APP_BUILD, provider)

  if (provider === PROVIDERS.BROWSER_AI) {
    const model = CHROME_MODEL.getModel(PROVIDER_ID, MODEL_ID)
    if (!model) throw new Error('Chrome Built-in AI model is not registered.')
    yield* streamPiAgent(CHROME_MODEL, model, messages, system, tools || [], signal)
    return
  }

  const modelId = (currentModel as any).modelId || 'local'
  const localModels = createLocalModels({ provider, model: currentModel, modelId })
  const model = localModels.getModels()[0]
  if (!model) throw new Error('Local model is not registered.')
  yield* streamPiAgent(localModels, model, messages, system, tools || [], signal)
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

/** Adds a file written by a tool to the visible tree when its parent is loaded. */
function addFileToTree(entries: FileEntry[] | null, path: string, entry: FileEntry): FileEntry[] | null {
  if (!entries) return null
  const parts = path.split('/')
  const walk = (list: FileEntry[], depth: number): FileEntry[] => {
    const name = parts[depth]
    const leaf = depth === parts.length - 1
    if (leaf) {
      if (list.some(candidate => candidate.path === path)) return list
      return [...list, entry].sort(compareEntries)
    }
    const index = list.findIndex(candidate => candidate.name === name && candidate.type === 'dir')
    if (index < 0) return list
    const dir = list[index]
    if (!dir.loaded || !dir.children) return list
    const children = walk(dir.children, depth + 1)
    if (children === dir.children) return list
    const next = list.slice()
    next[index] = { ...dir, children }
    return next
  }
  return walk(entries, 0)
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

function collectEditorDiagnostics(): HarnessDiagnostic[] {
  const diagnostics: HarnessDiagnostic[] = []
  for (const [path, model] of monacoModels) {
    for (const marker of monaco.editor.getModelMarkers({ resource: model.uri })) {
      diagnostics.push({
        path,
        severity: marker.severity === monaco.MarkerSeverity.Error ? 'error'
          : marker.severity === monaco.MarkerSeverity.Warning ? 'warning'
          : marker.severity === monaco.MarkerSeverity.Info ? 'info'
          : 'hint',
        message: marker.message,
        line: marker.startLineNumber,
        column: marker.startColumn,
      })
    }
  }
  return diagnostics.slice(0, 200)
}

function readOpenFileForTool(): HarnessOpenFile | null {
  if (!currentFilePath || !editor) return null
  const model = editor.getModel()
  if (!model) return null
  return { path: currentFilePath, content: model.getValue() }
}

function readSelectionForTool(): HarnessSelection | null {
  if (!editor) return null
  const model = editor.getModel()
  const selection = editor.getSelection()
  if (!model || !selection) return null
  return {
    path: currentFilePath,
    text: model.getValueInRange(selection),
    startLine: selection.startLineNumber,
    endLine: selection.endLineNumber,
  }
}

async function createFileForTool(path: string, content: string): Promise<boolean> {
  const safe = safeProjectPath(path)
  if (!safe) return false
  path = safe
  if (fileHandles.has(path)) return false
  if (!rootDirHandle) return false
  const parts = path.split('/')
  let dirHandle: FileSystemDirectoryHandle = rootDirHandle
  for (let i = 0; i < parts.length - 1; i++) {
    try { dirHandle = await dirHandle.getDirectoryHandle(parts[i], { create: true }) } catch { return false }
  }
  const name = parts[parts.length - 1]
  // Ask for the file without `create` first: with lazy indexing a file that
  // exists on disk may not be in fileHandles yet, and `create: true` would
  // silently return and overwrite it.
  try {
    await dirHandle.getFileHandle(name)
    return false
  } catch (err) {
    if (!(err instanceof DOMException) || err.name !== 'NotFoundError') return false
  }
  try {
    const fh = await dirHandle.getFileHandle(name, { create: true })
    await writeFileContent(fh, content)
    fileHandles.set(path, { handle: fh, name, path, type: 'file' })
    return true
  } catch { return false }
}

async function createDirectoryForTool(path: string): Promise<boolean> {
  const safe = safeProjectPath(path)
  if (!safe || !rootDirHandle) return false
  try {
    let dir: FileSystemDirectoryHandle = rootDirHandle
    for (const part of safe.split('/')) dir = await dir.getDirectoryHandle(part, { create: true })
    return true
  } catch {
    return false
  }
}

async function deleteFileForTool(path: string): Promise<boolean> {
  const safe = safeProjectPath(path)
  if (!safe) return false
  path = safe
  if (!rootDirHandle) return false
  try {
    const parts = path.split('/')
    let dirHandle: FileSystemDirectoryHandle = rootDirHandle
    for (let i = 0; i < parts.length - 1; i++) {
      dirHandle = await dirHandle.getDirectoryHandle(parts[i])
    }
    await dirHandle.removeEntry(parts[parts.length - 1])
    fileHandles.delete(path)
    if (path === currentFilePath) {
      currentFilePath = null
      currentFileHandle = null
      // Detach before disposing: the editor must not keep a disposed model.
      editor?.setModel(null)
    }
    if (monacoModels.has(path)) {
      monacoModels.get(path).dispose()
      monacoModels.delete(path)
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
  onOpenFolder: () => void
  restorableProject: string | null
  onRestoreProject: () => void
  recentCommits: RecentCommit[]
  onRefreshCommits: () => void
}

function Sidebar({ treeData, indexedCount, expandedPaths, onToggleDir, onFileClick, selectedFile, onOpenFolder, restorableProject, onRestoreProject, recentCommits, onRefreshCommits }: SidebarProps) {
  const visibleCount = useMemo(() => {
    if (!treeData) return 0
    let n = 0
    function w(e: FileEntry[]) { for (const x of e) { if (x.type === 'file') n++; if (x.children) w(x.children) } }
    w(treeData)
    return n
  }, [treeData])

  return html`
    <div class="flex flex-col h-full">
      <div class="px-3 pt-3 pb-2.5 border-b border-line space-y-2">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-semibold uppercase tracking-[0.16em] text-faint">Project</span>
          ${treeData ? html`<span class="font-mono text-[10px] text-faint">${Math.max(visibleCount, indexedCount)} file${Math.max(visibleCount, indexedCount) === 1 ? '' : 's'}</span>` : ''}
        </div>
        ${treeData ? html`<div class="flex items-center gap-1.5 text-muted"><${FolderIcon} open=${true} /><span class="truncate text-[11px]">${rootDirName}</span></div>` : ''}
        <button class="w-full px-3 py-1.5 bg-raised hover:bg-hover text-ink text-[11.5px] rounded-md border border-line hover:border-line-strong flex items-center justify-center gap-1.5 transition-colors" onClick=${onOpenFolder}>
          <${FolderPlusIcon} class="w-3.5 h-3.5 text-muted" />
          <span>${treeData ? 'Change folder' : 'Open folder'}</span>
        </button>
        ${!treeData && restorableProject ? html`
          <button class="w-full px-3 py-1.5 bg-accent/10 hover:bg-accent/15 text-accent text-[11.5px] rounded-md border border-accent/30 flex items-center justify-center gap-1.5 transition-colors" onClick=${onRestoreProject} title="Reopen the last project">
            <${FolderIcon} class="w-3.5 h-3.5" />
            <span class="truncate">Reopen ${restorableProject}</span>
          </button>
        ` : ''}
      </div>
      <div class="flex-1 overflow-y-auto py-1.5">
        ${treeData ? html`<${FileTree} entries=${treeData} expandedPaths=${expandedPaths} onToggle=${onToggleDir} onFileClick=${onFileClick} selectedFile=${selectedFile} depth=${0} />`
        : html`<div class="px-4 py-10 text-center">
            <${FolderIcon} class="w-6 h-6 mx-auto text-faint" />
            <p class="mt-2.5 text-[11.5px] text-muted">No folder open</p>
            <p class="mt-1 text-[10.5px] leading-relaxed text-faint">Files are read locally and never uploaded.</p>
          </div>`}
        ${recentCommits.length > 0 ? html`
          <div class="mt-2 border-t border-line pt-2">
            <div class="px-3 flex items-center justify-between">
              <span class="text-[10px] font-semibold uppercase tracking-[0.16em] text-faint">Recent changes</span>
              <button class="text-[10px] text-faint hover:text-ink transition-colors" onClick=${onRefreshCommits} title="Re-read .git/logs/HEAD">Refresh</button>
            </div>
            <ul class="mt-1 px-3 pb-2 space-y-1">
              ${recentCommits.map(commit => html`
                <li key=${`${commit.hash}:${commit.message}`} class="flex items-baseline gap-1.5 text-[10.5px] leading-snug" title=${commit.message}>
                  ${commit.hash ? html`<span class="font-mono text-[9.5px] text-faint shrink-0">${commit.hash}</span>` : ''}
                  <span class="text-muted truncate">${commit.message}</span>
                </li>
              `)}
            </ul>
          </div>
        ` : ''}
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
        <div class="flex items-center gap-1.5 pr-2 py-[3px] text-[11.5px] text-muted hover:bg-hover hover:text-ink transition-colors select-none file-tree-item" style=${{ paddingLeft: `${depth*12+8}px` }} onClick=${() => onToggle(e)}>
          <${Chevron} open=${open} class="w-2.5 h-2.5 text-faint" />
          <${FolderIcon} open=${open} class="w-3.5 h-3.5 shrink-0 text-faint" />
          <span class="truncate">${e.name}</span>
        </div>
        ${open
          ? e.loaded
            ? (e.children ? html`<${FileTree} entries=${e.children} expandedPaths=${expandedPaths} onToggle=${onToggle} onFileClick=${onFileClick} selectedFile=${selectedFile} depth=${depth+1} />` : '')
            : html`<div class="text-[10.5px] text-faint py-[3px] pl-10">Loading…</div>`
          : ''}
      </div>`
    }
    const selected = e.path === selectedFile
    return html`<div key=${e.path} class="flex items-center gap-1.5 pr-2 py-[3px] text-[11.5px] cursor-pointer transition-colors select-none file-tree-item ${selected ? 'bg-raised text-ink shadow-[inset_2px_0_0_var(--color-accent)]' : 'text-muted hover:bg-hover hover:text-ink'}" style=${{ paddingLeft: `${depth*12+8}px` }} onClick=${() => onFileClick(e.path)}>
      <span class="w-2.5 shrink-0"></span>
      <${FileIcon} class="w-3.5 h-3.5 shrink-0 text-faint" />
      <span class="truncate">${e.name}</span>
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

  const isUser = msg.role === 'user'
  return html`
    <div class="flex ${isUser ? 'justify-end' : 'justify-start'}">
      <div class="${isUser ? 'max-w-[88%] rounded-2xl rounded-br-md bg-raised border border-line px-3 py-2 text-ink' : 'max-w-[94%] w-full'}">
        ${!isUser ? html`<div class="flex items-center gap-1.5 mb-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-faint"><span class="w-1.5 h-1.5 rounded-full bg-accent"></span>Coder</div>` : ''}
        <div class="text-[12.5px] leading-relaxed whitespace-pre-wrap break-words">
          ${parts.map((p, i: number) => {
            if (p.t === 'text') return html`<span key=${i}>${p.v}</span>`
            const bi = appliedBlocks.findIndex((b: CodeBlock) => b.path===p.path && b.code===p.code)
            const applied = bi >= 0 && appliedBlocks[bi].applied
            return html`<div key=${i} class="my-2 rounded-lg overflow-hidden border border-line bg-canvas">
              <div class="flex items-center justify-between gap-2 px-2.5 py-1.5 bg-panel border-b border-line">
                <span class="font-mono text-[10px] text-muted truncate">${p.path || p.lang}</span>
                ${p.path ? html`<button class="text-[10px] px-2 py-0.5 rounded shrink-0 transition-colors ${applied ? 'text-ok border border-ok/30' : 'bg-accent hover:bg-accent-strong text-accent-ink font-medium'}" onClick=${() => !applied && onApply(p.path!, p.code!)} disabled=${applied}>${applied ? 'Applied' : 'Apply'}</button>` : ''}
              </div>
              <pre class="text-[11.5px] leading-relaxed p-3 overflow-x-auto font-mono text-ink"><code>${p.code}</code></pre>
            </div>`
          })}
          ${msg.isStreaming ? html`<span class="inline-block w-1.5 h-4 bg-accent animate-pulse ml-0.5 align-text-bottom"></span>` : ''}
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
  generationStatus: string
  onSend: (text: string) => void
  onStop: () => void
  onApply: (path: string, code: string) => void
  onClear: () => void
  onExport: () => void
  modelLoaded: boolean
  modelList: ModelDescriptor[]
  selectedModel: string
  onModelChange: (model: string) => void
  modelLoading: boolean
  initProgress: InitProgressState | null
  onLoadModel: () => void
  attachContext: boolean
  onAttachContextChange: (checked: boolean) => void
  customModelUrl: string
  onCustomModel: (url: string) => void
  selectedProvider: Provider
  onProviderChange: (provider: Provider) => void
  onShowBuiltInAI: () => void
  draft: string
  onDraftConsumed: () => void
}

function ChatPanel({ messages, streamingContent, codeBlocks, isGenerating, generationStatus, onSend, onStop, onApply, onClear, onExport, modelLoaded, modelList, selectedModel, onModelChange, modelLoading, initProgress, onLoadModel, attachContext, onAttachContextChange, customModelUrl, onCustomModel, selectedProvider, onProviderChange, onShowBuiltInAI, draft, onDraftConsumed }: ChatPanelProps) {
  const endRef = useRef<HTMLDivElement | null>(null)
  const [input, setInput] = useState('')
  const [confirmClear, setConfirmClear] = useState(false)
  const [tab, setTab] = useState<'chat' | 'model'>(modelLoaded ? 'chat' : 'model')
  const prevLoaded = useRef(modelLoaded)

  useEffect(() => {
    if (modelLoaded && !prevLoaded.current) setTab('chat')
    else if (!modelLoaded && prevLoaded.current) setTab('model')
    prevLoaded.current = modelLoaded
  }, [modelLoaded])

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, streamingContent, codeBlocks])

  useEffect(() => {
    if (!draft) return
    setInput(draft)
    onDraftConsumed()
  }, [draft, onDraftConsumed])

  useEffect(() => {
    if (!confirmClear) return
    const timer = setTimeout(() => setConfirmClear(false), 3000)
    return () => clearTimeout(timer)
  }, [confirmClear])

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
      <div class="h-10 border-b border-line flex items-center gap-1 px-2 shrink-0">
        <button
          class="px-2.5 py-1 rounded-md text-[11px] transition-colors ${tab === 'chat' ? 'bg-raised text-ink' : 'text-muted hover:text-ink'}"
          aria-pressed=${tab === 'chat'}
          onClick=${() => setTab('chat')}
        >Chat</button>
        <button
          class="px-2.5 py-1 rounded-md text-[11px] transition-colors flex items-center gap-1.5 ${tab === 'model' ? 'bg-raised text-ink' : 'text-muted hover:text-ink'}"
          aria-pressed=${tab === 'model'}
          onClick=${() => setTab('model')}
        >Model${!modelLoaded ? html`<span class="w-1.5 h-1.5 rounded-full bg-accent"></span>` : ''}</button>
        <div class="ml-auto flex items-center gap-2.5 text-[10.5px]">
          ${tab === 'chat' ? html`
            <button class="text-faint hover:text-ink transition-colors disabled:opacity-40 disabled:hover:text-faint" onClick=${onExport} disabled=${messages.length === 0}>Export</button>
            <button
              class="transition-colors ${confirmClear ? 'text-accent font-medium' : 'text-faint hover:text-ink'}"
              onClick=${() => { if (confirmClear) { onClear(); setConfirmClear(false) } else setConfirmClear(true) }}
            >${confirmClear ? 'Confirm clear' : 'Clear'}</button>
          ` : html`<span class="text-faint">${modelLoaded ? 'Ready' : modelLoading ? 'Loading' : 'Not loaded'}</span>`}
        </div>
      </div>
      ${tab === 'chat' ? html`
        <div class="flex-1 overflow-y-auto p-3 space-y-4">
          ${all.length===0 ? html`
            <div class="h-full flex flex-col items-center justify-center text-center px-4">
              <${Logo} class="w-10 h-10 mb-4" />
              <p class="text-[12.5px] font-medium text-ink">${modelLoaded ? 'Ask about your code' : 'No model loaded'}</p>
              <p class="mt-1.5 text-[11px] leading-relaxed text-muted max-w-[240px]">${modelLoaded
                ? 'The assistant reads and edits your project only when context is attached.'
                : 'Choose a model in the Model tab to start. Everything runs on this device.'}</p>
              ${modelLoaded ? html`
                <div class="mt-4 flex flex-wrap justify-center gap-1.5">
                  ${['Explain the open file', 'Find bugs in this project', 'Write a test'].map(prompt => html`
                    <button key=${prompt} class="px-2.5 py-1 rounded-full border border-line bg-raised text-[10.5px] text-muted hover:text-ink hover:border-line-strong transition-colors" onClick=${() => onSend(prompt)}>${prompt}</button>
                  `)}
                </div>` : html`
                <button class="mt-4 px-3 py-1.5 bg-accent hover:bg-accent-strong text-accent-ink text-[11.5px] font-medium rounded-md transition-colors" onClick=${() => setTab('model')}>Choose a model</button>`}
            </div>
          ` : all.map((m: Message, i: number) => html`<${MessageBubble} key=${i} msg=${m} onApply=${onApply} appliedBlocks=${codeBlocks} />`)}
          ${isGenerating && !streamingContent ? html`<div class="flex items-center gap-2 px-1 py-1"><div class="flex items-center gap-1 text-accent"><span class="dot-pulse">●</span><span class="dot-pulse">●</span><span class="dot-pulse">●</span></div><span class="text-[10.5px] text-muted">${generationStatus || 'Working on-device…'}</span></div>` : ''}
          <div ref=${endRef} />
        </div>
        <div class="p-3 border-t border-line space-y-2">
          ${codeBlocks.filter(b => !b.applied).length > 0 ? html`
            <div class="space-y-1 max-h-24 overflow-y-auto">
              ${codeBlocks.filter(b => !b.applied).map((b, i) => html`
                <div key=${i} class="flex items-center justify-between gap-2 bg-raised rounded-md px-2.5 py-1.5 border border-line">
                  <span class="font-mono text-[10px] text-muted truncate">${b.path || b.lang}</span>
                  <button class="text-[10px] px-2 py-0.5 bg-accent hover:bg-accent-strong text-accent-ink font-medium rounded shrink-0" onClick=${() => onApply(b.path, b.code)}>Apply</button>
                </div>
              `)}
            </div>
          ` : ''}
          <div class="flex gap-2 items-end">
            <textarea class="flex-1 bg-raised text-ink text-[12px] leading-relaxed rounded-lg px-3 py-2 border border-line outline-none focus:border-accent/70 resize-none placeholder:text-faint disabled:opacity-60" rows=${2} placeholder=${modelLoaded?"Ask about your code…":"Load a model first…"} value=${input} onInput=${(e: Event)=>setInput((e.target as HTMLTextAreaElement).value)} onKeyDown=${kd} disabled=${!modelLoaded}></textarea>
            <div class="flex flex-col gap-1">
              <button class="px-3 py-1.5 text-[11.5px] rounded-lg transition-colors ${modelLoaded&&!isGenerating?'bg-accent hover:bg-accent-strong text-accent-ink font-medium':'bg-raised text-faint border border-line cursor-not-allowed'}" onClick=${send} disabled=${!modelLoaded||isGenerating}>Send</button>
              ${isGenerating ? html`<button class="px-3 py-1.5 text-[11.5px] rounded-lg border border-danger/40 text-danger hover:bg-danger/10 transition-colors" onClick=${onStop}>Stop</button>` : ''}
            </div>
          </div>
        </div>
      ` : html`
        <div class="flex-1 overflow-y-auto p-3 space-y-3">
          <div>
            <p class="text-[12.5px] font-medium text-ink">Model</p>
            <p class="mt-1 text-[10.5px] leading-relaxed text-muted">Choose a model that runs locally in this browser. Chrome Built-in AI downloads its model once; the other providers download the selected weights.</p>
          </div>
          <label class="block space-y-1">
            <span class="text-[10px] text-faint">Provider</span>
            <select class="w-full bg-raised text-ink text-[11.5px] rounded-md px-2.5 py-1.5 border border-line outline-none focus:border-accent/70 disabled:opacity-50" value=${selectedProvider} onChange=${(e: Event) => onProviderChange((e.target as HTMLSelectElement).value as Provider)} disabled=${modelLoading}>
              ${window.__gpuAvailable ? html`<option value=${PROVIDERS.WEBLLM}>WebLLM</option>` : ''}
              ${window.__transformersAvailable ? html`<option value=${PROVIDERS.TRANSFORMERS_JS}>Transformers.js</option>` : ''}
              ${window.__browserAiAvailable ? html`<option value=${PROVIDERS.BROWSER_AI}>Chrome Built-in AI</option>` : html`<option disabled>${window.__browserAiNote || 'Chrome AI (not available)'}</option>`}
            </select>
          </label>
          <button class="w-full flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-md border border-line bg-raised hover:bg-hover text-[10.5px] text-muted hover:text-ink transition-colors" onClick=${onShowBuiltInAI}>
            <span class="flex items-center gap-1.5"><${SparkIcon} />Built-in AI APIs</span>
            <span class="text-faint">›</span>
          </button>
          <label class="block space-y-1">
            <span class="text-[10px] text-faint">Model</span>
            <select class="w-full bg-raised text-ink text-[11.5px] rounded-md px-2.5 py-1.5 border border-line outline-none focus:border-accent/70 disabled:opacity-50" value=${selectedModel} onChange=${(e: Event) => onModelChange((e.target as HTMLSelectElement).value)} disabled=${modelLoading}>
              ${modelList.filter(m => m.provider === selectedProvider).length === 0
                ? html`<option value="">No models available</option>`
                : modelList.filter(m => m.provider === selectedProvider).map(m => html`<option value=${m.id}>${m.label || m.id}${m.vram ? ' · '+m.vram+'MB' : ''}</option>`)}
            </select>
          </label>
          <input type="text" class="w-full bg-raised text-ink text-[11.5px] rounded-md px-2.5 py-1.5 border border-line outline-none focus:border-accent/70 placeholder:text-faint" placeholder="Custom model ID or URL" value=${customModelUrl} onInput=${(e: Event) => onCustomModel((e.target as HTMLInputElement).value)} />
          ${initProgress ? html`
            <div class="space-y-1.5">
              <div class="flex justify-between gap-2 text-[10.5px] text-muted"><span class="line-clamp-3 leading-snug">${initProgress.text}</span><span class="font-mono shrink-0">${Math.round(initProgress.progress * 100)}%</span></div>
              <div class="h-1 bg-raised rounded-full overflow-hidden"><div class="h-full bg-accent rounded-full transition-all duration-300" style=${{ width: `${initProgress.progress * 100}%` }}></div></div>
            </div>
          ` : ''}
          <button class="w-full px-3 py-1.5 text-[11.5px] rounded-md transition-colors flex items-center justify-center gap-1.5 ${modelLoaded ? 'border border-ok/40 bg-ok/10 text-ok' : modelLoading ? 'bg-raised text-muted border border-line cursor-not-allowed' : 'bg-accent hover:bg-accent-strong text-accent-ink font-medium'}" onClick=${onLoadModel} disabled=${modelLoading}>
            ${modelLoaded ? html`<${CheckIcon} /><span>Model ready</span>` : modelLoading ? 'Loading…' : 'Load model'}
          </button>
          ${modelLoaded ? html`<button class="w-full px-3 py-1.5 text-[11.5px] rounded-md border border-line bg-raised hover:bg-hover text-muted hover:text-ink transition-colors" onClick=${() => setTab('chat')}>Open chat</button>` : ''}
          <label class="flex items-start gap-2 text-[11px] text-muted cursor-pointer leading-snug">
            <input type="checkbox" checked=${attachContext} onChange=${(e: Event) => onAttachContextChange((e.target as HTMLInputElement).checked)} class="accent-accent mt-0.5" />
            <span>Attach project context</span>
          </label>
          <p class="text-[10px] text-faint leading-relaxed">With context attached, the assistant can read and edit files with tools. Models stay on this device.</p>
          <p class="text-[9px] text-faint font-mono">Build ${APP_BUILD}</p>
        </div>
      `}
    </div>
  `

}

const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)

function ResizeHandle({ width, min, max, invert, onWidth }: { width: number; min: number; max: number; invert?: boolean; onWidth: (width: number) => void }) {
  const start = useRef<{ x: number; width: number } | null>(null)

  useEffect(() => () => { document.body.style.cursor = '' }, [])

  const onPointerDown = (e: PointerEvent) => {
    e.preventDefault()
    start.current = { x: e.clientX, width }
    const onMove = (ev: PointerEvent) => {
      const origin = start.current
      if (!origin) return
      const delta = ev.clientX - origin.x
      onWidth(Math.min(max, Math.max(min, origin.width + (invert ? -delta : delta))))
    }
    const onUp = () => {
      start.current = null
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      document.body.style.cursor = ''
    }
    document.body.style.cursor = 'col-resize'
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  return html`<div class="hidden md:block w-1 shrink-0 cursor-col-resize bg-line hover:bg-accent/60 active:bg-accent transition-colors z-20" role="separator" aria-orientation="vertical" onPointerDown=${onPointerDown}></div>`
}

function QuickOpen({ files, onSelect, onClose }: { files: string[]; onSelect: (path: string) => void; onClose: () => void }) {
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const inputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => { inputRef.current?.focus() }, [])

  const results = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return files.slice(0, 100)
    const scored: Array<{ path: string; index: number }> = []
    for (const path of files) {
      const index = path.toLowerCase().indexOf(q)
      if (index >= 0) scored.push({ path, index })
    }
    scored.sort((a, b) => a.index - b.index || a.path.length - b.path.length)
    return scored.slice(0, 100).map(entry => entry.path)
  }, [files, query])

  useEffect(() => { setActive(0) }, [query])
  useEffect(() => {
    if (active > results.length - 1) setActive(Math.max(0, results.length - 1))
  }, [results, active])

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Escape') { e.preventDefault(); onClose() }
    else if (e.key === 'ArrowDown') { e.preventDefault(); setActive(a => Math.min(a + 1, Math.max(0, results.length - 1))) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(a => Math.max(a - 1, 0)) }
    else if (e.key === 'Enter') {
      e.preventDefault()
      const pick = results[active]
      if (pick) onSelect(pick)
    }
  }

  return html`
    <div class="fixed inset-0 z-50 flex items-start justify-center pt-[12vh] px-4">
      <div class="absolute inset-0 bg-black/60 backdrop-blur-[2px]" onClick=${onClose}></div>
      <div role="dialog" aria-modal="true" aria-label="Open file" class="relative w-full max-w-lg bg-panel border border-line rounded-xl shadow-2xl overflow-hidden">
        <div class="flex items-center gap-2 px-3 py-2.5 border-b border-line">
          <${SearchIcon} class="w-3.5 h-3.5 text-faint" />
          <input
            ref=${inputRef}
            class="flex-1 bg-transparent text-ink text-[12px] outline-none placeholder:text-faint"
            placeholder="Search files by path…"
            value=${query}
            onInput=${(e: Event) => setQuery((e.target as HTMLInputElement).value)}
            onKeyDown=${onKeyDown}
          />
          <span class="font-mono text-[9.5px] text-faint">${files.length} files</span>
        </div>
        <div class="max-h-[50vh] overflow-y-auto py-1">
          ${results.length === 0
            ? html`<p class="px-3 py-6 text-center text-[11px] text-muted">${files.length === 0 ? 'No files indexed yet. Open a folder first.' : 'No matching files.'}</p>`
            : results.map((path, index) => {
                const name = path.split('/').pop() || path
                const dir = path.slice(0, path.length - name.length).replace(/\/$/, '')
                return html`<button
                  key=${path}
                  class="w-full flex items-center gap-2 px-3 py-1.5 text-left transition-colors ${index === active ? 'bg-raised text-ink' : 'text-muted hover:bg-raised hover:text-ink'}"
                  onMouseEnter=${() => setActive(index)}
                  onClick=${() => onSelect(path)}
                >
                  <${FileIcon} class="w-3.5 h-3.5 shrink-0 text-faint" />
                  <span class="text-[11.5px] truncate">${name}</span>
                  ${dir ? html`<span class="ml-auto font-mono text-[9.5px] text-faint truncate max-w-[45%]">${dir}</span>` : ''}
                </button>`
              })}
        </div>
        <div class="px-3 py-2 border-t border-line text-[9.5px] text-faint flex items-center gap-3">
          <span><span class="font-mono">↑↓</span> navigate</span>
          <span><span class="font-mono">↵</span> open</span>
          <span><span class="font-mono">esc</span> close</span>
        </div>
      </div>
    </div>
  `
}

class ErrorBoundary extends Component<{ children?: unknown }, { error: Error | null }> {
  state: { error: Error | null } = { error: null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error) {
    console.error('UI error:', error)
    addNotification(`UI error: ${error.message}`, 'error')
  }

  render() {
    if (this.state.error) {
      return html`
        <div class="h-screen flex items-center justify-center bg-canvas p-6">
          <div class="text-center max-w-sm">
            <${Logo} class="w-10 h-10 mx-auto mb-4" />
            <h1 class="text-ink text-[14px] font-semibold">Something went wrong</h1>
            <p class="mt-2 text-[11px] text-muted leading-relaxed">${this.state.error.message}</p>
            <button class="mt-4 px-3 py-1.5 rounded-md bg-accent hover:bg-accent-strong text-accent-ink text-[11.5px] font-medium transition-colors" onClick=${() => location.reload()}>Reload</button>
          </div>
        </div>
      `
    }
    return this.props.children
  }
}

const BUILT_IN_STATUS: Record<BuiltInApiStatus, { label: string; dot: string; text: string }> = {
  available: { label: 'Available', dot: 'bg-ok', text: 'text-ok' },
  downloadable: { label: 'Download needed', dot: 'bg-accent', text: 'text-accent' },
  downloading: { label: 'Downloading', dot: 'bg-accent animate-pulse', text: 'text-accent' },
  unavailable: { label: 'Unavailable', dot: 'bg-danger', text: 'text-danger' },
  unsupported: { label: 'Not exposed', dot: 'bg-faint', text: 'text-faint' },
}

function BuiltInAIPanel({ onClose }: { onClose: () => void }) {
  const [apis, setApis] = useState<BuiltInApiInfo[] | null>(null)

  useEffect(() => {
    let alive = true
    getBuiltInAICapabilities()
      .then(list => { if (alive) setApis(list) })
      .catch(() => { if (alive) setApis([]) })
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => {
      alive = false
      window.removeEventListener('keydown', onKey)
    }
  }, [onClose])

  const ready = apis?.filter(api => api.status === 'available').length ?? 0

  return html`
    <div class="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div class="absolute inset-0 bg-black/60 backdrop-blur-[2px]" onClick=${onClose}></div>
      <div role="dialog" aria-modal="true" aria-labelledby="built-in-ai-title" class="relative w-full max-w-lg bg-panel border border-line rounded-xl shadow-2xl overflow-hidden">
        <div class="flex items-center gap-2.5 px-4 py-3 border-b border-line">
          <${Logo} class="w-5 h-5" />
          <div class="min-w-0">
            <h2 id="built-in-ai-title" class="text-[12.5px] font-semibold text-ink leading-tight">Built-in AI</h2>
            <p class="text-[10px] text-faint">On-device APIs exposed by this browser</p>
          </div>
          <span class="ml-auto font-mono text-[10px] text-muted">${apis ? `${ready}/${apis.length} ready` : ''}</span>
          <button class="p-1 rounded text-muted hover:text-ink hover:bg-raised transition-colors" aria-label="Close" onClick=${onClose}>✕</button>
        </div>
        <div class="p-2 max-h-[65vh] overflow-y-auto">
          ${apis
            ? apis.map(api => {
                const s = BUILT_IN_STATUS[api.status]
                return html`<div key=${api.id} class="flex items-start gap-2.5 rounded-lg px-2.5 py-2 hover:bg-raised transition-colors">
                  <span class="mt-[5px] w-1.5 h-1.5 rounded-full ${s.dot} shrink-0"></span>
                  <div class="min-w-0">
                    <div class="flex items-baseline gap-2">
                      <span class="text-[11.5px] font-medium text-ink">${api.label}</span>
                      <span class="text-[9px] font-semibold uppercase tracking-[0.12em] ${s.text}">${s.label}</span>
                    </div>
                    <p class="text-[10.5px] leading-snug text-muted mt-0.5">${api.description}</p>
                    ${api.note ? html`<p class="text-[9.5px] text-faint mt-0.5">${api.note}</p>` : ''}
                  </div>
                </div>`
              })
            : html`<div class="px-3 py-8 text-center text-[11px] text-muted">Probing built-in APIs…</div>`}
        </div>
        <p class="px-4 py-3 border-t border-line text-[9.5px] leading-relaxed text-faint">
          Only official Google Chrome on desktop exposes these APIs, in a secure context. Open-source Chromium and other browsers do not ship the on-device model — use WebLLM or Transformers.js there.
        </p>
      </div>
    </div>
  `
}

const TRANSLATE_LANGUAGES: Array<[string, string]> = [
  ['en', 'English'], ['es', 'Spanish'], ['fr', 'French'], ['de', 'German'],
  ['it', 'Italian'], ['pt', 'Portuguese'], ['nl', 'Dutch'], ['pl', 'Polish'],
  ['ru', 'Russian'], ['uk', 'Ukrainian'], ['tr', 'Turkish'], ['ar', 'Arabic'],
  ['hi', 'Hindi'], ['ja', 'Japanese'], ['ko', 'Korean'], ['zh', 'Chinese'],
  ['vi', 'Vietnamese'], ['th', 'Thai'], ['id', 'Indonesian'],
]

const languageLabel = (code: string) => TRANSLATE_LANGUAGES.find(([value]) => value === code)?.[1] ?? code

interface SummarizerInstance {
  summarizeStreaming(text: string, options?: { context?: string }): AsyncIterable<string>
  destroy?: () => void
}
interface SummarizerConstructor {
  availability(): Promise<string>
  create(options?: Record<string, unknown>): Promise<SummarizerInstance>
}
interface LanguageDetection {
  detectedLanguage: string
  confidence: number
}
interface LanguageDetectorInstance {
  detect(text: string): Promise<LanguageDetection[]>
  destroy?: () => void
}
interface LanguageDetectorConstructor {
  create(options?: Record<string, unknown>): Promise<LanguageDetectorInstance>
}
interface TranslatorInstance {
  translateStreaming(text: string): AsyncIterable<string>
  destroy?: () => void
}
interface TranslatorConstructor {
  availability(options: { sourceLanguage: string; targetLanguage: string }): Promise<string>
  create(options: Record<string, unknown>): Promise<TranslatorInstance>
}
interface ProofreadResult {
  correctedInput: string
  corrections: Array<{ startIndex: number; endIndex: number; correction: string }>
}
interface ProofreaderInstance {
  proofread(text: string): Promise<ProofreadResult>
  destroy?: () => void
}
interface ProofreaderConstructor {
  availability(): Promise<string>
  create(options?: Record<string, unknown>): Promise<ProofreaderInstance>
}
interface RewriterInstance {
  rewriteStreaming(text: string, options?: { context?: string }): AsyncIterable<string>
  destroy?: () => void
}
interface RewriterConstructor {
  availability(): Promise<string>
  create(options?: Record<string, unknown>): Promise<RewriterInstance>
}

/** Reads an untyped built-in AI global by name; the names are from a fixed list. */
function builtInCtor<T>(name: string): T | undefined {
  const globals = globalThis as unknown as Record<string, unknown>
  const value = globals[name]
  if (typeof value !== 'function') return undefined
  return value as T
}

function BuiltInTaskDialog({ request, onClose, onReplace }: { request: BuiltInActionRequest; onClose: () => void; onReplace: (text: string) => void }) {
  const [output, setOutput] = useState('')
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')
  const [preparing, setPreparing] = useState(true)
  const [copied, setCopied] = useState(false)
  const [detected, setDetected] = useState('')
  const [source, setSource] = useState('auto')
  const [target, setTarget] = useState('en')
  const [tone, setTone] = useState('more-formal')
  const [length, setLength] = useState('shorter')
  const [corrections, setCorrections] = useState(0)
  const detectorAvailable = hasBuiltInApi('LanguageDetector')

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  useEffect(() => {
    if (request.kind !== 'summarize' && request.kind !== 'summarize-file') return
    let stale = false
    ;(async () => {
      setPreparing(true); setError(''); setInfo(''); setOutput('')
      const SummarizerCtor = builtInCtor<SummarizerConstructor>('Summarizer')
      if (!SummarizerCtor) {
        setError('The Summarizer API is not exposed in this browser.')
        setPreparing(false)
        return
      }
      try {
        const availability = await SummarizerCtor.availability()
        if (availability === 'unavailable') {
          setError('The on-device summarizer is not available. Check Built-in AI APIs for its status.')
          setPreparing(false)
          return
        }
        if (availability !== 'available') setInfo('The summary model is downloading. This can take a few minutes.')
        const summarizer = await SummarizerCtor.create({
          type: 'key-points',
          format: 'markdown',
          length: 'short',
          sharedContext: 'Text selected in a source file or documentation from a developer project.',
        })
        if (stale) { summarizer.destroy?.(); return }
        setPreparing(false)
        let text = ''
        for await (const chunk of summarizer.summarizeStreaming(request.text)) {
          if (stale) break
          text += chunk
          setOutput(text)
        }
        summarizer.destroy?.()
      } catch (err) {
        if (!stale) {
          setError((err as Error).message || 'Summarization failed.')
          setPreparing(false)
        }
      }
    })()
    return () => { stale = true }
  }, [request])

  useEffect(() => {
    if (request.kind !== 'translate') return
    let stale = false
    ;(async () => {
      if (!detectorAvailable) {
        setSource('en')
        setTarget('es')
        return
      }
      const DetectorCtor = builtInCtor<LanguageDetectorConstructor>('LanguageDetector')
      if (!DetectorCtor) return
      try {
        const detector = await DetectorCtor.create()
        const results = await detector.detect(request.text)
        detector.destroy?.()
        if (stale) return
        const best = [...results].sort((a, b) => b.confidence - a.confidence)[0]
        if (best?.detectedLanguage) {
          setDetected(best.detectedLanguage)
          setTarget(best.detectedLanguage === 'en' ? 'es' : 'en')
        } else {
          setSource('en')
          setTarget('es')
        }
      } catch {
        if (!stale) { setSource('en'); setTarget('es') }
      }
    })()
    return () => { stale = true }
  }, [request, detectorAvailable])

  useEffect(() => {
    if (request.kind !== 'translate') return
    const sourceLanguage = source === 'auto' ? detected : source
    if (!sourceLanguage || !target) return
    if (sourceLanguage === target) {
      setOutput('')
      setError('')
      setInfo('Source and target languages are the same.')
      setPreparing(false)
      return
    }
    let stale = false
    ;(async () => {
      setPreparing(true); setError(''); setInfo(''); setOutput('')
      const TranslatorCtor = builtInCtor<TranslatorConstructor>('Translator')
      if (!TranslatorCtor) {
        setError('The Translator API is not exposed in this browser.')
        setPreparing(false)
        return
      }
      try {
        const availability = await TranslatorCtor.availability({ sourceLanguage, targetLanguage: target })
        if (availability === 'unavailable') {
          setError(`This browser cannot translate ${languageLabel(sourceLanguage)} → ${languageLabel(target)} yet.`)
          setPreparing(false)
          return
        }
        if (availability !== 'available') setInfo('The language pack is downloading. This can take a minute.')
        const translator = await TranslatorCtor.create({ sourceLanguage, targetLanguage: target })
        if (stale) { translator.destroy?.(); return }
        setPreparing(false)
        let text = ''
        for await (const chunk of translator.translateStreaming(request.text)) {
          if (stale) break
          text += chunk
          setOutput(text)
        }
        translator.destroy?.()
      } catch (err) {
        if (!stale) {
          setError((err as Error).message || 'Translation failed.')
          setPreparing(false)
        }
      }
    })()
    return () => { stale = true }
  }, [request, source, target, detected])

  useEffect(() => {
    if (request.kind !== 'proofread') return
    let stale = false
    ;(async () => {
      setPreparing(true); setError(''); setInfo(''); setOutput(''); setCorrections(0)
      const ProofreaderCtor = builtInCtor<ProofreaderConstructor>('Proofreader')
      if (!ProofreaderCtor) {
        setError('The Proofreader API is not exposed in this browser.')
        setPreparing(false)
        return
      }
      try {
        const availability = await ProofreaderCtor.availability()
        if (availability === 'unavailable') {
          setError('The on-device proofreader is not available. Check Built-in AI APIs for its status.')
          setPreparing(false)
          return
        }
        if (availability !== 'available') setInfo('The proofreading model is downloading. This can take a few minutes.')
        const proofreader = await ProofreaderCtor.create()
        if (stale) { proofreader.destroy?.(); return }
        const result = await proofreader.proofread(request.text)
        proofreader.destroy?.()
        if (stale) return
        setCorrections(result.corrections?.length ?? 0)
        setOutput(result.correctedInput || '')
        setPreparing(false)
      } catch (err) {
        if (!stale) {
          setError((err as Error).message || 'Proofreading failed.')
          setPreparing(false)
        }
      }
    })()
    return () => { stale = true }
  }, [request])

  useEffect(() => {
    if (request.kind !== 'rewrite') return
    let stale = false
    ;(async () => {
      setPreparing(true); setError(''); setInfo(''); setOutput('')
      const RewriterCtor = builtInCtor<RewriterConstructor>('Rewriter')
      if (!RewriterCtor) {
        setError('The Rewriter API is not exposed in this browser.')
        setPreparing(false)
        return
      }
      try {
        const availability = await RewriterCtor.availability()
        if (availability === 'unavailable') {
          setError('The on-device rewriter is not available. Check Built-in AI APIs for its status.')
          setPreparing(false)
          return
        }
        if (availability !== 'available') setInfo('The rewriting model is downloading. This can take a few minutes.')
        const rewriter = await RewriterCtor.create({
          tone,
          length,
          format: 'as-is',
          sharedContext: 'Comments, documentation, and technical text from a developer project.',
        })
        if (stale) { rewriter.destroy?.(); return }
        setPreparing(false)
        let text = ''
        for await (const chunk of rewriter.rewriteStreaming(request.text)) {
          if (stale) break
          text += chunk
          setOutput(text)
        }
        rewriter.destroy?.()
      } catch (err) {
        if (!stale) {
          setError((err as Error).message || 'Rewrite failed.')
          setPreparing(false)
        }
      }
    })()
    return () => { stale = true }
  }, [request, tone, length])

  const copy = async () => {
    if (!output) return
    try {
      await navigator.clipboard.writeText(output)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {}
  }

  const title = request.kind === 'summarize'
    ? 'Summarize selection'
    : request.kind === 'summarize-file'
      ? `Summarize ${request.path?.split('/').pop() || 'file'}`
      : request.kind === 'translate'
        ? 'Translate selection'
        : request.kind === 'proofread'
          ? 'Proofread selection'
          : 'Rewrite selection'

  const apiLabel = request.kind === 'summarize' || request.kind === 'summarize-file'
    ? 'Chrome Summarizer · on-device'
    : request.kind === 'translate'
      ? `Chrome Translator${detectorAvailable ? ' + Language Detector' : ''} · on-device`
      : request.kind === 'proofread'
        ? 'Chrome Proofreader · on-device'
        : 'Chrome Rewriter · on-device'

  return html`
    <div class="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div class="absolute inset-0 bg-black/60 backdrop-blur-[2px]" onClick=${onClose}></div>
      <div role="dialog" aria-modal="true" aria-labelledby="built-in-task-title" class="relative w-full max-w-xl bg-panel border border-line rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[80vh]">
        <div class="flex items-center gap-2.5 px-4 py-3 border-b border-line shrink-0">
          <${SparkIcon} class="w-4 h-4 text-accent" />
          <div class="min-w-0">
            <h2 id="built-in-task-title" class="text-[12.5px] font-semibold text-ink leading-tight">${title}</h2>
            <p class="text-[10px] text-faint">${apiLabel}</p>
          </div>
          <button class="ml-auto p-1 rounded text-muted hover:text-ink hover:bg-raised transition-colors" aria-label="Close" onClick=${onClose}>✕</button>
        </div>
        <div class="px-4 py-3 flex-1 overflow-y-auto min-h-[140px]">
          ${request.kind === 'translate' ? html`
            <div class="flex flex-wrap items-center gap-2 mb-3 text-[10.5px] text-muted">
              <label class="flex items-center gap-1.5">Source
                <select class="bg-raised text-ink text-[10.5px] rounded-md px-2 py-1 border border-line outline-none focus:border-accent/70" value=${source} onChange=${(e: Event) => setSource((e.target as HTMLSelectElement).value)}>
                  ${detectorAvailable ? html`<option value="auto">Auto${detected ? ` (${languageLabel(detected)})` : ''}</option>` : ''}
                  ${TRANSLATE_LANGUAGES.map(([code, label]) => html`<option value=${code}>${label}</option>`)}
                </select>
              </label>
              <span class="text-faint">→</span>
              <label class="flex items-center gap-1.5">Target
                <select class="bg-raised text-ink text-[10.5px] rounded-md px-2 py-1 border border-line outline-none focus:border-accent/70" value=${target} onChange=${(e: Event) => setTarget((e.target as HTMLSelectElement).value)}>
                  ${TRANSLATE_LANGUAGES.map(([code, label]) => html`<option value=${code}>${label}</option>`)}
                </select>
              </label>
            </div>
          ` : ''}
          ${request.kind === 'rewrite' ? html`
            <div class="flex flex-wrap items-center gap-2 mb-3 text-[10.5px] text-muted">
              <label class="flex items-center gap-1.5">Tone
                <select class="bg-raised text-ink text-[10.5px] rounded-md px-2 py-1 border border-line outline-none focus:border-accent/70" value=${tone} onChange=${(e: Event) => setTone((e.target as HTMLSelectElement).value)}>
                  <option value="more-formal">Formal</option>
                  <option value="as-is">As written</option>
                  <option value="more-casual">Casual</option>
                </select>
              </label>
              <label class="flex items-center gap-1.5">Length
                <select class="bg-raised text-ink text-[10.5px] rounded-md px-2 py-1 border border-line outline-none focus:border-accent/70" value=${length} onChange=${(e: Event) => setLength((e.target as HTMLSelectElement).value)}>
                  <option value="shorter">Shorter</option>
                  <option value="as-is">As written</option>
                  <option value="longer">Longer</option>
                </select>
              </label>
            </div>
          ` : ''}
          ${error
            ? html`<p class="text-[11.5px] leading-relaxed text-danger">${error}</p>`
            : preparing
              ? html`<p class="text-[11.5px] text-muted">Working on-device…</p>`
              : html`<div class="text-[11.5px] leading-relaxed whitespace-pre-wrap text-ink">${output || 'No text returned.'}</div>`}
          ${request.kind === 'proofread' && !preparing && !error ? html`<p class="mt-3 text-[10px] text-faint">${corrections === 0 ? 'No corrections suggested.' : `${corrections} correction${corrections === 1 ? '' : 's'} suggested.`}</p>` : ''}
          ${info ? html`<p class="mt-3 text-[10px] text-accent leading-snug">${info}</p>` : ''}
        </div>
        <div class="px-4 py-3 border-t border-line flex items-center gap-2 shrink-0">
          <span class="text-[9.5px] text-faint">Nothing leaves this device</span>
          <button class="ml-auto px-2.5 py-1 text-[10.5px] rounded-md border border-line bg-raised text-muted hover:text-ink hover:border-line-strong transition-colors disabled:opacity-40" onClick=${copy} disabled=${!output}>${copied ? 'Copied' : 'Copy'}</button>
          ${request.kind !== 'summarize' && request.kind !== 'summarize-file' ? html`
            <button class="px-2.5 py-1 text-[10.5px] rounded-md bg-accent hover:bg-accent-strong text-accent-ink font-medium transition-colors disabled:opacity-40" onClick=${() => onReplace(output)} disabled=${!output}>Replace selection</button>
          ` : ''}
        </div>
      </div>
    </div>
  `
}

function App() {
  const saved = loadState()
  const [treeData, setTreeData] = useState<FileEntry[] | null>(null)
  const [restorableProject, setRestorableProject] = useState<string | null>(null)
  const [indexedCount, setIndexedCount] = useState(0)
  const [recentCommits, setRecentCommits] = useState<RecentCommit[]>([])
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
  const [generationStatus, setGenerationStatus] = useState('')
  const [codeBlocks, setCodeBlocks] = useState<CodeBlock[]>([])
  const [attachContext, setAttachContext] = useState(saved.attachContext ?? true)
  const [customModelUrl, setCustomModelUrl] = useState(saved.customModelUrl || '')
  const [savedFile, setSavedFile] = useState('')
  const [mobilePanel, setMobilePanel] = useState<'files' | 'chat' | null>(null)
  const [showBuiltInAI, setShowBuiltInAI] = useState(false)
  const [builtInTask, setBuiltInTask] = useState<BuiltInActionRequest | null>(null)
  const [openFiles, setOpenFiles] = useState<string[]>([])
  const [dirtyFiles, setDirtyFiles] = useState<Set<string>>(new Set())
  const [quickOpen, setQuickOpen] = useState(false)
  const [chatDraft, setChatDraft] = useState('')
  const [sidebarWidth, setSidebarWidth] = useState(() => loadWidth('coder-sidebar-width', 256))
  const [chatWidth, setChatWidth] = useState(() => loadWidth('coder-chat-width', 380))
  const [notifs, setNotifs] = useState<Notification[]>([])

  useEffect(() => {
    const update = () => setNotifs([...notifications])
    notificationListeners.push(update)
    return () => {
      const idx = notificationListeners.indexOf(update)
      if (idx >= 0) notificationListeners.splice(idx, 1)
    }
  }, [])

  useEffect(() => {
    onFileSaved = (path) => setSavedFile(path)
    return () => { onFileSaved = null }
  }, [])

  useEffect(() => {
    onRecentCommits = setRecentCommits
    return () => { onRecentCommits = null }
  }, [])

  useEffect(() => {
    onBuiltInAction = (request) => setBuiltInTask(request)
    return () => { onBuiltInAction = null }
  }, [])

  useEffect(() => {
    onFileOpened = (path) => setOpenFiles(prev => prev.includes(path) ? prev : [...prev, path])
    onFileDirtyChanged = (path, dirty) => setDirtyFiles(prev => {
      if (dirty === prev.has(path)) return prev
      const next = new Set(prev)
      if (dirty) next.add(path)
      else next.delete(path)
      return next
    })
    return () => {
      onFileOpened = null
      onFileDirtyChanged = null
    }
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem('coder-sidebar-width', String(sidebarWidth))
      localStorage.setItem('coder-chat-width', String(chatWidth))
    } catch {}
  }, [sidebarWidth, chatWidth])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== 'k') return
      e.preventDefault()
      e.stopPropagation()
      setQuickOpen(open => !open)
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [])

  const editorRef = useRef<HTMLDivElement | null>(null)
  const abortRef = useRef(false)
  const generatingRef = useRef(false)
  const closeBuiltInAI = useCallback(() => setShowBuiltInAI(false), [])
  const closeBuiltInTask = useCallback(() => setBuiltInTask(null), [])
  const replaceBuiltInSelection = useCallback((text: string) => {
    if (builtInTask) replaceEditorRange(builtInTask.range, text)
    setBuiltInTask(null)
  }, [builtInTask])

  useEffect(() => {
    saveState({ selectedProvider, selectedModel, customModelUrl, attachContext, messages })
  }, [selectedProvider, selectedModel, customModelUrl, attachContext, messages])

  // Open tabs only make sense with a project loaded; saving them earlier would
  // overwrite the snapshot before the restore effect gets to read it.
  useEffect(() => {
    if (!treeData) return
    saveState({ openFiles, selectedFile, projectName: rootDirName })
  }, [treeData, openFiles, selectedFile])

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
    editorReadyPromise = initMonaco(editorRef.current!).then(() => {})
  }, [])

  const applyFolder = useCallback((entries: FileEntry[]) => {
    setMobilePanel(null)
    setIndexedCount(fileHandles.size)
    editor?.setModel(null)
    for (const m of monacoModels.values()) m.dispose()
    monacoModels.clear()
    savedContents.clear()
    setOpenFiles([])
    setDirtyFiles(new Set())
    setTreeData(entries)
    setExpandedPaths(new Set())
    setSelectedFile(null)
    currentFilePath = null
    currentFileHandle = null
    setSavedFile('')
    setRestorableProject(null)
  }, [])

  const restoreTabs = useCallback(async (files: string[], active: string | null) => {
    if (editorReadyPromise) await editorReadyPromise
    const valid: string[] = []
    for (const path of files) {
      if (await loadFile(path)) valid.push(path)
    }
    setOpenFiles(valid)
    const target = active && valid.includes(active) ? active : valid[valid.length - 1] ?? null
    setSelectedFile(target)
    if (target) {
      setSavedFile('')
      await loadFile(target)
    }
  }, [])

  const handleOpenFolder = useCallback(async () => {
    if (dirtyFiles.size > 0 && !window.confirm('Change folder? Unsaved changes will be discarded.')) return
    const entries = await openFolder(setIndexedCount)
    if (entries) applyFolder(entries)
  }, [dirtyFiles, applyFolder])

  const refreshTree = useCallback(async () => {
    if (!rootDirHandle) return
    const generation = folderGeneration
    try {
      const entries = await readExpandedTree(rootDirHandle, expandedPaths, generation)
      if (generation !== folderGeneration) return
      setTreeData(entries)
      setIndexedCount(fileHandles.size)
    } catch (err) {
      console.warn('refreshTree:', err)
    }
  }, [expandedPaths])

  const handleRefreshCommits = useCallback(async () => {
    if (!rootDirHandle) {
      setRecentCommits([])
      return
    }
    setRecentCommits(await readRecentCommits(rootDirHandle))
  }, [])

  // Restore the last project without a picker. Reloads keep an already granted
  // permission; after a browser restart Chrome reports "prompt", which needs a
  // click, so offer a button instead of failing silently.
  useEffect(() => {
    let alive = true
    ;(async () => {
      const handle = await readStoredRootHandle()
      if (!handle || !alive) return
      try {
        const permission = await handle.queryPermission({ mode: 'readwrite' })
        if (permission === 'granted') {
          const entries = await activateFolder(handle, setIndexedCount)
          if (entries && alive) {
            applyFolder(entries)
            if (saved.projectName === handle.name) {
              const files = [...(saved.openFiles ?? [])]
              if (saved.selectedFile && !files.includes(saved.selectedFile)) files.push(saved.selectedFile)
              void restoreTabs(files, saved.selectedFile ?? null)
            }
          }
        } else if (permission === 'prompt') {
          setRestorableProject(handle.name)
        } else {
          void clearStoredRootHandle()
        }
      } catch (err) {
        console.warn('restore folder:', err)
        void clearStoredRootHandle()
      }
    })()
    return () => { alive = false }
  }, [applyFolder, restoreTabs])

  const handleRestoreProject = useCallback(async () => {
    const handle = await readStoredRootHandle()
    if (!handle) {
      setRestorableProject(null)
      return
    }
    try {
      if (!await verifyPermission(handle, true)) {
        addNotification('Permission to reopen the project was denied.', 'warning')
        return
      }
      const entries = await activateFolder(handle, setIndexedCount)
      if (entries) {
        applyFolder(entries)
        if (saved.projectName === handle.name) {
          const files = [...(saved.openFiles ?? [])]
          if (saved.selectedFile && !files.includes(saved.selectedFile)) files.push(saved.selectedFile)
          void restoreTabs(files, saved.selectedFile ?? null)
        }
      }
    } catch (err) {
      console.error('restore folder:', err)
      addNotification(`Could not reopen ${handle.name}.`, 'error')
      void clearStoredRootHandle()
      setRestorableProject(null)
    }
  }, [applyFolder, restoreTabs])

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
    setSavedFile('')
    setMobilePanel(null)
    if (dirtyFiles.has(path) && monacoModels.has(path)) {
      editor?.setModel(monacoModels.get(path))
      currentFilePath = path
      onFileOpened?.(path)
      return
    }
    await loadFile(path)
  }, [dirtyFiles])

  const handleCloseTab = useCallback((path: string) => {
    if (dirtyFiles.has(path) && !window.confirm(`Close ${path.split('/').pop() || path} with unsaved changes?`)) return
    const remaining = openFiles.filter(item => item !== path)
    const wasActive = selectedFile === path
    setOpenFiles(remaining)
    setDirtyFiles(prev => {
      if (!prev.has(path)) return prev
      const next = new Set(prev)
      next.delete(path)
      return next
    })
    if (wasActive) {
      const index = openFiles.indexOf(path)
      const fallback = remaining[Math.min(index, remaining.length - 1)] || null
      setSelectedFile(fallback)
      setSavedFile('')
      const model = fallback ? monacoModels.get(fallback) : null
      if (model) {
        editor?.setModel(model)
        currentFilePath = fallback
      } else if (fallback) {
        void loadFile(fallback)
      } else {
        currentFilePath = null
        currentFileHandle = null
        editor?.setModel(null)
      }
    }
    const closedModel = monacoModels.get(path)
    if (closedModel) {
      closedModel.dispose()
      monacoModels.delete(path)
    }
    savedContents.delete(path)
  }, [dirtyFiles, openFiles, selectedFile])

  const summarizeOpenFile = useCallback(() => {
    if (!selectedFile) return
    const model = editor?.getModel()
    const text = model ? model.getValue() : ''
    if (!text.trim()) {
      addNotification('This file is empty.', 'info')
      return
    }
    setBuiltInTask({ kind: 'summarize-file', text, path: selectedFile })
  }, [selectedFile])

  const handleSend = useCallback(async (text: string) => {
    if (!currentModel || !modelLoaded || generatingRef.current) return
    generatingRef.current = true
    abortController = new AbortController()
    const userMsg: Message = { role: 'user', content: text }
    const newMsgs = [...messages, userMsg]
    setMessages(newMsgs)
    setIsGenerating(true)
    setStreamingContent('')
    setGenerationStatus('')
    setCodeBlocks([])
    abortRef.current = false

    // Pure greetings should not pay for a 15-tool structured session.
    const conversational = text.trim().length < 80 && /^(hi|hello|hey|thanks|thank you|ok|okay|sure|yes|no|who are you|what can you do|how are you)\b[!.?\s]*$/i.test(text.trim())
    const useContext = attachContext && !!treeData && !conversational
    const fileTree = useContext && treeData ? buildFileTree(treeData) : null
    let plainSystem = fileTree ? `${SYSTEM_PROMPT}\n\nProject file tree:\n${fileTree}` : SYSTEM_PROMPT
    // pi rebuilds the transcript per provider request, so send the full history.
    const chatMsgs: ModelMessage[] = [
      ...newMsgs
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => ({ role: m.role, content: m.content } as ModelMessage)),
    ]

    let tools: HarnessTool[] | undefined
    if (useContext) {
      tools = createHarnessTools({
        listFiles: async () => { await ensureIndexed(); return [...fileHandles.keys()] },
        readFile: readFileForTool,
        writeFile: applyFileChanges,
        createFile: createFileForTool,
        deleteFile: deleteFileForTool,
        createDirectory: createDirectoryForTool,
        getDiagnostics: collectEditorDiagnostics,
        getOpenFile: readOpenFileForTool,
        getSelection: readSelectionForTool,
      })
    }

    // Seed tool-enabled turns with the open file. Small models lack project-wide
    // memory; with tools this also saves an extra readFile round trip.
    let openFileBlock = ''
    if (useContext && currentFilePath && editor) {
      const model = editor.getModel()
      if (model) {
        let code = model.getValue()
        if (code.length > 8000) {
          code = `${code.slice(0, 8000)}\n…[truncated ${code.length - 8000} characters; use readFile for the rest]`
        }
        openFileBlock = `\n\nCurrent open file: ${currentFilePath}\n\`\`\`\n${code}\n\`\`\``
        plainSystem += openFileBlock
      }
    }

    const systemMsg = useContext
      ? (fileTree ? `${TOOL_SYSTEM_PROMPT}\n\nProject file tree:\n${fileTree}${openFileBlock}` : `${TOOL_SYSTEM_PROMPT}${openFileBlock}`)
      : plainSystem

    try {
      let full = ''
      for await (const chunk of streamChat(chatMsgs, systemMsg, tools, abortController.signal)) {
        if (abortRef.current) break
        if (chunk.tool?.ok) {
          const { name, path, from } = chunk.tool
          if (name === 'moveFile' && from && path) {
            const oldModel = monacoModels.get(from)
            if (oldModel) {
              oldModel.dispose()
              monacoModels.delete(from)
            }
            savedContents.delete(from)
            setOpenFiles((prev: string[]) => prev.map(item => item === from ? path : item))
            setDirtyFiles(prev => {
              const next = new Set(prev)
              next.delete(from)
              next.delete(path)
              return next
            })
            if (selectedFile === from) {
              setSelectedFile(path)
              void loadFile(path)
            }
          } else if (name === 'deleteFile' && path) {
            setOpenFiles((prev: string[]) => prev.filter(item => item !== path))
            if (selectedFile === path) setSelectedFile(null)
          }
          if (MUTATING_TOOLS.has(name)) void refreshTree()
        }
        if (chunk.status) setGenerationStatus(chunk.status)
        full = chunk.text
        setStreamingContent(full)
        const bl = parseCodeBlocks(full)
        if (bl.length) setCodeBlocks(bl)
      }
      if (!abortRef.current) {
        if (!full.trim()) {
          full = `The model returned an empty response. Build ${APP_BUILD}. Open DevTools and check for [pi-structured] diagnostics.`
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
      setGenerationStatus('')
    }
  }, [messages, modelLoaded, currentModel, attachContext, treeData, selectedProvider, selectedFile, refreshTree])

  const handleStop = useCallback(() => { abortRef.current = true; stopGeneration() }, [])
  const handleClear = useCallback(() => { setMessages([]); setStreamingContent(''); setCodeBlocks([]); saveState({ messages: [] }) }, [])
  const consumeChatDraft = useCallback(() => setChatDraft(''), [])

  useEffect(() => {
    onSelectionAction = (kind, text, path) => {
      const instruction = kind === 'explain'
        ? 'Explain what this code does, call out non-obvious behavior, and flag anything that looks wrong.'
        : kind === 'refactor'
          ? 'Refactor this code for clarity and correctness. Keep the public behavior identical and apply the minimal change.'
          : 'Add focused tests for this code, covering edge cases and failure modes.'
      const clipped = text.length > 8000 ? `${text.slice(0, 8000)}\n…[selection truncated]` : text
      const prompt = `${instruction}${path ? `\n\nFile: ${path}` : ''}\n\n\`\`\`\n${clipped}\n\`\`\``
      setMobilePanel('chat')
      if (!modelLoaded) {
        addNotification('Load a model first; the prompt is ready in the chat box.', 'warning')
        setChatDraft(prompt)
        return
      }
      setChatDraft('')
      void handleSend(prompt)
    }
    return () => { onSelectionAction = null }
  }, [modelLoaded, handleSend])

  const handleExportChat = useCallback(() => {
    if (!messages.length) return
    const body = messages
      .map(message => `## ${message.role === 'user' ? 'You' : 'Coder'}\n\n${message.content}`)
      .join('\n\n---\n\n')
    const blob = new Blob([`# Coder conversation\n\n${body}\n`], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `coder-chat-${new Date().toISOString().slice(0, 10)}.md`
    link.click()
    URL.revokeObjectURL(url)
  }, [messages])

  const handleApply = useCallback(async (path: string, code: string) => {
    const ok = await applyFileChanges(path, code)
    if (ok) {
      const entry = fileHandles.get(path)
      if (entry) setTreeData((prev: FileEntry[] | null) => addFileToTree(prev, path, entry))
      setCodeBlocks((prev: CodeBlock[]) => prev.map((b: CodeBlock) => (b.path===path && b.code===code) ? { ...b, applied: true } : b))
      if (path === currentFilePath) setSavedFile(path)
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
      ;(model as any).modelId = modelId
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

  const handleProviderChange = useCallback((p: Provider) => {
    if (p === PROVIDERS.BROWSER_AI && !window.__browserAiAvailable) return
    setSelectedProvider(p)
    setSelectedModel('')
    setModelLoaded(false)
    const models = getAvailableModels(4096)
    const first = models.find(m => m.provider === p)
    if (first) setSelectedModel(first.id)
  }, [])

  const handleModelChange = useCallback((m: string) => {
    setSelectedModel(m)
    setModelLoaded(false)
  }, [])

  const sidebarProps: SidebarProps = { treeData, indexedCount, expandedPaths, onToggleDir: handleToggleDir, onFileClick: handleFileClick, selectedFile, onOpenFolder: handleOpenFolder, restorableProject, onRestoreProject: handleRestoreProject, recentCommits, onRefreshCommits: handleRefreshCommits }

  const chatProps: ChatPanelProps = { messages, streamingContent, codeBlocks, isGenerating, generationStatus, onSend: handleSend, onStop: handleStop, onApply: handleApply, onClear: handleClear, onExport: handleExportChat, modelLoaded, modelList, selectedModel, onModelChange: handleModelChange, modelLoading, initProgress, onLoadModel: handleLoadModel, attachContext, onAttachContextChange: setAttachContext, customModelUrl, onCustomModel: setCustomModelUrl, selectedProvider, onProviderChange: handleProviderChange, onShowBuiltInAI: () => setShowBuiltInAI(true), draft: chatDraft, onDraftConsumed: consumeChatDraft }
  const quickFiles = useMemo(() => quickOpen ? [...fileHandles.keys()].sort() : [], [quickOpen, indexedCount])
  const modelLabel = modelList.find(m => m.id === selectedModel)?.label || selectedModel

  return html`
    <div class="h-screen flex flex-col bg-canvas text-ink">
      ${notifs.length > 0 ? html`
        <div class="fixed top-14 right-4 z-50 space-y-2 max-w-sm">
          ${notifs.map(n => html`
            <div key=${n.id} class="px-3 py-2 rounded-lg text-[11.5px] bg-panel/95 backdrop-blur border border-line shadow-lg ${n.level === 'warning' ? 'border-l-2 border-l-accent' : n.level === 'error' ? 'border-l-2 border-l-danger' : 'border-l-2 border-l-ok'}">
              <div class="font-medium text-ink mb-0.5">${n.level === 'warning' ? 'Warning' : n.level === 'error' ? 'Error' : 'Info'}</div>
              <div class="text-muted leading-snug">${n.text}</div>
            </div>
          `)}
        </div>
      ` : ''}
      <header class="h-11 bg-panel border-b border-line flex items-center gap-2.5 px-3 shrink-0 select-none">
        <${Logo} class="w-5 h-5" />
        <span class="font-semibold text-ink text-[12.5px] tracking-tight">Coder</span>
        <span class="hidden sm:inline text-[9px] font-medium uppercase tracking-[0.16em] text-faint border border-line rounded px-1.5 py-[1px]">Local-first</span>
        <div class="ml-auto flex items-center gap-1.5 md:gap-3 text-[11px] min-w-0">
          <div class="hidden md:flex items-center gap-3 min-w-0">
            ${modelLoaded ? html`<span class="inline-flex items-center gap-1.5 text-muted min-w-0"><span class="w-1.5 h-1.5 rounded-full bg-ok shrink-0"></span><span class="truncate max-w-[220px]">${modelLabel}</span></span>` : ''}
            ${savedFile ? html`<span class="inline-flex items-center gap-1.5 text-muted min-w-0" title=${savedFile}><${CheckIcon} class="w-3 h-3 text-ok shrink-0" /><span>Saved</span></span>` : ''}
          </div>
          <button
            class="md:hidden p-1.5 rounded-md border transition-colors ${mobilePanel === 'files' ? 'border-accent/60 bg-accent/10 text-accent' : 'border-line bg-raised text-muted hover:text-ink'}"
            aria-label="Project files"
            aria-pressed=${mobilePanel === 'files'}
            onClick=${() => setMobilePanel(p => p === 'files' ? null : 'files')}
          ><${FolderIcon} /></button>
          <button
            class="md:hidden p-1.5 rounded-md border transition-colors ${mobilePanel === 'chat' ? 'border-accent/60 bg-accent/10 text-accent' : 'border-line bg-raised text-muted hover:text-ink'}"
            aria-label="Chat"
            aria-pressed=${mobilePanel === 'chat'}
            onClick=${() => setMobilePanel(p => p === 'chat' ? null : 'chat')}
          ><${ChatIcon} /></button>
        </div>
      </header>
      <div class="flex-1 flex overflow-hidden relative">
        ${mobilePanel ? html`<div key="backdrop" class="md:hidden absolute inset-0 z-20 bg-black/60" onClick=${() => setMobilePanel(null)} />` : ''}
        <div key="sidebar" style=${`--sidebar-width:${sidebarWidth}px`} class="${mobilePanel === 'files' ? 'flex' : 'hidden'} md:flex absolute inset-y-0 left-0 z-30 w-72 max-w-[85vw] md:static md:z-auto md:max-w-none md:w-[var(--sidebar-width)] shrink-0 border-r border-line md:border-r-0 flex-col bg-panel overflow-hidden shadow-2xl md:shadow-none">
          <${Sidebar} ...${sidebarProps} />
        </div>
        <${ResizeHandle} width=${sidebarWidth} min=${200} max=${420} onWidth=${setSidebarWidth} />
        <div key="editor" class="flex-1 flex flex-col overflow-hidden bg-canvas min-w-0">
          <div class="h-10 border-b border-line flex items-stretch shrink-0 select-none bg-panel overflow-hidden">
            <div class="flex-1 flex items-stretch overflow-x-auto no-scrollbar">
              ${openFiles.length === 0
                ? html`<span class="px-3 flex items-center text-[10px] font-semibold uppercase tracking-[0.16em] text-faint">Editor</span>`
                : openFiles.map(path => {
                    const active = path === selectedFile
                    const name = path.split('/').pop() || path
                    return html`<div
                      key=${path}
                      class="group flex items-center gap-1.5 pl-3 pr-1.5 border-r border-line cursor-pointer transition-colors ${active ? 'bg-canvas text-ink shadow-[inset_0_-2px_0_var(--color-accent)]' : 'text-muted hover:text-ink hover:bg-hover'}"
                      onClick=${() => handleFileClick(path)}
                      title=${path}
                    >
                      <span class="text-[11px] truncate max-w-[220px]">${path}</span>
                      ${dirtyFiles.has(path) ? html`<span class="w-1.5 h-1.5 rounded-full bg-accent shrink-0" title="Unsaved changes"></span>` : ''}
                      <button
                        class="p-0.5 rounded text-faint hover:text-ink hover:bg-raised transition-opacity ${active ? 'opacity-100' : 'opacity-100 md:opacity-0 md:group-hover:opacity-100'}"
                        aria-label=${`Close ${name}`}
                        onClick=${(e: Event) => { e.stopPropagation(); handleCloseTab(path) }}
                      >✕</button>
                    </div>`
                  })}
            </div>
            <div class="flex items-center gap-0.5 px-1.5 shrink-0">
              ${selectedFile && hasBuiltInApi('Summarizer') ? html`
                <button class="flex items-center gap-1.5 px-2 py-1 rounded-md text-[10.5px] text-muted hover:text-ink hover:bg-raised transition-colors" onClick=${summarizeOpenFile} title="Summarize this file on-device" aria-label="Summarize current file">
                  <${SparkIcon} class="w-3 h-3" /><span class="hidden lg:inline">Summarize</span>
                </button>` : ''}
              <button class="flex items-center gap-1.5 px-2 py-1 rounded-md text-[10.5px] text-faint hover:text-ink hover:bg-raised transition-colors" onClick=${() => setQuickOpen(true)} title="Open file (Ctrl/Cmd+K)" aria-label="Open file (Ctrl/Cmd+K)">
                <${SearchIcon} class="w-3 h-3" /><span class="font-mono hidden lg:inline">${IS_MAC ? '⌘K' : 'Ctrl K'}</span>
              </button>
            </div>
          </div>
          <div class="flex-1 relative overflow-hidden">
            <div ref=${editorRef} class="absolute inset-0"></div>
            ${!treeData ? html`
              <div class="absolute inset-0 flex items-center justify-center bg-canvas z-10">
                <div class="text-center max-w-sm px-6">
                  <${Logo} class="w-12 h-12 mx-auto mb-5" />
                  <h1 class="text-ink text-[15px] font-semibold tracking-tight">A local-first AI editor</h1>
                  <p class="mt-2 text-[11.5px] leading-relaxed text-muted">Open a folder to browse and edit files with an assistant that runs entirely in your browser. Nothing is uploaded.</p>
                  <button class="mt-5 inline-flex items-center gap-2 px-4 py-2 bg-accent hover:bg-accent-strong text-accent-ink text-[12px] font-medium rounded-lg transition-colors" onClick=${handleOpenFolder}>
                    <${FolderPlusIcon} class="w-4 h-4" />
                    <span>Open folder</span>
                  </button>
                  <p class="mt-4 text-[10px] text-faint">Chromium-based browser · files stay on this device</p>
                </div>
              </div>
            ` : !selectedFile ? html`
              <div class="absolute inset-0 flex items-center justify-center bg-canvas z-10 pointer-events-none">
                <div class="text-center">
                  <${FileIcon} class="w-6 h-6 mx-auto text-faint" />
                  <p class="mt-2.5 text-[11.5px] text-muted">Select a file to start editing</p>
                  <p class="mt-1 text-[10.5px] text-faint">${rootDirName} is indexed for the assistant</p>
                </div>
              </div>
            ` : ''}
          </div>
        </div>
        <${ResizeHandle} width=${chatWidth} min=${280} max=${620} invert onWidth=${setChatWidth} />
        <div key="chat" style=${`--chat-width:${chatWidth}px`} class="${mobilePanel === 'chat' ? 'flex' : 'hidden'} md:flex absolute inset-y-0 right-0 z-30 w-full md:static md:z-auto md:w-[var(--chat-width)] shrink-0 border-l border-line md:border-l-0 flex-col bg-panel overflow-hidden shadow-2xl md:shadow-none">
          <${ChatPanel} ...${chatProps} />
        </div>
      </div>
      ${quickOpen ? html`<${QuickOpen} files=${quickFiles} onSelect=${(path: string) => { setQuickOpen(false); void handleFileClick(path) }} onClose=${() => setQuickOpen(false)} />` : ''}
      ${showBuiltInAI ? html`<${BuiltInAIPanel} onClose=${closeBuiltInAI} />` : ''}
      ${builtInTask ? html`<${BuiltInTaskDialog} request=${builtInTask} onClose=${closeBuiltInTask} onReplace=${replaceBuiltInSelection} />` : ''}
    </div>
  `
}

async function hasWebGPU(): Promise<boolean> {
  if (!navigator.gpu) return false
  try {
    return !!(await navigator.gpu.requestAdapter())
  } catch {
    return false
  }
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
  window.__gpuAvailable = await hasWebGPU()
  try {
    window.__transformersAvailable = await checkProviderAvailability(PROVIDERS.TRANSFORMERS_JS)
  } catch {
    window.__transformersAvailable = typeof WebAssembly !== 'undefined'
  }
  window.__browserAiAvailable = await hasBrowserAI()
}

function errorText(value: unknown): string {
  if (value instanceof Error) return value.message
  if (typeof value === 'string') return value
  try { return JSON.stringify(value) } catch { return String(value) }
}

window.addEventListener('unhandledrejection', (event) => {
  addNotification(`Unhandled error: ${errorText(event.reason)}`, 'error')
})

window.addEventListener('error', (event) => {
  if (event.message) addNotification(`UI error: ${event.message}`, 'error')
})

console.info('[coder] build', APP_BUILD)
;(window as any).__coderBuild = APP_BUILD

checkAvailability().then(() => {
  const root = document.getElementById('root')
  if (!root) return
  render(html`<${ErrorBoundary}><${App} /></${ErrorBoundary}>`, root)
})
