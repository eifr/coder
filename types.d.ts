declare module 'htm/preact/standalone.mjs' {
  export function html(strings: TemplateStringsArray, ...values: any[]): any
  export function render(vnode: any, parent: HTMLElement): void
  export function useState<T>(initial: T): [T, (v: T | ((prev: T) => T)) => void]
  export function useEffect(fn: () => void | (() => void), deps?: any[]): void
  export function useRef<T>(initial: T): { current: T }
  export function useCallback<T extends Function>(fn: T, deps: any[]): T
  export function useMemo<T>(fn: () => T, deps: any[]): T
  export class Component<P = {}, S = {}> {
    constructor(props: P)
    props: P
    state: S
    setState(state: Partial<S> | ((prev: S, props: P) => Partial<S>)): void
    render(): any
  }
}

declare module '*?worker' {
  const WorkerConstructor: new () => Worker
  export default WorkerConstructor
}

interface FileSystemHandlePermissionDescriptor {
  mode?: 'read' | 'readwrite'
}

interface FileSystemHandle {
  queryPermission(descriptor: FileSystemHandlePermissionDescriptor): Promise<PermissionState>
  requestPermission(descriptor: FileSystemHandlePermissionDescriptor): Promise<PermissionState>
}

interface Window {
  showDirectoryPicker?(): Promise<FileSystemDirectoryHandle>
}

declare const self: WorkerGlobalScope
