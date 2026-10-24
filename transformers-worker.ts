import { TransformersJSWorkerHandler } from '@browser-ai/transformers-js'

const handler = new TransformersJSWorkerHandler()
const scope = self as unknown as { onmessage: ((event: MessageEvent) => void) | null }
scope.onmessage = event => handler.onmessage(event)
