/**
 * Local smoke test for pi-ai + pi-agent-core on Chrome's built-in Prompt API.
 *
 * Run:
 *   bun run dev -- --host 127.0.0.1 --port 5174
 *   open http://127.0.0.1:5174/pi-test.html in Chrome 148+ Stable
 *
 * Tool calls use Prompt API `responseConstraint` JSON instead of native
 * tool-call content: Chrome 156's on-device model emits markdown tool blocks,
 * not structured calls. No AIPromptAPIToolUse flag is required.
 */
import { Agent, type AgentTool } from '@earendil-works/pi-agent-core'
import { Type } from '@earendil-works/pi-ai'
import { createChromeModels, getChromeAvailability, MODEL_ID, PROVIDER_ID } from './pi-chrome-provider'

const statusEl = document.querySelector<HTMLDivElement>('#status')!
const logEl = document.querySelector<HTMLPreElement>('#log')!
const probeButton = document.querySelector<HTMLButtonElement>('#probe')!
const textButton = document.querySelector<HTMLButtonElement>('#text')!
const toolButton = document.querySelector<HTMLButtonElement>('#tool')!
const clearButton = document.querySelector<HTMLButtonElement>('#clear')!

function append(text: string, kind = 'meta') {
  const span = document.createElement('span')
  span.className = kind
  span.textContent = text
  logEl.appendChild(span)
  logEl.scrollTop = logEl.scrollHeight
}

function line(text: string, kind = 'meta') {
  append(`${new Date().toLocaleTimeString()} ${text}\n`, kind)
}

async function refreshStatus() {
  const info = await getChromeAvailability()
  const globalModel = (globalThis as Record<string, unknown>).LanguageModel
  const constructors = {
    toolCall: typeof (globalThis as Record<string, unknown>).LanguageModelToolCall,
    success: typeof (globalThis as Record<string, unknown>).LanguageModelToolSuccess,
    error: typeof (globalThis as Record<string, unknown>).LanguageModelToolError,
  }
  statusEl.textContent = [
    `LanguageModel: ${typeof globalModel}`,
    `availability: ${info.availability}`,
    `tool constructors: ${JSON.stringify(constructors)}`,
  ].join('\n')
  return info
}

async function probe() {
  probeButton.disabled = true
  try {
    const info = await refreshStatus()
    line(`probe: availability=${info.availability}`, 'meta')
    const languageModel = (globalThis as Record<string, unknown>).LanguageModel as {
      create(options: Record<string, unknown>): Promise<{
        contextWindow?: number
        contextUsage?: number
        destroy?: () => void
      }>
    } | undefined
    if (!languageModel) {
      line('probe: LanguageModel missing', 'error')
      return
    }
    let lastPercent = -1
    const session = await languageModel.create({
      monitor(monitor: { addEventListener(type: string, listener: (event: { loaded: number; total?: number }) => void): void }) {
        monitor.addEventListener('downloadprogress', event => {
          const total = event.total ?? 0
          const percent = total > 0 ? Math.round((event.loaded / total) * 100) : -1
          if (percent === lastPercent) return
          lastPercent = percent
          statusEl.textContent = `Downloading model… ${percent >= 0 ? `${percent}%` : ''}`
        })
      },
    })
    line(`probe: session created; contextWindow=${session.contextWindow}`, 'meta')
    session.destroy?.()
    await refreshStatus()
  } catch (error) {
    line(`probe error: ${error instanceof Error ? `${error.name}: ${error.message}` : String(error)}`, 'error')
  } finally {
    probeButton.disabled = false
  }
}

interface RunOptions {
  withTools: boolean
  prompt: string
}

async function runAgent({ withTools, prompt }: RunOptions) {
  textButton.disabled = true
  toolButton.disabled = true
  line(`--- ${withTools ? 'tool' : 'text'} turn ---`, 'meta')
  const models = createChromeModels()
  const model = models.getModel(PROVIDER_ID, MODEL_ID)
  if (!model) throw new Error('Chrome model not registered')

  const echoParameters = Type.Object({
    value: Type.String({ description: 'The value to echo.' }),
  })
  const tools: AgentTool<typeof echoParameters>[] = withTools
    ? [{
        name: 'echo',
        label: 'Echo',
        description: 'Echo the provided value back to the caller.',
        parameters: echoParameters,
        execute: async (_toolCallId, args) => ({
          content: [{ type: 'text' as const, text: `echo:${args.value}` }],
          details: undefined,
        }),
      }]
    : []

  const agent = new Agent({
    initialState: {
      systemPrompt: 'You are a terse assistant. Follow the user instruction exactly.',
      model,
      tools,
    },
    streamFn: models.streamSimple.bind(models),
    convertToLlm: messages => messages,
  })

  let currentText = ''
  agent.subscribe(event => {
    if (event.type === 'message_update') {
      const update = event.assistantMessageEvent
      if (update.type === 'text_start') {
        currentText = ''
        line('assistant: ', 'text')
      } else if (update.type === 'text_delta') {
        currentText += update.delta
        append(update.delta, 'text')
      } else if (update.type === 'text_end') {
        append('\n', 'text')
      } else if (update.type === 'toolcall_end') {
        line(`tool call: ${update.toolCall.name} ${JSON.stringify(update.toolCall.arguments)}`, 'tool')
      }
    } else if (event.type === 'tool_execution_end') {
      const content = event.result.content
        .map((part: { type: string; text?: string }) => part.type === 'text' ? part.text ?? '' : '[image]')
        .join('')
      line(`tool result: ${event.toolName} -> ${content}`, 'tool')
    } else if (event.type === 'turn_end') {
      if (event.message.role === 'assistant') {
        line(`turn end: stopReason=${event.message.stopReason} usage=${event.message.usage.input}/${event.message.usage.output}`, 'meta')
      } else {
        line(`turn end: ${event.message.role}`, 'meta')
      }
    } else if (event.type === 'agent_end') {
      line(`agent end: ${event.messages.length} messages`, 'meta')
    } else if (event.type === 'agent_start' || event.type === 'turn_start') {
      line(event.type, 'meta')
    }
  })

  try {
    await agent.prompt(prompt)
  } catch (error) {
    line(`agent error: ${error instanceof Error ? `${error.name}: ${error.message}` : String(error)}`, 'error')
  } finally {
    textButton.disabled = false
    toolButton.disabled = false
  }
}

probeButton.addEventListener('click', () => { void probe() })
textButton.addEventListener('click', () => {
  void runAgent({
    withTools: false,
    prompt: 'In one short sentence, say hello and name the model you are running on.',
  })
})
toolButton.addEventListener('click', () => {
  void runAgent({
    withTools: true,
    prompt: 'Call the echo tool with value "pi". Then tell me exactly what it returned.',
  })
})
clearButton.addEventListener('click', () => { logEl.textContent = '' })

line('page ready', 'meta')
void refreshStatus()
