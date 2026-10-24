import { MockLanguageModelV3 } from 'ai/test'
import { Agent, type AgentTool } from '@earendil-works/pi-agent-core'
import { Type } from '@earendil-works/pi-ai'
import { createLocalModels } from './pi-local-provider'

function textStream(text: string): ReadableStream {
  return new ReadableStream({
    start(controller) {
      controller.enqueue({ type: 'stream-start', warnings: [] })
      controller.enqueue({ type: 'text-start', id: 't1' })
      controller.enqueue({ type: 'text-delta', id: 't1', delta: text })
      controller.enqueue({ type: 'text-end', id: 't1' })
      controller.enqueue({
        type: 'finish',
        usage: {
          inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 1, text: 1, reasoning: 0 },
        },
        finishReason: 'stop',
      })
      controller.close()
    },
  })
}

export async function runLocalProviderSmokeTest() {
  let call = 0
  const model = new MockLanguageModelV3({
    doStream: async () => ({
      stream: textStream(call++ === 0
        ? '{"tool":"echo","arguments":{"value":"pi"}}'
        : '{"text":"done"}'),
    }),
  })

  const models = createLocalModels({ provider: 'transformers-js', model, modelId: 'mock' })
  const localModel = models.getModels()[0]
  if (!localModel) throw new Error('mock model not registered')

  const parameters = Type.Object({ value: Type.String() })
  const tools: AgentTool<typeof parameters>[] = [{
    name: 'echo',
    label: 'Echo',
    description: 'Echo a value.',
    parameters,
    execute: async (_id, args) => ({
      content: [{ type: 'text' as const, text: `echo:${args.value}` }],
      details: undefined,
    }),
  }]

  const agent = new Agent({
    initialState: { systemPrompt: 'test', model: localModel, tools },
    streamFn: models.streamSimple.bind(models),
    convertToLlm: messages => messages,
  })

  const deltas: string[] = []
  const toolResults: string[] = []
  agent.subscribe(event => {
    if (event.type === 'message_update' && event.assistantMessageEvent.type === 'text_delta') {
      deltas.push(event.assistantMessageEvent.delta)
    }
    if (event.type === 'tool_execution_end') {
      toolResults.push(event.toolName)
    }
  })

  await agent.prompt('call echo')
  return {
    deltas,
    toolResults,
    final: agent.state.messages.at(-1),
  }
}
