import { pipeline } from '@huggingface/transformers'

const MODEL_ID = 'onnx-community/Qwen2.5-0.5B-Instruct'
let generatorPromise

async function getGenerator() {
  if (!generatorPromise) {
    generatorPromise = pipeline('text-generation', MODEL_ID, {
      dtype: 'q8',
      progress_callback: (progress) => {
        if (progress.status === 'progress') {
          self.postMessage({
            type: 'progress',
            file: progress.file,
            loaded: progress.loaded,
            total: progress.total,
            percent: Number.isFinite(progress.progress) ? progress.progress : 0,
          })
        }
      },
    })
  }
  return generatorPromise
}

self.onmessage = async ({ data }) => {
  if (data.type !== 'ask') return

  try {
    self.postMessage({ type: 'status', status: 'loading' })
    const generator = await getGenerator()
    self.postMessage({ type: 'status', status: 'generating' })
    const output = await generator(data.prompt, { max_new_tokens: 220, temperature: 0.2, do_sample: false, return_full_text: false })
    const answer = output?.[0]?.generated_text?.trim()
    if (!answer) throw new Error('The micro-model returned an empty answer.')
    self.postMessage({ type: 'answer', answer })
  } catch (error) {
    self.postMessage({ type: 'error', error: error.message || 'The micro-model failed to answer.' })
  }
}
