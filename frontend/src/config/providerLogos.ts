/**
 * Provider logo resource map.
 *
 * Logo assets are vendored from @lobehub/icons-static-svg. The local provider
 * intentionally falls back to ProviderAvatar's generated avatar because it is
 * not a third-party service brand.
 */

import OpenAIProviderLogo from '../assets/images/providers/openai.svg'
import AliyunProviderLogo from '../assets/images/providers/aliyun.svg'
import SiliconProviderLogo from '../assets/images/providers/silicon.svg'
import MoonshotProviderLogo from '../assets/images/providers/moonshot.svg'
import DeepSeekProviderLogo from '../assets/images/providers/deepseek.svg'
import ZhipuProviderLogo from '../assets/images/providers/zhipu.svg'
import MinimaxProviderLogo from '../assets/images/providers/minimax.svg'
import GoogleProviderLogo from '../assets/images/providers/google.svg'
import BaiduProviderLogo from '../assets/images/providers/baidu-cloud.svg'
import TencentProviderLogo from '../assets/images/providers/tencent-cloud.svg'
import VolcengineProviderLogo from '../assets/images/providers/volcengine.svg'
import SparkProviderLogo from '../assets/images/providers/spark.svg'
import OllamaProviderLogo from '../assets/images/providers/ollama.svg'
import AnthropicProviderLogo from '../assets/images/providers/anthropic.svg'
import GeminiProviderLogo from '../assets/images/providers/gemini.svg'
import GrokProviderLogo from '../assets/images/providers/grok.svg'
import DoubaoProviderLogo from '../assets/images/providers/doubao.svg'
import QwenProviderLogo from '../assets/images/providers/qwen.svg'
import ZeroOneProviderLogo from '../assets/images/providers/zero-one.svg'
import MistralProviderLogo from '../assets/images/providers/mistral.svg'
import CohereProviderLogo from '../assets/images/providers/cohere.svg'
import NvidiaProviderLogo from '../assets/images/providers/nvidia.svg'
import BaichuanProviderLogo from '../assets/images/providers/baichuan.svg'
import StepProviderLogo from '../assets/images/providers/step.svg'
import HunyuanProviderLogo from '../assets/images/providers/hunyuan.svg'
import YiProviderLogo from '../assets/images/providers/yi.svg'

export const PROVIDER_LOGO_MAP: Record<string, string> = {
  openai: OpenAIProviderLogo,
  aliyun: AliyunProviderLogo,
  silicon: SiliconProviderLogo,
  moonshot: MoonshotProviderLogo,
  deepseek: DeepSeekProviderLogo,
  zhipu: ZhipuProviderLogo,
  minimax: MinimaxProviderLogo,
  google: GoogleProviderLogo,
  baidu: BaiduProviderLogo,
  tencent: TencentProviderLogo,
  volcengine: VolcengineProviderLogo,
  spark: SparkProviderLogo,
  ollama: OllamaProviderLogo,
  anthropic: AnthropicProviderLogo,
  gemini: GeminiProviderLogo,
  grok: GrokProviderLogo,
  doubao: DoubaoProviderLogo,
  qwen: QwenProviderLogo,
  'zero-one': ZeroOneProviderLogo,
  mistral: MistralProviderLogo,
  cohere: CohereProviderLogo,
  nvidia: NvidiaProviderLogo,
  baichuan: BaichuanProviderLogo,
  step: StepProviderLogo,
  hunyuan: HunyuanProviderLogo,
  yi: YiProviderLogo,
}

/**
 * Get provider logo URL.
 * @param providerId - Provider id.
 * @returns Logo image URL if configured, otherwise undefined.
 */
export function getProviderLogo(providerId: string): string | undefined {
  return PROVIDER_LOGO_MAP[providerId]
}
