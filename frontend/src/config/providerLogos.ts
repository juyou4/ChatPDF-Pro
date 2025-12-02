/**
 * Provider Logo资源映射配置
 * 管理所有embedding服务商的图标资源
 */

// 导入所有provider图标
import LocalProviderLogo from '../assets/images/providers/local.png'
import OpenAIProviderLogo from '../assets/images/providers/openai.png'
import AliyunProviderLogo from '../assets/images/providers/aliyun.png'
import SiliconProviderLogo from '../assets/images/providers/silicon.png'
import MoonshotProviderLogo from '../assets/images/providers/moonshot.png'
import DeepSeekProviderLogo from '../assets/images/providers/deepseek.png'
import ZhipuProviderLogo from '../assets/images/providers/zhipu.png'
import MinimaxProviderLogo from '../assets/images/providers/minimax.png'
import GoogleProviderLogo from '../assets/images/providers/google.png'
import BaiduProviderLogo from '../assets/images/providers/baidu-cloud.svg'
import TencentProviderLogo from '../assets/images/providers/tencent-cloud-ti.png'
import VolcengineProviderLogo from '../assets/images/providers/volcengine.png'
import SparkProviderLogo from '../assets/images/providers/xirang.png' // 讯飞星火通常用xirang或spark
import OllamaProviderLogo from '../assets/images/providers/ollama.png'
import AnthropicProviderLogo from '../assets/images/providers/anthropic.png'
import GeminiProviderLogo from '../assets/images/providers/gemini.png'
import GrokProviderLogo from '../assets/images/providers/grok.png'
import DoubaoProviderLogo from '../assets/images/providers/doubao.png'
import QwenProviderLogo from '../assets/images/providers/aliyun.png' // 通义千问复用阿里云
import ZeroOneProviderLogo from '../assets/images/providers/zero-one.png'
import MistralProviderLogo from '../assets/images/providers/mistral.png'
import CohereProviderLogo from '../assets/images/providers/cohere.png'
import NvidiaProviderLogo from '../assets/images/providers/nvidia.png'
import BaichuanProviderLogo from '../assets/images/providers/baichuan.png'
import StepProviderLogo from '../assets/images/providers/step.png'
import HunyuanProviderLogo from '../assets/images/providers/tencent-cloud-ti.png' // 混元复用腾讯云
import YiProviderLogo from '../assets/images/providers/zero-one.png' // Yi复用零一万物
// meta.png 和 shanghai-ai.png 不存在，这些映射将回退到默认头像

/**
 * Provider Logo映射表
 * key: provider id
 * value: logo图片路径
 */
export const PROVIDER_LOGO_MAP: Record<string, string> = {
  local: LocalProviderLogo,
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
  yi: YiProviderLogo
  // meta 和 internlm 图标不存在，将使用 ProviderAvatar 的回退逻辑
}

/**
 * 获取Provider Logo URL
 * @param providerId - Provider的ID
 * @returns Logo图片URL，如果不存在则返回undefined
 */
export function getProviderLogo(providerId: string): string | undefined {
  return PROVIDER_LOGO_MAP[providerId]
}
