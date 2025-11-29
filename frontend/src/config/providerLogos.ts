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
  minimax: MinimaxProviderLogo
}

/**
 * 获取Provider Logo URL
 * @param providerId - Provider的ID
 * @returns Logo图片URL，如果不存在则返回undefined
 */
export function getProviderLogo(providerId: string): string | undefined {
  return PROVIDER_LOGO_MAP[providerId]
}
