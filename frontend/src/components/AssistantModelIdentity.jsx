import React from 'react'
import { useProvider } from '../contexts/ProviderContext'
import { resolveModelBrandProviderId } from '../utils/modelBrandUtils'
import ProviderAvatar from './ProviderAvatar'

const BRAND_NAMES = {
  anthropic: 'Anthropic',
  baichuan: '百川',
  baidu: '百度',
  cohere: 'Cohere',
  deepseek: 'DeepSeek',
  doubao: '豆包',
  gemini: 'Google Gemini',
  grok: 'xAI',
  hunyuan: '腾讯混元',
  minimax: 'MiniMax',
  mistral: 'Mistral AI',
  moonshot: 'Moonshot AI',
  openai: 'OpenAI',
  qwen: '通义千问',
  spark: '讯飞星火',
  step: '阶跃星辰',
  xiaomi: '小米',
  yi: '零一万物',
  zhipu: '智谱 AI',
}

export default function AssistantModelIdentity({ model, providerId, darkMode = false }) {
  const { getProviderById } = useProvider()
  const modelName = String(model || '').trim() || 'ASSISTANT'
  const serviceProviderId = String(providerId || '').trim().toLowerCase()
  const brandProviderId = resolveModelBrandProviderId(modelName, serviceProviderId)
  const configuredBrand = getProviderById?.(brandProviderId)
  const configuredService = serviceProviderId ? getProviderById?.(serviceProviderId) : null
  const brandName = configuredBrand?.name || BRAND_NAMES[brandProviderId] || brandProviderId
  const serviceName = configuredService?.name || serviceProviderId
  const provider = configuredBrand || { id: brandProviderId, name: brandName, logo: null }
  const title = serviceProviderId && serviceProviderId !== brandProviderId
    ? `${brandName} 模型，由 ${serviceName} 提供 API 服务`
    : `${brandName} 模型`

  return (
    <div
      className={`mb-2 inline-flex max-w-full min-w-0 items-center gap-2 rounded-[13px] border py-1 pl-1 pr-3 select-none transition-[transform,box-shadow,border-color,background-color] duration-200 ease-out hover:-translate-y-px motion-reduce:transform-none ${
        darkMode
          ? 'border-white/[0.09] bg-[#25282e]/95 shadow-[0_1px_2px_rgba(0,0,0,0.2),0_10px_22px_-13px_rgba(0,0,0,0.8),inset_0_1px_0_rgba(255,255,255,0.06)] hover:border-white/[0.14] hover:shadow-[0_2px_4px_rgba(0,0,0,0.24),0_13px_26px_-13px_rgba(0,0,0,0.88),inset_0_1px_0_rgba(255,255,255,0.07)]'
          : 'border-[#ebe3dc]/90 bg-white/95 shadow-[0_1px_2px_rgba(83,65,55,0.07),0_9px_20px_-12px_rgba(83,65,55,0.4),inset_0_1px_0_rgba(255,255,255,0.95)] hover:border-[#e1d6ce] hover:bg-white hover:shadow-[0_2px_4px_rgba(83,65,55,0.09),0_13px_24px_-12px_rgba(83,65,55,0.46),inset_0_1px_0_rgba(255,255,255,0.98)]'
      }`}
      data-model-brand={brandProviderId}
      data-surface="floating"
      title={title}
    >
      <span
        className={`flex h-7 w-7 flex-none items-center justify-center rounded-[9px] p-[5px] transition-colors ${
          darkMode
            ? 'bg-white/[0.07] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.07)]'
            : 'bg-[#f8f4f1] shadow-[inset_0_0_0_1px_rgba(155,125,108,0.09),inset_0_1px_0_rgba(255,255,255,0.9)]'
        }`}
        aria-label={title}
      >
        <ProviderAvatar provider={provider} providerId={brandProviderId} size={16} className="rounded-[5px]" />
      </span>
      <span
        className={`min-w-0 max-w-[min(72vw,24rem)] truncate text-[10.5px] font-semibold uppercase leading-none ${
          darkMode ? 'text-[#FFD4C6]' : 'text-[#B85F47]'
        }`}
        title={modelName}
      >
        {modelName}
      </span>
    </div>
  )
}
