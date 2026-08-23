import React, { useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Check,
  CheckCircle2,
  AlertCircle,
  ChevronDown,
  Eye,
  EyeOff,
  Key,
  Plug,
  Plus,
  RefreshCw,
  Search,
  Server,
  Settings,
  Shield,
  Trash2,
  ChevronLeft,
  Box, Edit3, Link2, Play, ChevronRight, CheckSquare, Sparkles, Cloud, Moon, Fish, Cpu, MessageSquare, Zap, Settings2
} from 'lucide-react'

const GlassInput = ({ icon: Icon, placeholder, value, onChange, type = "text", disabled = false, trailing = null, className = '', ...props }) => (
  <div className="relative group w-full">
    {Icon && (
      <div className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-400 group-focus-within:text-[#B85F47] transition-colors">
        <Icon size={18} />
      </div>
    )}
    <input 
      type={type} 
      placeholder={placeholder}
      value={value}
      onChange={onChange}
      disabled={disabled}
      className={`w-full rounded-[16px] border border-[#e8e4df] bg-[#faf9f7] text-[14px] text-gray-800 font-medium focus:outline-none focus:ring-2 focus:ring-[#FFA07A]/20 focus:border-[#E8C4B4] transition-colors ${Icon ? 'pl-10' : 'pl-4'} ${trailing ? 'pr-12' : 'pr-4'} py-3 ${disabled ? 'opacity-60 cursor-not-allowed bg-gray-100' : ''} ${className}`}
      {...props}
    />
    {trailing && (
      <div className="absolute right-2.5 top-1/2 -translate-y-1/2">
        {trailing}
      </div>
    )}
  </div>
);

/**
 * 保留 <option> 子元素的写法，内部转成 SoftSelect 的 options 数组，
 * 这样 7 个调用点不必逐个改造结构。
 */
const GlassSelect = ({ value, onChange, children, disabled = false, 'aria-label': ariaLabel }) => {
  const options = React.Children.toArray(children)
    .filter(child => React.isValidElement(child))
    .map(child => ({ value: child.props.value, label: child.props.children }))

  return (
    <SoftSelect
      size="sm"
      value={value}
      options={options}
      disabled={disabled}
      ariaLabel={ariaLabel}
      onChange={next => onChange?.({ target: { value: next } })}
    />
  )
}

const Tag = ({ text, active, onClick }) => (
  <span 
    onClick={onClick}
    className={`${active ? 'bg-[#F3EDE8] text-[#5c564f]' : 'bg-[#f4f2ef] text-[#8a827b] hover:bg-[#ece8e3] hover:text-[#5c564f]'} text-[12px] font-medium px-3.5 py-1.5 rounded-full cursor-pointer transition-colors`}
  >
    {text}
  </span>
);

const PROVIDER_CAPABILITY_LABELS = [
  ['chat', '对话'],
  ['embedding', '向量'],
  ['rerank', '重排'],
  ['imageGeneration', '图像'],
]

const getProviderCapabilitySummary = provider => {
  const labels = PROVIDER_CAPABILITY_LABELS
    .filter(([key]) => provider.capabilities?.[key])
    .map(([, label]) => label)
  return labels.length > 0 ? labels.join(' · ') : '模型服务'
}

const LabeledField = ({ label, hint, children }) => (
  <label className="block space-y-1.5">
    <span className="flex items-baseline justify-between gap-2 px-0.5">
      <span className="text-[11px] font-bold text-gray-600">{label}</span>
      {hint && <span className="truncate text-[10px] font-medium text-gray-400">{hint}</span>}
    </span>
    {children}
  </label>
)

const CapabilityToggle = ({ checked, disabled = false, label, ariaLabel = label, onToggle }) => (
  <button
    type="button"
    role="checkbox"
    aria-checked={checked}
    aria-label={ariaLabel}
    disabled={disabled}
    title={label}
    onClick={() => onToggle(!checked)}
    className={`flex min-h-9 items-center gap-2 rounded-[10px] border px-2.5 py-2 text-[11px] font-semibold transition-colors ${
      disabled
        ? checked
          ? 'cursor-not-allowed border-[#F3B39D]/60 bg-[#FFF8F4] text-[#A8533D]/70'
          : 'cursor-not-allowed border-gray-200 bg-gray-50 text-gray-300'
        : checked
          ? 'border-[#F3B39D] bg-[#FFF4EF] text-[#A8533D]'
          : 'cursor-pointer border-gray-200 bg-white text-gray-500 hover:border-gray-300 hover:bg-gray-50'
    }`}
  >
    <span
      aria-hidden="true"
      className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-[5px] border transition-colors ${
        checked ? 'border-[#E98260] bg-[#F0653A] text-white' : 'border-gray-300 bg-white text-transparent'
      }`}
    >
      <Check size={11} strokeWidth={3} />
    </span>
    <span className="min-w-0 whitespace-nowrap">{label}</span>
  </button>
)

/* 扁平列表行：选中态 = 白色悬浮卡（中性投影），切换时卡片在行间滑动跟随 */
const ProviderItem = ({ provider, isActive, onClick, onDelete, isDeleting = false }) => {
  const capabilitySummary = getProviderCapabilitySummary(provider)
  const canDelete = !provider.isSystem && typeof onDelete === 'function'

  return (
  <div className="group relative">
    <button
      type="button"
      onClick={onClick}
      aria-pressed={isActive}
      className={`relative flex min-h-[62px] w-full items-center rounded-[18px] px-2.5 py-2 text-left transition-colors ${
        isActive ? '' : 'hover:bg-white/70'
      } ${canDelete ? 'pr-14' : ''}`}>
      {isActive && (
        <motion.span
          layoutId="provider-active-card"
          transition={{ type: 'spring', stiffness: 420, damping: 34, mass: 0.8 }}
          aria-hidden="true"
          className="absolute inset-0 rounded-[18px] bg-white ring-1 ring-black/[0.05]"
        >
          <span className="absolute left-0 top-1/2 h-6 w-[3px] -translate-y-1/2 rounded-full bg-[#F0653A]" />
        </motion.span>
      )}
      <div className="relative z-10 flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-[14px] bg-[#faf9f7] ring-1 ring-black/[0.04]">
        <ProviderAvatar providerId={provider.id} size={22} />
      </div>
      <div className="relative z-10 ml-2.5 flex-1 min-w-0">
        <h3
          className={`line-clamp-2 text-[13px] leading-[1.25] ${isActive ? 'font-extrabold text-gray-900' : 'font-bold text-gray-800'}`}
          title={provider.name}
        >
          {provider.name}
        </h3>
        <p className={`mt-0.5 truncate font-mono text-[10px] font-medium ${isActive ? 'text-[#B85F47]' : 'text-gray-400'}`} title={provider.id}>
          {provider.id}
        </p>
        <p className="mt-0.5 truncate text-[10px] font-medium text-gray-400" title={capabilitySummary}>
          {capabilitySummary}
        </p>
      </div>
      <span
        className={`relative z-10 ml-2 h-2 w-2 shrink-0 rounded-full ${provider.enabled ? 'bg-[#3FBE7C] ring-2 ring-[#3FBE7C]/20' : 'bg-gray-300'}`}
        title={provider.enabled ? '已启用' : '未启用'}
        aria-label={provider.enabled ? '已启用' : '未启用'}
      />
    </button>
    {canDelete && (
      <button
        type="button"
        onClick={event => {
          event.stopPropagation()
          onDelete()
        }}
        disabled={isDeleting}
        aria-label={`删除 ${provider.name}`}
        title="删除自定义服务"
        className="absolute right-7 top-1/2 z-20 flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-[9px] text-gray-400 opacity-0 transition-all hover:bg-red-50 hover:text-red-500 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-200 group-hover:opacity-100 disabled:cursor-wait disabled:opacity-60"
      >
        {isDeleting ? <RefreshCw size={14} className="animate-spin" /> : <Trash2 size={14} />}
      </button>
    )}
  </div>
  )
}
import { useProvider } from '../contexts/ProviderContext'
import { useModel } from '../contexts/ModelContext'
import { useDefaults } from '../contexts/DefaultsContext'
import ProviderAvatar from './ProviderAvatar'
import SoftSelect from './SoftSelect'
import { fetchModelsFromProvider } from '../services/modelService'

/**
 * 模型标签选项列表
 * 用于新增模型表单中的多选标签组件
 */
const TAG_OPTIONS = [
  { value: 'free', label: '免费' },
  { value: 'vision', label: '视觉' },
  { value: 'chinese_optimized', label: '中文优化' },
  { value: 'reasoning', label: '推理' },
  { value: 'function_calling', label: '函数调用' },
  { value: 'web_search', label: '网络搜索' },
]

/**
 * 标签值到中文显示名称的映射
 * 用于模型列表中标签徽章的渲染
 */
const TAG_LABELS = {
  free: '免费',
  vision: '视觉',
  chinese_optimized: '中文优化',
  reasoning: '推理',
  function_calling: '函数调用',
  web_search: '网络搜索',
  embedding: 'Embedding',
  rerank: 'Rerank',
}

const MODEL_TEST_TYPES = new Set(['chat', 'embedding', 'rerank'])

const CUSTOM_PROVIDER_PROTOCOL_PRESETS = {
  openai: {
    label: 'OpenAI Compatible',
    badge: 'Chat Completions',
    apiHost: '',
    chatEndpoint: '/chat/completions',
    apiKeyHeader: 'Authorization',
    apiKeyPrefix: 'Bearer ',
  },
  anthropic: {
    label: 'Anthropic Messages',
    badge: 'Messages API',
    apiHost: 'https://api.anthropic.com/v1',
    chatEndpoint: '/messages',
    apiKeyHeader: 'x-api-key',
    apiKeyPrefix: '',
  },
}

const createInitialCustomProviderForm = () => ({
  id: '',
  name: '',
  apiHost: '',
  apiKey: '',
  modelId: '',
  modelName: '',
  modelType: 'chat',
  protocol: 'openai',
  dimension: '',
  setAsDefault: true,
  fetchModelsEndpoint: '/models',
  chatEndpoint: '/chat/completions',
  embeddingEndpoint: '/embeddings',
  rerankEndpoint: '/rerank',
  supportsStreaming: true,
  supportsReasoning: false,
  reasoningMode: '',
  reasoningOptions: [],
  reasoningDefault: '',
  reasoningAlwaysEnabled: '',
  reasoningOffControl: '',
  reasoningOnControl: '',
  apiKeyHeader: 'Authorization',
  apiKeyPrefix: 'Bearer ',
  chat: true,
  embedding: false,
  rerank: false,
})

// 只用于当前页面内判断测试结果是否仍对应当前配置，不保存也不发送原始密钥。
const getSecretFingerprint = value => {
  const normalized = String(value || '')
  let hash = 0
  for (let index = 0; index < normalized.length; index += 1) {
    hash = (hash * 31 + normalized.charCodeAt(index)) >>> 0
  }
  return `${normalized.length}:${hash.toString(16)}`
}

// API 错误可能来自 FastAPI/Pydantic，detail 既可能是字符串，也可能是对象数组。
// 只提取安全的提示字段，避免把原始 input（可能包含密钥）或对象直接交给 React 渲染。
const formatApiMessage = (value, fallback = '') => {
  if (value === null || value === undefined || value === '') return fallback
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  if (Array.isArray(value)) {
    const messages = value
      .map(item => formatApiMessage(item))
      .filter(Boolean)
    return messages.length > 0 ? messages.join('；') : fallback
  }
  if (typeof value === 'object') {
    if ('detail' in value) return formatApiMessage(value.detail, fallback)
    if ('message' in value) return formatApiMessage(value.message, fallback)
    if ('error' in value) return formatApiMessage(value.error, fallback)
    if ('msg' in value) {
      const message = formatApiMessage(value.msg, fallback)
      const location = Array.isArray(value.loc)
        ? value.loc.filter(item => typeof item === 'string' || typeof item === 'number').join('.')
        : ''
      return location && message ? `${location}: ${message}` : message
    }
  }
  return fallback
}

const buildModelTestPayload = ({
  providerId,
  modelId,
  apiKey,
  apiHost,
  modelType,
  providerType,
  chatEndpoint,
  embeddingEndpoint,
  rerankEndpoint,
  apiKeyHeader,
  apiKeyPrefix,
}) => ({
  providerId,
  modelId,
  apiKey,
  apiHost,
  modelType,
  providerType,
  ...(modelType === 'chat' && chatEndpoint ? { chatEndpoint } : {}),
  ...(modelType === 'embedding' && embeddingEndpoint ? { embeddingEndpoint } : {}),
  ...(modelType === 'rerank' && rerankEndpoint ? { rerankEndpoint } : {}),
  apiKeyHeader,
  apiKeyPrefix,
})

/**
 * “模型服务管理”面板
 * 对齐 cherry-studio 的三栏结构：左侧 Provider 列表，中间连接配置，右侧模型清单。
 */
export default function EmbeddingSettings({ isOpen, onClose, onExitComplete }) {
  const {
    providers,
    addProvider,
    deleteProvider = async () => {},
    updateProvider,
    testConnection
  } = useProvider()

  const {
    userCollection,
    systemModels,
    addModelToCollection,
    removeModelFromCollection,
    removeModelsByProvider = () => {},
    fetchAndAddModels,
    isFetching,
    fetchError
  } = useModel()

  const { getDefaultModel, setDefaultModel, clearDefaultModel = () => {} } = useDefaults()

  const [activeProviderId, setActiveProviderId] = useState(
    providers[0]?.id || null
  )
  const [providerSearch, setProviderSearch] = useState('')
  const [providerApiKeyVisible, setProviderApiKeyVisible] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)
  const [collapsedTypes, setCollapsedTypes] = useState({ chat: true, embedding: true, rerank: true, image: true })
  const [addModelForm, setAddModelForm] = useState({
    id: '',
    name: '',
    type: 'chat',
    dimension: ''
  })
  // 新增模型表单的标签选择状态
  const [newModelTags, setNewModelTags] = useState([])
  const [addSuccess, setAddSuccess] = useState(null)
  const [lastAddedModelKey, setLastAddedModelKey] = useState(null)
  // Optimistic update：保存进局部 state，不依赖 context 传播时序
  const [locallyAddedModels, setLocallyAddedModels] = useState([])
  const [modelTestResults, setModelTestResults] = useState({})
  const [testingModelKey, setTestingModelKey] = useState('')
  const [customProviderFormOpen, setCustomProviderFormOpen] = useState(false)
  const [customProviderSaving, setCustomProviderSaving] = useState(false)
  const [customProviderError, setCustomProviderError] = useState('')
  const [customProviderForm, setCustomProviderForm] = useState(createInitialCustomProviderForm)
  const [customProviderAdvancedOpen, setCustomProviderAdvancedOpen] = useState(false)
  const [customProviderApiKeyVisible, setCustomProviderApiKeyVisible] = useState(false)
  const [customProviderTesting, setCustomProviderTesting] = useState(false)
  const [customProviderTestResult, setCustomProviderTestResult] = useState(null)
  const [customProviderFetchingModels, setCustomProviderFetchingModels] = useState(false)
  const [customProviderFetchResult, setCustomProviderFetchResult] = useState(null)
  const [customProviderModels, setCustomProviderModels] = useState([])
  const [customProviderModelTags, setCustomProviderModelTags] = useState([])
  const [deletingProviderId, setDeletingProviderId] = useState('')
  const [providerDeleteError, setProviderDeleteError] = useState('')

  // 默认模型键名映射
  const DEFAULT_TYPE_MAP = {
    embedding: 'embeddingModel',
    rerank: 'rerankModel',
    chat: 'assistantModel'
  }
  const TYPE_META = {
    chat: { label: 'Chat 对话' },
    embedding: { label: 'Embedding 向量' },
    rerank: { label: 'Rerank 重排' },
    image: { label: 'Image 图像' }
  }
  const GLASS_CARD_CLASS = 'soft-card'
  const RADIUS_CLASS = 'rounded-[18px]'

  const activeProvider = useMemo(
    () => providers.find(p => p.id === activeProviderId) || providers[0] || null,
    [providers, activeProviderId]
  )

  // 当 providers 变化时，保持选中第一项
  useEffect(() => {
    if (!activeProvider && providers.length > 0) {
      setActiveProviderId(providers[0].id)
    }
  }, [providers, activeProvider])

  // 切换服务时收起密钥明文，避免上一个服务的 Key 继续暴露。
  useEffect(() => {
    setProviderApiKeyVisible(false)
  }, [activeProviderId])

  const filteredProviders = useMemo(() => {
    const normalizedSearch = providerSearch.trim().toLowerCase()
    if (!normalizedSearch) return providers
    return providers.filter(p =>
      `${p.name} ${p.id}`.toLowerCase().includes(normalizedSearch)
    )
  }, [providerSearch, providers])

  // 直接订阅 userCollection + systemModels 基础状态（对齐 Cherry Studio 单一数据源模式）
  // 避免通过 getModelsByProvider 函数派生，确保状态更新时立即反映在 UI 上
  const modelsByType = useMemo(() => {
    if (!activeProvider) return {}
    const sysForProvider = systemModels.filter(m => m.providerId === activeProvider.id)
    const userForProvider = userCollection.filter(m => m.providerId === activeProvider.id)
    const localForProvider = locallyAddedModels.filter(m => m.providerId === activeProvider.id)
    // 优先级：local > user > system
    const map = new Map()
    sysForProvider.forEach(m => map.set(m.id, m))
    userForProvider.forEach(m => map.set(m.id, m))
    localForProvider.forEach(m => map.set(m.id, m))
    const list = Array.from(map.values())

    const grouped = list.reduce((acc, model) => {
      acc[model.type] = acc[model.type] || []
      acc[model.type].push(model)
      return acc
    }, {})

    Object.values(grouped).forEach(group => {
      group.sort((a, b) => {
        const aKey = `${a.providerId}:${a.id}`
        const bKey = `${b.providerId}:${b.id}`
        if (aKey === lastAddedModelKey) return -1
        if (bKey === lastAddedModelKey) return 1
        if (!!a.isUserAdded !== !!b.isUserAdded) return a.isUserAdded ? -1 : 1
        return (a.name || a.id).localeCompare(b.name || b.id, 'zh-Hans-CN', { sensitivity: 'base' })
      })
    })

    return grouped
  }, [activeProvider, systemModels, userCollection, locallyAddedModels, lastAddedModelKey])

  // 自定义 Provider 下的模型是独立适配器：同一个 endpoint 可以有多个
  // model ID，但每个模型都要有自己的测试、默认和删除状态。
  const providerModels = useMemo(() => {
    const map = new Map()
    Object.values(modelsByType).flat().forEach(model => {
      const key = `${model.providerId}:${model.id}`
      if (!map.has(key)) map.set(key, model)
    })
    return Array.from(map.values())
  }, [modelsByType])

  const isCustomProvider = Boolean(activeProvider && !activeProvider.isSystem)

  const handleProviderUpdate = (field, value) => {
    if (!activeProvider) return
    updateProvider(activeProvider.id, { [field]: value })
  }

  const handleProviderApiConfigUpdate = (field, value) => {
    if (!activeProvider) return
    updateProvider(activeProvider.id, {
      apiConfig: {
        ...(activeProvider.apiConfig || {}),
        [field]: value,
      },
    })
  }

  const handleDeleteCustomProvider = async provider => {
    if (!provider || provider.isSystem || deletingProviderId) return
    const providerName = provider.name || provider.id
    if (!window.confirm(`确定删除“${providerName}”吗？\n该服务下的自定义模型和默认模型引用也会被移除。`)) return

    setProviderDeleteError('')
    setDeletingProviderId(provider.id)
    try {
      await deleteProvider(provider.id)
      removeModelsByProvider(provider.id)
      setLocallyAddedModels(prev => prev.filter(model => model.providerId !== provider.id))
      setModelTestResults(prev => Object.fromEntries(
        Object.entries(prev).filter(([key]) => !key.startsWith(`${provider.id}:`))
      ))
      Object.entries(DEFAULT_TYPE_MAP).forEach(([, defaultKey]) => {
        const currentDefault = getDefaultModel(defaultKey)
        if (typeof currentDefault === 'string' && currentDefault.startsWith(`${provider.id}:`)) {
          clearDefaultModel(defaultKey)
        }
      })

      if (activeProviderId === provider.id) {
        const nextProvider = providers.find(item => item.id !== provider.id)
        setActiveProviderId(nextProvider?.id || null)
        setTestResult(null)
      }
      setLastAddedModelKey(null)
      setAddSuccess(`已删除 ${providerName}`)
      window.setTimeout(() => setAddSuccess(null), 3000)
    } catch (error) {
      setProviderDeleteError(error instanceof Error ? error.message : '删除自定义模型服务失败')
    } finally {
      setDeletingProviderId(current => current === provider.id ? '' : current)
    }
  }

  const handleTest = async () => {
    if (!activeProvider) {
      setTesting(false)
      return
    }
    setTesting(true)
    setTestResult(null)
    const preferredChatModel = modelsByType.chat?.[0]?.id
    const result = await testConnection(activeProvider.id, { modelId: preferredChatModel })
    setTestResult(result)
    setTesting(false)
  }

  const handleSyncModels = async () => {
    if (!activeProvider) return
    await fetchAndAddModels(activeProvider)
  }

  const getModelAdapterKey = model => `${model.providerId}:${model.id}`

  const getModelAdapterSignature = model => [
    activeProvider?.id || '',
    activeProvider?.apiHost || '',
    getSecretFingerprint(activeProvider?.apiKey),
    activeProvider?.apiConfig?.protocol || '',
    activeProvider?.apiConfig?.chatEndpoint || '',
    activeProvider?.apiConfig?.embeddingEndpoint || '',
    activeProvider?.apiConfig?.rerankEndpoint || '',
    activeProvider?.apiConfig?.apiKeyHeader || '',
    activeProvider?.apiConfig?.apiKeyPrefix || '',
    model.id,
    model.type,
  ].join('|')

  const handleTestModel = async model => {
    if (!activeProvider || !MODEL_TEST_TYPES.has(model.type)) return
    const adapterKey = getModelAdapterKey(model)
    const signature = getModelAdapterSignature(model)
    setTestingModelKey(adapterKey)
    try {
      const response = await fetch('/api/models/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildModelTestPayload({
          providerId: activeProvider.id,
          modelId: model.id,
          apiKey: activeProvider.apiKey || '',
          apiHost: activeProvider.apiHost || '',
          modelType: model.type,
          providerType: activeProvider.apiConfig?.protocol,
          chatEndpoint: activeProvider.apiConfig?.chatEndpoint,
          embeddingEndpoint: activeProvider.apiConfig?.embeddingEndpoint,
          rerankEndpoint: activeProvider.apiConfig?.rerankEndpoint,
          apiKeyHeader: activeProvider.apiConfig?.apiKeyHeader,
          apiKeyPrefix: activeProvider.apiConfig?.apiKeyPrefix,
        })),
      })
      const payload = await response.json().catch(() => ({}))
      const result = response.ok
        ? {
          ...payload,
          message: formatApiMessage(payload.message, payload.success === false ? '测试失败' : ''),
        }
        : {
          success: false,
          message: formatApiMessage(payload.detail ?? payload.message, `请求失败（${response.status}）`),
        }
      setModelTestResults(prev => ({
        ...prev,
        [adapterKey]: { ...result, signature, testedAt: Date.now() },
      }))
    } catch (error) {
      setModelTestResults(prev => ({
        ...prev,
        [adapterKey]: {
          success: false,
          message: formatApiMessage(error instanceof Error ? error.message : error, '模型测试失败'),
          signature,
          testedAt: Date.now(),
        },
      }))
    } finally {
      setTestingModelKey(current => current === adapterKey ? '' : current)
    }
  }

  const handleAddModel = () => {
    if (!activeProvider || !addModelForm.id.trim()) return
    const modelType = addModelForm.type
    if (modelType === 'embedding' && (!Number.isInteger(Number(addModelForm.dimension)) || Number(addModelForm.dimension) <= 0)) {
      alert('Embedding 模型需要填写有效的向量维度')
      return
    }
    const newId = addModelForm.id.trim()
    const newModel = {
      id: newId,
      name: addModelForm.name.trim() || newId,
      providerId: activeProvider.id,
      type: modelType,
      capabilities: [{ type: modelType, isUserSelected: true }],
      tags: newModelTags,
      metadata: modelType === 'embedding' && addModelForm.dimension
        ? { dimension: Number(addModelForm.dimension) }
        : {},
      isSystem: false,
      isUserAdded: true
    }
    // 将模型写入局部 state（立即显示）和 context（持久化）
    setLocallyAddedModels(prev => {
      const exists = prev.some(m => m.id === newId && m.providerId === activeProvider.id)
      return exists ? prev : [...prev, newModel]
    })
    addModelToCollection(newModel)
    setLastAddedModelKey(`${activeProvider.id}:${newId}`)
    // 自动展开对应类型的分组，让用户看到新增的模型
    setCollapsedTypes(prev => ({ ...prev, [modelType]: false }))
    setAddSuccess(`已添加: ${newId} (${modelType})`)
    setTimeout(() => setAddSuccess(null), 3000)
    setAddModelForm({ id: '', name: '', type: 'chat', dimension: '' })
    // 重置标签选择
    setNewModelTags([])
  }

  const buildDefaultKey = (type, modelId) => `${activeProvider?.id || ''}:${modelId}`
  const isDefaultModel = (type, modelId) => {
    const key = DEFAULT_TYPE_MAP[type]
    if (!key) return false
    return getDefaultModel(key) === buildDefaultKey(type, modelId)
  }

  const handleSetDefault = (type, modelId) => {
    const key = DEFAULT_TYPE_MAP[type]
    if (!key) return
    setDefaultModel(key, buildDefaultKey(type, modelId))
  }

  const toggleCollapse = (type) => {
    setCollapsedTypes(prev => ({
      ...prev,
      [type]: !prev?.[type]
    }))
  }

  const updateCustomProviderForm = (field, value) => {
    setCustomProviderError('')
    setCustomProviderTestResult(null)
    if (['apiHost', 'apiKey', 'fetchModelsEndpoint', 'apiKeyHeader', 'apiKeyPrefix'].includes(field)) {
      setCustomProviderFetchResult(null)
    }
    setCustomProviderForm(prev => ({ ...prev, [field]: value }))
  }

  const resetCustomProviderDraft = () => {
    setCustomProviderForm(createInitialCustomProviderForm())
    setCustomProviderModelTags([])
    setCustomProviderModels([])
    setCustomProviderFetchResult(null)
    setCustomProviderTestResult(null)
    setCustomProviderError('')
    setCustomProviderApiKeyVisible(false)
    setCustomProviderAdvancedOpen(false)
  }

  const handleCustomProviderProtocolChange = protocol => {
    const nextProtocol = protocol === 'anthropic' ? 'anthropic' : 'openai'
    const nextPreset = CUSTOM_PROVIDER_PROTOCOL_PRESETS[nextProtocol]
    setCustomProviderError('')
    setCustomProviderTestResult(null)
    setCustomProviderFetchResult(null)
    setCustomProviderForm(prev => {
      const previousProtocol = prev.protocol === 'anthropic' ? 'anthropic' : 'openai'
      const previousPreset = CUSTOM_PROVIDER_PROTOCOL_PRESETS[previousProtocol]
      const shouldReplace = (value, previousValue) => !String(value || '').trim() || value === previousValue
      const nextIsAnthropic = nextProtocol === 'anthropic'
      return {
        ...prev,
        protocol: nextProtocol,
        apiHost: shouldReplace(prev.apiHost, previousPreset.apiHost) ? nextPreset.apiHost : prev.apiHost,
        chatEndpoint: shouldReplace(prev.chatEndpoint, previousPreset.chatEndpoint) ? nextPreset.chatEndpoint : prev.chatEndpoint,
        apiKeyHeader: shouldReplace(prev.apiKeyHeader, previousPreset.apiKeyHeader) ? nextPreset.apiKeyHeader : prev.apiKeyHeader,
        apiKeyPrefix: shouldReplace(prev.apiKeyPrefix, previousPreset.apiKeyPrefix) ? nextPreset.apiKeyPrefix : prev.apiKeyPrefix,
        modelType: nextIsAnthropic ? 'chat' : prev.modelType,
        chat: nextIsAnthropic ? true : prev.chat,
        embedding: nextIsAnthropic ? false : prev.embedding,
        rerank: nextIsAnthropic ? false : prev.rerank,
      }
    })
  }

  const buildCustomProviderDraft = () => ({
    id: customProviderForm.id.trim() || 'custom-preview',
    name: customProviderForm.name.trim() || customProviderForm.id.trim() || '自定义模型服务',
    apiKey: customProviderForm.apiKey.trim(),
    apiHost: customProviderForm.apiHost.trim(),
    enabled: true,
    isSystem: false,
    capabilities: {
      chat: customProviderForm.protocol === 'anthropic' || customProviderForm.chat || customProviderForm.modelType === 'chat',
      embedding: customProviderForm.protocol !== 'anthropic' && (customProviderForm.embedding || customProviderForm.modelType === 'embedding'),
      rerank: customProviderForm.protocol !== 'anthropic' && (customProviderForm.rerank || customProviderForm.modelType === 'rerank'),
    },
    apiConfig: {
      protocol: customProviderForm.protocol || 'openai',
      fetchModelsEndpoint: customProviderForm.fetchModelsEndpoint.trim() || '/models',
      chatEndpoint: customProviderForm.chatEndpoint.trim() || (customProviderForm.protocol === 'anthropic' ? '/messages' : '/chat/completions'),
      embeddingEndpoint: customProviderForm.embeddingEndpoint.trim() || '/embeddings',
      rerankEndpoint: customProviderForm.rerankEndpoint.trim(),
      supportsStreaming: customProviderForm.supportsStreaming,
      supportsReasoning: customProviderForm.supportsReasoning,
      ...(customProviderForm.supportsReasoning ? {
        reasoningMode: customProviderForm.reasoningMode || undefined,
        reasoningOptions: customProviderForm.reasoningOptions.length > 0
          ? customProviderForm.reasoningOptions
          : undefined,
        reasoningDefault: customProviderForm.reasoningDefault || undefined,
        reasoningAlwaysEnabled: customProviderForm.reasoningAlwaysEnabled === ''
          ? undefined
          : customProviderForm.reasoningAlwaysEnabled === 'true',
        reasoningOffControl: customProviderForm.reasoningOffControl || undefined,
        reasoningOnControl: customProviderForm.reasoningOnControl || undefined,
      } : {}),
      apiKeyHeader: customProviderForm.apiKeyHeader.trim() || (customProviderForm.protocol === 'anthropic' ? 'x-api-key' : 'Authorization'),
      apiKeyPrefix: customProviderForm.apiKeyPrefix,
    },
  })

  const validateCustomProviderDraft = ({ requireIdentity = false, requireModel = false } = {}) => {
    if (requireIdentity && (!customProviderForm.id.trim() || !customProviderForm.name.trim())) {
      return '请填写服务标识和显示名称'
    }
    if (!customProviderForm.apiHost.trim()) return '请填写 API 地址'
    if (!customProviderForm.apiKey.trim()) return '请填写访问密钥'
    if (requireModel && !customProviderForm.modelId.trim()) return '请填写或获取一个模型 ID'
    if (
      requireModel
      && customProviderForm.modelType === 'embedding'
      && (!Number.isInteger(Number(customProviderForm.dimension)) || Number(customProviderForm.dimension) <= 0)
    ) {
      return 'Embedding 模型需要填写有效的向量维度'
    }
    if (requireIdentity && providers.some(provider => provider.id === customProviderForm.id.trim())) {
      return '这个服务标识已存在，请换一个唯一标识'
    }
    return ''
  }

  const handleCustomModelTypeChange = value => {
    if (customProviderForm.protocol === 'anthropic' && value !== 'chat') return
    setCustomProviderError('')
    setCustomProviderTestResult(null)
    setCustomProviderForm(prev => ({
      ...prev,
      modelType: value,
      chat: value === 'chat' ? true : prev.chat,
      embedding: value === 'embedding' ? true : prev.embedding,
      rerank: value === 'rerank' ? true : prev.rerank,
    }))
  }

  const handleCustomModelIdChange = value => {
    const matched = customProviderModels.find(model => model.id === value)
    setCustomProviderError('')
    setCustomProviderTestResult(null)
    setCustomProviderForm(prev => ({
      ...prev,
      modelId: value,
      modelName: matched?.name && matched.name !== matched.id ? matched.name : prev.modelName,
      modelType: prev.protocol === 'anthropic'
        ? 'chat'
        : (matched?.type && MODEL_TEST_TYPES.has(matched.type) ? matched.type : prev.modelType),
      dimension: matched?.metadata?.dimension ? String(matched.metadata.dimension) : prev.dimension,
      chat: prev.protocol === 'anthropic' ? true : (matched?.type === 'chat' ? true : prev.chat),
      embedding: prev.protocol === 'anthropic' ? false : (matched?.type === 'embedding' ? true : prev.embedding),
      rerank: prev.protocol === 'anthropic' ? false : (matched?.type === 'rerank' ? true : prev.rerank),
    }))
    if (matched?.tags?.length) setCustomProviderModelTags(matched.tags)
  }

  const handleFetchCustomProviderModels = async () => {
    setCustomProviderError('')
    setCustomProviderFetchResult(null)
    const validationError = validateCustomProviderDraft()
    if (validationError) {
      setCustomProviderError(validationError)
      return
    }

    setCustomProviderFetchingModels(true)
    try {
      const models = (await fetchModelsFromProvider(buildCustomProviderDraft()))
        .filter(model => MODEL_TEST_TYPES.has(model.type))
      setCustomProviderModels(models)
      if (models.length === 0) {
        setCustomProviderFetchResult({ success: false, message: '服务未返回可用模型，请手动填写模型 ID' })
        return
      }
      setCustomProviderFetchResult({ success: true, message: `已获取 ${models.length} 个模型` })
      if (!customProviderForm.modelId.trim()) {
        const first = models[0]
        setCustomProviderForm(prev => ({
          ...prev,
          modelId: first.id,
          modelName: first.name && first.name !== first.id ? first.name : prev.modelName,
          modelType: prev.protocol === 'anthropic' ? 'chat' : (MODEL_TEST_TYPES.has(first.type) ? first.type : prev.modelType),
          dimension: first.metadata?.dimension ? String(first.metadata.dimension) : prev.dimension,
          chat: prev.protocol === 'anthropic' ? true : (first.type === 'chat' ? true : prev.chat),
          embedding: prev.protocol === 'anthropic' ? false : (first.type === 'embedding' ? true : prev.embedding),
          rerank: prev.protocol === 'anthropic' ? false : (first.type === 'rerank' ? true : prev.rerank),
        }))
        if (first.tags?.length) setCustomProviderModelTags(first.tags)
      }
    } catch (error) {
      setCustomProviderFetchResult({
        success: false,
        message: error instanceof Error ? error.message : '获取模型失败',
      })
    } finally {
      setCustomProviderFetchingModels(false)
    }
  }

  const handleTestCustomProvider = async () => {
    setCustomProviderError('')
    setCustomProviderTestResult(null)
    const validationError = validateCustomProviderDraft({ requireModel: true })
    if (validationError) {
      setCustomProviderError(validationError)
      return
    }

    const provider = buildCustomProviderDraft()
    setCustomProviderTesting(true)
    try {
      const response = await fetch('/api/models/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildModelTestPayload({
          providerId: provider.id,
          modelId: customProviderForm.modelId.trim(),
          apiKey: provider.apiKey,
          apiHost: provider.apiHost,
          modelType: customProviderForm.modelType,
          providerType: provider.apiConfig.protocol,
          chatEndpoint: provider.apiConfig.chatEndpoint,
          embeddingEndpoint: provider.apiConfig.embeddingEndpoint,
          rerankEndpoint: provider.apiConfig.rerankEndpoint,
          apiKeyHeader: provider.apiConfig.apiKeyHeader,
          apiKeyPrefix: provider.apiConfig.apiKeyPrefix,
        })),
      })
      const payload = await response.json().catch(() => ({}))
      const success = response.ok && payload.success !== false
      setCustomProviderTestResult({
        success,
        message: success
          ? formatApiMessage(payload.message, '模型调用成功')
          : formatApiMessage(payload.detail ?? payload.message, `测试失败（${response.status}）`),
        latency: payload.responseTime ?? payload.latency,
      })
    } catch (error) {
      setCustomProviderTestResult({
        success: false,
        message: formatApiMessage(error instanceof Error ? error.message : error, '测试请求失败'),
      })
    } finally {
      setCustomProviderTesting(false)
    }
  }

  const handleAddCustomProvider = async () => {
    setCustomProviderError('')
    const validationError = validateCustomProviderDraft({ requireIdentity: true, requireModel: true })
    if (validationError) {
      setCustomProviderError(validationError)
      return
    }

    const providerId = customProviderForm.id.trim()
    const modelId = customProviderForm.modelId.trim()
    const provider = buildCustomProviderDraft()
    const fetchedModel = customProviderModels.find(model => model.id === modelId)
    const modelType = customProviderForm.modelType
    const model = {
      ...(fetchedModel || {}),
      id: modelId,
      name: customProviderForm.modelName.trim() || fetchedModel?.name || modelId,
      providerId,
      type: modelType,
      capabilities: [{ type: modelType, isUserSelected: true }],
      tags: customProviderModelTags,
      metadata: {
        ...(fetchedModel?.metadata || {}),
        ...(modelType === 'embedding' ? { dimension: Number(customProviderForm.dimension) } : {}),
        ...(modelType === 'chat' && customProviderForm.supportsReasoning ? {
          ...(customProviderForm.reasoningMode ? { reasoningMode: customProviderForm.reasoningMode } : {}),
          ...(customProviderForm.reasoningOptions.length > 0 ? { reasoningOptions: customProviderForm.reasoningOptions } : {}),
          ...(customProviderForm.reasoningDefault ? { reasoningDefault: customProviderForm.reasoningDefault } : {}),
          ...(customProviderForm.reasoningAlwaysEnabled !== '' ? { reasoningAlwaysEnabled: customProviderForm.reasoningAlwaysEnabled === 'true' } : {}),
          ...(customProviderForm.reasoningOffControl ? { reasoningOffControl: customProviderForm.reasoningOffControl } : {}),
          ...(customProviderForm.reasoningOnControl ? { reasoningOnControl: customProviderForm.reasoningOnControl } : {}),
        } : {}),
      },
      isSystem: false,
      isUserAdded: true,
    }
    setCustomProviderSaving(true)
    try {
      await addProvider(provider)
      addModelToCollection(model)
      setLocallyAddedModels(prev => {
        const exists = prev.some(item => item.id === modelId && item.providerId === providerId)
        return exists ? prev : [...prev, model]
      })
      setLastAddedModelKey(`${providerId}:${modelId}`)
      setCollapsedTypes(prev => ({ ...prev, [modelType]: false }))
      if (customProviderForm.setAsDefault && DEFAULT_TYPE_MAP[modelType]) {
        setDefaultModel(DEFAULT_TYPE_MAP[modelType], `${providerId}:${modelId}`)
      }
      setActiveProviderId(providerId)
      setCustomProviderFormOpen(false)
      setAddSuccess(`已添加 ${customProviderForm.name.trim()} · ${model.name}`)
      setTimeout(() => setAddSuccess(null), 3500)
      resetCustomProviderDraft()
    } catch (error) {
      setCustomProviderError(error instanceof Error ? error.message : '保存自定义 Provider 失败')
    } finally {
      setCustomProviderSaving(false)
    }
  }

  /**
   * 根据模型ID获取对应的图标Provider ID
   * 用于在聚合厂商（如SiliconFlow）中显示具体的模型厂商图标
   */
  const getIconProviderId = (model) => {
    // 如果不是聚合厂商，直接返回原providerId
    // 目前主要针对 silicon (SiliconFlow) 做特殊处理，也可以扩展到其他聚合厂商
    if (model.providerId !== 'silicon' && model.providerId !== 'openrouter') {
      return model.providerId
    }

    const modelId = model.id.toLowerCase()

    // 映射规则
    if (modelId.includes('qwen')) return 'qwen'
    if (modelId.includes('deepseek')) return 'deepseek'
    if (modelId.includes('thudm') || modelId.includes('glm')) return 'zhipu'
    if (modelId.includes('01-ai') || modelId.includes('yi-')) return 'yi'
    if (modelId.includes('mistral')) return 'mistral'
    if (modelId.includes('google') || modelId.includes('gemma')) return 'google'
    if (modelId.includes('meta') || modelId.includes('llama')) return 'meta' // We don't have meta icon yet, might fallback to local or silicon
    if (modelId.includes('nvidia')) return 'nvidia'
    if (modelId.includes('baichuan')) return 'baichuan'
    if (modelId.includes('internlm')) return 'internlm' // No icon yet
    if (modelId.includes('hunyuan')) return 'hunyuan'
    if (modelId.includes('step')) return 'step'
    if (modelId.includes('cohere')) return 'cohere'

    return model.providerId
  }

  const renderCustomModelCard = model => {
    const adapterKey = getModelAdapterKey(model)
    const signature = getModelAdapterSignature(model)
    const storedResult = modelTestResults[adapterKey]
    const testResult = storedResult?.signature === signature ? storedResult : null
    const isTesting = testingModelKey === adapterKey
    const canTestModel = MODEL_TEST_TYPES.has(model.type)
    const canSetDefault = Boolean(DEFAULT_TYPE_MAP[model.type])
    const typeLabel = TYPE_META[model.type]?.label || model.type
    const isDefault = isDefaultModel(model.type, model.id)
    const isRecentlyAdded = adapterKey === lastAddedModelKey

    return (
      <motion.article
        layout
        key={adapterKey}
        initial={{ opacity: 0, y: 5 }}
        animate={{ opacity: 1, y: 0 }}
        className={`group rounded-[18px] border bg-[#faf9f7] p-3.5 transition-colors ${
          isRecentlyAdded
            ? 'border-green-200 bg-green-50/50'
            : 'border-transparent hover:bg-[#f3efe9]'
        }`}
      >
        <div className="flex items-start gap-3 min-w-0">
          <div className="w-9 h-9 rounded-[11px] bg-[#FFF7F3] border border-[#F8DED3] flex items-center justify-center shrink-0">
            <ProviderAvatar providerId={getIconProviderId(model)} size={22} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5 min-w-0">
              <h4 className="text-[13px] font-bold text-gray-900 truncate" title={model.name || model.id}>
                {model.name || model.id}
              </h4>
              {isDefault && (
                <span className="shrink-0 text-[10px] font-bold text-[#B85F47] bg-[#FFF1EA] px-1.5 py-0.5 rounded-md">
                  默认
                </span>
              )}
            </div>
            <p className="mt-1 truncate text-[11px] text-gray-500 font-mono" title={model.id}>
              {model.id}
            </p>
            <div className="mt-2 flex items-center gap-1.5 flex-wrap">
              <span className="text-[10px] font-semibold text-gray-600 bg-gray-100 px-1.5 py-0.5 rounded-md">
                {typeLabel}
              </span>
              {model.metadata?.dimension && (
                <span className="text-[10px] font-semibold text-amber-700 bg-amber-50 px-1.5 py-0.5 rounded-md">
                  {model.metadata.dimension}维
                </span>
              )}
              {(model.tags || []).slice(0, 3).map(tag => (
                <span key={tag} className="text-[10px] font-medium text-gray-500 bg-gray-50 px-1.5 py-0.5 rounded-md">
                  {TAG_LABELS[tag] || tag}
                </span>
              ))}
              {isRecentlyAdded && (
                <span className="text-[10px] font-semibold text-green-700 bg-green-50 px-1.5 py-0.5 rounded-md">
                  刚同步
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="mt-3 pt-2.5 border-t border-gray-100 flex items-center justify-between gap-2">
          <div className={`min-w-0 flex items-center gap-1.5 text-[11px] font-medium ${
            isTesting
              ? 'text-[#B85F47]'
              : testResult?.success
                ? 'text-green-700'
                : testResult
                  ? 'text-red-600'
                  : 'text-gray-400'
          }`} title={testResult?.message || undefined}>
            {isTesting && <RefreshCw size={13} className="animate-spin shrink-0" />}
            {!isTesting && testResult?.success && <CheckCircle2 size={13} className="shrink-0" />}
            {!isTesting && testResult && !testResult.success && <AlertCircle size={13} className="shrink-0" />}
            <span className="truncate">
              {!canTestModel
                ? '暂不支持连通测试'
                : isTesting
                ? '测试中'
                : testResult?.success
                  ? `可用${testResult.responseTime ? ` · ${testResult.responseTime}ms` : ''}`
                  : testResult
                    ? (testResult.message || '测试失败')
                    : (activeProvider?.apiKey?.trim() ? '尚未测试' : '先填写 API Key')}
            </span>
          </div>

          <div className="flex items-center gap-1 shrink-0">
            {canTestModel && (
              <button
                type="button"
                onClick={() => handleTestModel(model)}
                disabled={Boolean(testingModelKey) || !activeProvider?.apiKey?.trim()}
                className="inline-flex items-center gap-1 px-2 py-1.5 rounded-[9px] text-[11px] font-semibold text-[#B85F47] hover:bg-[#FFF1EA] disabled:text-gray-300 disabled:hover:bg-transparent transition-colors"
                title={activeProvider?.apiKey?.trim() ? '测试模型连接' : '请先填写 API Key'}
              >
                {isTesting ? <RefreshCw size={13} className="animate-spin" /> : <Play size={13} />}
                测试
              </button>
            )}
            {canSetDefault && !isDefault && (
              <button
                type="button"
                onClick={() => handleSetDefault(model.type, model.id)}
                className="inline-flex items-center gap-1 rounded-full bg-[#F3EDE8] px-2.5 py-1 text-[11px] font-medium text-[#5c564f] transition-colors hover:bg-[#ece6df] hover:text-[#3f3a35]"
                title="设为默认模型"
                aria-label={`将 ${model.name || model.id} 设为默认模型`}
              >
                <CheckCircle2 size={13} strokeWidth={2.1} />
                设为默认
              </button>
            )}
            {model.isUserAdded && (
              <button
                type="button"
                onClick={() => removeModelFromCollection(model.id, model.providerId)}
                className="p-1.5 rounded-[9px] text-gray-400 hover:text-red-500 hover:bg-red-50 transition-colors"
                title="移除模型"
                aria-label={`移除 ${model.name || model.id}`}
              >
                <Trash2 size={15} />
              </button>
            )}
          </div>
        </div>
      </motion.article>
    )
  }

  const renderModelRow = (model) => {
    const isRecentlyAdded = `${model.providerId}:${model.id}` === lastAddedModelKey
    return (
    <div
      key={`${model.providerId}-${model.id}`}
      className={`group flex items-center justify-between px-4 py-3 rounded-xl border transition-all ${isRecentlyAdded ? 'bg-green-50 border-green-200 shadow-sm' : 'hover:bg-[var(--color-bg-subtle)] border-transparent hover:border-purple-100'}`}
    >
      <div className="flex items-center gap-4 overflow-hidden">
        <ProviderAvatar providerId={getIconProviderId(model)} size={36} className="flex-shrink-0 shadow-sm" />
        <div className="min-w-0 flex flex-col gap-0.5">
          <div className="flex items-center gap-2 flex-wrap">
            <div className="text-sm font-bold text-gray-900 truncate" title={model.name || model.id}>
              {model.name || model.id}
            </div>
            {isRecentlyAdded && (
              <span className="flex-shrink-0 text-[10px] font-medium text-green-700 bg-green-100 px-1.5 py-0.5 rounded border border-green-200">
                刚添加
              </span>
            )}
            {model.metadata?.dimension && (
              <span className="flex-shrink-0 text-[10px] font-medium text-amber-700 bg-amber-50 px-1.5 py-0.5 rounded border border-amber-100">
                {model.metadata.dimension}维
              </span>
            )}
            {/* 模型标签徽章 */}
            {model.tags?.map(tag => (
              <span key={tag} className="flex-shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-gray-50 text-gray-600 border border-gray-100">
                {TAG_LABELS[tag] || tag}
              </span>
            ))}
          </div>
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="truncate max-w-[200px]" title={model.id}>{model.id}</span>
            {model.type === 'chat' && (
              <span className="flex-shrink-0 px-1.5 py-0.5 rounded bg-purple-50 text-purple-600 text-[10px] font-medium border border-purple-100">
                Chat
              </span>
            )}
            {model.type === 'embedding' && (
              <span className="flex-shrink-0 px-1.5 py-0.5 rounded bg-purple-50 text-purple-600 text-[10px] font-medium border border-purple-100">
                Embedding
              </span>
            )}
            {model.type === 'rerank' && (
              <span className="flex-shrink-0 px-1.5 py-0.5 rounded bg-green-50 text-green-600 text-[10px] font-medium border border-green-100">
                Rerank
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 pl-4 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={() => handleSetDefault(model.type, model.id)}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${isDefaultModel(model.type, model.id)
            ? 'bg-purple-600 text-white shadow-sm hover:bg-purple-700'
            : 'bg-white border border-gray-200 text-gray-600 hover:border-purple-300 hover:text-purple-600'
            }`}
        >
          {isDefaultModel(model.type, model.id) ? '默认' : '设为默认'}
        </button>
        {model.isUserAdded && (
          <button
            onClick={() => removeModelFromCollection(model.id, model.providerId)}
            className="p-1.5 rounded-lg text-gray-400 hover:text-red-600 hover:bg-red-50 transition-colors"
            title="删除模型"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>
      {/* Always show default badge if it is default, even when not hovering */}
      {isDefaultModel(model.type, model.id) && (
        <div className="group-hover:hidden px-3 py-1.5 rounded-lg text-xs font-medium bg-purple-50 text-purple-700 border border-purple-100">
          默认
        </div>
      )}
    </div>
  )
  }

  try {
    return (
      <AnimatePresence initial={false} onExitComplete={onExitComplete}>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.1 }}
            aria-hidden={!isOpen}
            className={`fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/25 p-3 sm:p-6 font-sans overflow-hidden ${isOpen ? 'pointer-events-auto' : 'pointer-events-none'}`}
          >
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0, transition: { duration: 0.16, ease: [0.22, 1, 0.36, 1] } }}
              exit={{ opacity: 0, y: 6, transition: { duration: 0.09, ease: [0.4, 0, 1, 1] } }}
              className="model-service-page settings-modal-surface settings-solid settings-shell w-full max-w-[1320px] h-[92vh] min-h-0 bg-[#f5f3f0] p-4 sm:p-5 border border-white/80 relative z-10 flex flex-col"
            >
              {/* Header */}
              <div className="flex items-center mb-4 px-2">
                <div className="flex items-center gap-2">
                  <button onClick={onClose} className="w-9 h-9 flex items-center justify-center bg-white/80 hover:bg-white rounded-[14px] ring-1 ring-black/[0.05] transition-colors text-gray-600" title="返回设置中心" aria-label="返回设置中心">
                    <ChevronLeft size={18} />
                  </button>
                  <div className="p-2 rounded-[14px] bg-[#F3EDE8] text-[#5c564f]">
                    <Settings2 className="w-5 h-5" />
                  </div>
                  <div>
                    <h2 className="text-[18px] font-bold text-gray-900">模型服务管理</h2>
                  </div>
                </div>
              </div>

              {/* Main Content Area */}
              <div className="model-service-layout flex-1 min-h-0 pb-2">
                {/* Left Column: Sidebar */}
                <aside className="flex w-full min-w-0 flex-col gap-4 min-h-[220px] lg:min-h-0">
                  {/* Search Bar */}
                  <div className="relative">
                    <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
                    <input 
                      type="text" 
                      value={providerSearch}
                      onChange={(e) => setProviderSearch(e.target.value)}
                      placeholder="搜索模型平台..." 
                      className="w-full rounded-[16px] border border-[#e8e4df] bg-white pl-10 pr-4 py-3 text-[13px] text-gray-700 focus:outline-none focus:ring-2 focus:ring-[#FFA07A]/20 focus:border-[#E8C4B4]"
                    />
                  </div>
                  {providerDeleteError && (
                    <div role="alert" aria-live="polite" className="flex items-start gap-2 rounded-[12px] border border-red-200 bg-red-50 px-3 py-2.5 text-[11px] font-medium leading-relaxed text-red-700">
                      <AlertCircle size={14} className="mt-0.5 shrink-0" />
                      <span>{providerDeleteError}</span>
                    </div>
                  )}

                  {/* Provider List */}
                  <div className="flex-1 overflow-y-auto pr-1 space-y-2 custom-scrollbar pb-2">
                    {filteredProviders.map(p => (
                      <ProviderItem 
                        key={p.id}
                        provider={p}
                        isActive={p.id === activeProvider?.id}
                        onClick={() => setActiveProviderId(p.id)}
                        onDelete={!p.isSystem ? () => handleDeleteCustomProvider(p) : undefined}
                        isDeleting={deletingProviderId === p.id}
                      />
                    ))}

                    <div className="mt-4 border-t border-gray-200 pt-3">
                      <button
                        type="button"
                        onClick={() => {
                          setCustomProviderFormOpen(true)
                          setCustomProviderError('')
                        }}
                        aria-expanded={customProviderFormOpen}
                        className={`settings-card settings-card-interactive flex w-full items-center rounded-[22px] bg-white p-3 text-left ${customProviderFormOpen ? 'ring-2 ring-[#E8C4B4]/70' : ''}`}
                      >
                        <div className="accent-control w-10 h-10 rounded-[14px] flex items-center justify-center shrink-0 shadow-inner">
                          <Settings2 size={20} />
                        </div>
                        <div className="ml-3 flex-1">
                          <h3 className="text-[14px] font-bold text-gray-800">添加自定义服务</h3>
                          <p className="text-[11px] font-medium text-gray-500">服务、密钥与模型一次配置</p>
                        </div>
                        <ChevronRight className="w-4 h-4 text-gray-400" />
                      </button>
                    </div>
                  </div>
                </aside>

                {customProviderFormOpen ? (
                  <motion.section
                    key="custom-provider-editor"
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="settings-card col-span-1 flex min-h-0 min-w-0 flex-col overflow-hidden bg-white lg:col-span-2"
                  >
                    <div className="flex items-start justify-between gap-4 border-b border-gray-100 px-6 py-5">
                      <div className="flex min-w-0 items-start gap-3.5">
                        <div className="accent-control flex h-11 w-11 shrink-0 items-center justify-center rounded-[14px]">
                          <Settings2 size={20} />
                        </div>
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-2">
                            <h3 className="text-[17px] font-bold text-gray-900">{customProviderForm.protocol === 'anthropic' ? '添加 Anthropic 服务' : '添加 OpenAI 兼容服务'}</h3>
                            <span className="rounded-[8px] bg-[#FFF1EA] px-2 py-1 text-[10px] font-bold text-[#A8533D]">
                              {CUSTOM_PROVIDER_PROTOCOL_PRESETS[customProviderForm.protocol === 'anthropic' ? 'anthropic' : 'openai'].badge}
                            </span>
                          </div>
                          <p className="mt-1 text-[12px] font-medium text-gray-500">同时保存连接凭据和首个模型</p>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => {
                          setCustomProviderFormOpen(false)
                          resetCustomProviderDraft()
                        }}
                        className="rounded-[11px] px-3 py-2 text-[12px] font-semibold text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-800"
                      >
                        取消
                      </button>
                    </div>

                    <div className="custom-scrollbar min-h-0 flex-1 overflow-y-auto px-6 py-5">
                      <div className="grid grid-cols-1 gap-x-7 gap-y-6 xl:grid-cols-2">
                        <section className="min-w-0 space-y-4">
                          <div className="border-b border-gray-100 pb-3">
                            <p className="text-[13px] font-bold text-gray-800">连接信息</p>
                            <p className="mt-1 text-[11px] font-medium text-gray-400">密钥仅保存在当前设备</p>
                          </div>
                          <div className="space-y-2">
                            <p className="px-0.5 text-[11px] font-bold text-gray-600">接口协议</p>
                            <div role="tablist" aria-label="自定义服务接口协议" className="grid grid-cols-2 gap-1 rounded-[13px] bg-[#F7F4F1] p-1">
                              {Object.keys(CUSTOM_PROVIDER_PROTOCOL_PRESETS).map(protocol => {
                                const active = customProviderForm.protocol === protocol
                                return (
                                  <button
                                    key={protocol}
                                    type="button"
                                    role="tab"
                                    aria-selected={active}
                                    onClick={() => handleCustomProviderProtocolChange(protocol)}
                                    className={`rounded-[10px] px-3 py-2 text-[12px] font-bold transition-all ${active
                                      ? 'bg-white text-[#A8533D] shadow-[0_2px_7px_rgba(61,45,35,0.12)]'
                                      : 'text-gray-500 hover:bg-white/70 hover:text-gray-700'}`}
                                  >
                                    {protocol === 'anthropic' ? 'Anthropic' : 'OpenAI'}
                                  </button>
                                )
                              })}
                            </div>
                            <p className="px-0.5 text-[10px] font-medium leading-relaxed text-gray-400">
                              {customProviderForm.protocol === 'anthropic'
                                ? '使用原生 Messages API，认证头默认为 x-api-key。'
                                : '使用 OpenAI-compatible Chat Completions 请求格式。'}
                            </p>
                          </div>
                          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                            <LabeledField label="服务标识" hint="唯一 ID">
                              <GlassInput
                                aria-label="Provider ID"
                                placeholder="例如 kimi-k3"
                                value={customProviderForm.id}
                                onChange={e => updateCustomProviderForm('id', e.target.value)}
                              />
                            </LabeledField>
                            <LabeledField label="显示名称">
                              <GlassInput
                                aria-label="自定义 Provider 显示名称"
                                placeholder="例如 Kimi K3"
                                value={customProviderForm.name}
                                onChange={e => updateCustomProviderForm('name', e.target.value)}
                              />
                            </LabeledField>
                          </div>
                          <LabeledField label="API 地址" hint="包含版本前缀">
                            <GlassInput
                              icon={Link2}
                              aria-label="自定义 Provider API Host"
                              placeholder={customProviderForm.protocol === 'anthropic' ? 'https://api.anthropic.com/v1' : 'https://api.example.com/v1'}
                              value={customProviderForm.apiHost}
                              onChange={e => updateCustomProviderForm('apiHost', e.target.value)}
                            />
                          </LabeledField>
                          <LabeledField label="访问密钥" hint="仅本机保存">
                            <GlassInput
                              icon={Key}
                              aria-label="自定义 Provider API Key"
                              type={customProviderApiKeyVisible ? 'text' : 'password'}
                              autoComplete="off"
                              placeholder={customProviderForm.protocol === 'anthropic' ? 'sk-ant-...' : 'sk-...'}
                              value={customProviderForm.apiKey}
                              onChange={e => updateCustomProviderForm('apiKey', e.target.value)}
                              trailing={(
                                <button
                                  type="button"
                                  onClick={() => setCustomProviderApiKeyVisible(value => !value)}
                                  aria-label={customProviderApiKeyVisible ? '隐藏访问密钥' : '显示访问密钥'}
                                  className="flex h-8 w-8 items-center justify-center rounded-[9px] text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-700"
                                >
                                  {customProviderApiKeyVisible ? <EyeOff size={16} /> : <Eye size={16} />}
                                </button>
                              )}
                            />
                          </LabeledField>
                        </section>

                        <section className="min-w-0 space-y-4">
                          <div className="border-b border-gray-100 pb-3">
                            <p className="text-[13px] font-bold text-gray-800">首个模型</p>
                            <p className="mt-1 text-[11px] font-medium text-gray-400">保存后会独立显示在模型列表</p>
                          </div>
                          <div className="grid grid-cols-1 gap-3 sm:grid-cols-[150px_minmax(0,1fr)]">
                            <LabeledField label="模型类型">
                              <SoftSelect
                                ariaLabel="自定义模型类型"
                                value={customProviderForm.modelType}
                                onChange={handleCustomModelTypeChange}
                                disabled={customProviderForm.protocol === 'anthropic'}
                                options={customProviderForm.protocol === 'anthropic'
                                  ? [{ value: 'chat', label: 'Chat' }]
                                  : [
                                    { value: 'chat', label: 'Chat' },
                                    { value: 'embedding', label: 'Embedding' },
                                    { value: 'rerank', label: 'Rerank' },
                                  ]}
                              />
                            </LabeledField>
                            <LabeledField label="模型 ID">
                              <div className="flex min-w-0 gap-2">
                                <div className="min-w-0 flex-1">
                                  <GlassInput
                                    aria-label="自定义模型 ID"
                                    list="custom-provider-model-options"
                                    placeholder="例如 kimi-k3"
                                    value={customProviderForm.modelId}
                                    onChange={e => handleCustomModelIdChange(e.target.value)}
                                  />
                                  <datalist id="custom-provider-model-options">
                                    {customProviderModels.map(model => <option key={model.id} value={model.id} />)}
                                  </datalist>
                                </div>
                                <button
                                  type="button"
                                  onClick={handleFetchCustomProviderModels}
                                  disabled={customProviderFetchingModels}
                                  className="flex min-w-[92px] shrink-0 items-center justify-center gap-1.5 rounded-[13px] border border-gray-200 bg-gray-50 px-3 text-[12px] font-bold text-gray-700 transition-colors hover:border-[#F3B39D] hover:bg-[#FFF7F3] disabled:cursor-not-allowed disabled:opacity-60"
                                >
                                  <RefreshCw size={14} className={customProviderFetchingModels ? 'animate-spin' : ''} />
                                  {customProviderFetchingModels ? '获取中' : '获取模型'}
                                </button>
                              </div>
                            </LabeledField>
                          </div>
                          <div className={`grid grid-cols-1 gap-3 ${customProviderForm.modelType === 'embedding' ? 'sm:grid-cols-2' : ''}`}>
                            <LabeledField label="模型显示名称" hint="可选">
                              <GlassInput
                                aria-label="自定义模型显示名称"
                                placeholder="默认使用模型 ID"
                                value={customProviderForm.modelName}
                                onChange={e => updateCustomProviderForm('modelName', e.target.value)}
                              />
                            </LabeledField>
                            {customProviderForm.modelType === 'embedding' && (
                              <LabeledField label="向量维度">
                                <GlassInput
                                  aria-label="自定义模型向量维度"
                                  type="number"
                                  min="1"
                                  placeholder="例如 1024"
                                  value={customProviderForm.dimension}
                                  onChange={e => updateCustomProviderForm('dimension', e.target.value)}
                                />
                              </LabeledField>
                            )}
                          </div>
                          <div className="space-y-2">
                            <p className="px-0.5 text-[11px] font-bold text-gray-600">模型标签</p>
                            <div className="flex flex-wrap gap-2">
                              {TAG_OPTIONS.map(tag => (
                                <Tag
                                  key={tag.value}
                                  text={tag.label}
                                  active={customProviderModelTags.includes(tag.value)}
                                  onClick={() => {
                                    setCustomProviderModelTags(current => current.includes(tag.value)
                                      ? current.filter(value => value !== tag.value)
                                      : [...current, tag.value])
                                  }}
                                />
                              ))}
                            </div>
                          </div>
                          <label className="flex cursor-pointer items-center gap-2.5 rounded-[12px] bg-gray-50 px-3 py-2.5 text-[12px] font-semibold text-gray-700">
                            <input
                              type="checkbox"
                              checked={customProviderForm.setAsDefault}
                              onChange={e => updateCustomProviderForm('setAsDefault', e.target.checked)}
                              className="h-4 w-4 rounded border-gray-300 accent-[#F0653A]"
                            />
                            保存后设为该类型的默认模型
                          </label>
                          {customProviderFetchResult && (
                            <div className={`rounded-[11px] px-3 py-2 text-[11px] font-semibold ${customProviderFetchResult.success ? 'bg-green-50 text-green-700' : 'bg-amber-50 text-amber-700'}`}>
                              {customProviderFetchResult.message}
                            </div>
                          )}
                        </section>
                      </div>

                      <div className="mt-6 border-t border-gray-100 pt-4">
                        <button
                          type="button"
                          aria-expanded={customProviderAdvancedOpen}
                          onClick={() => setCustomProviderAdvancedOpen(value => !value)}
                          className="flex w-full items-center justify-between rounded-[12px] px-2 py-2 text-left transition-colors hover:bg-gray-50"
                        >
                          <div>
                            <p className="text-[13px] font-bold text-gray-800">高级配置</p>
                            <p className="mt-0.5 text-[11px] font-medium text-gray-400">大多数服务无需修改，仅在网关文档有明确要求时调整</p>
                          </div>
                          <ChevronDown size={17} className={`text-gray-400 transition-transform ${customProviderAdvancedOpen ? 'rotate-180' : ''}`} />
                        </button>
                        <AnimatePresence initial={false}>
                          {customProviderAdvancedOpen && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: 'auto', opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              className="overflow-hidden"
                            >
                              <div className="px-2 pb-2 pt-4">
                                <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.25fr)_minmax(250px,0.75fr)]">
                                  <section className="min-w-0">
                                    <div className="mb-3 flex items-start gap-2.5">
                                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[10px] bg-[#F7F4F1] text-gray-500">
                                        <Link2 size={15} />
                                      </div>
                                      <div className="min-w-0">
                                        <p className="text-[12px] font-bold text-gray-800">接口路径</p>
                                        <p className="mt-0.5 text-[10px] font-medium leading-relaxed text-gray-400">只显示当前已启用能力需要的接口</p>
                                      </div>
                                    </div>
                                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                                      <LabeledField label="对话接口" hint={customProviderForm.protocol === 'anthropic' ? 'Messages API' : 'Chat Completions'}>
                                        <GlassInput aria-label="Chat 路径" value={customProviderForm.chatEndpoint} onChange={e => updateCustomProviderForm('chatEndpoint', e.target.value)} />
                                      </LabeledField>
                                      <LabeledField label="模型列表" hint="用于自动获取模型">
                                        <GlassInput aria-label="模型列表路径" value={customProviderForm.fetchModelsEndpoint} onChange={e => updateCustomProviderForm('fetchModelsEndpoint', e.target.value)} />
                                      </LabeledField>
                                      {customProviderForm.protocol !== 'anthropic' && customProviderForm.embedding && (
                                        <LabeledField label="向量接口" hint="Embedding">
                                          <GlassInput aria-label="Embedding 路径" value={customProviderForm.embeddingEndpoint} onChange={e => updateCustomProviderForm('embeddingEndpoint', e.target.value)} />
                                        </LabeledField>
                                      )}
                                      {customProviderForm.protocol !== 'anthropic' && customProviderForm.rerank && (
                                        <LabeledField label="重排接口" hint="Rerank">
                                          <GlassInput aria-label="Rerank 路径" value={customProviderForm.rerankEndpoint} onChange={e => updateCustomProviderForm('rerankEndpoint', e.target.value)} />
                                        </LabeledField>
                                      )}
                                    </div>
                                  </section>

                                  <section className="min-w-0 border-t border-gray-100 pt-5 xl:border-l xl:border-t-0 xl:pl-6 xl:pt-0">
                                    <div className="mb-3 flex items-start gap-2.5">
                                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[10px] bg-[#FFF4EF] text-[#B85F47]">
                                        <Zap size={15} />
                                      </div>
                                      <div className="min-w-0">
                                        <p className="text-[12px] font-bold text-gray-800">服务能力</p>
                                        <p className="mt-0.5 text-[10px] font-medium leading-relaxed text-gray-400">按服务商文档开启，避免调用不存在的接口</p>
                                      </div>
                                    </div>
                                    <p className="mb-2 text-[10px] font-bold text-gray-400">模型能力</p>
                                    <div className="grid grid-cols-2 gap-2">
                                      <CapabilityToggle checked={customProviderForm.chat} disabled={customProviderForm.protocol === 'anthropic' || customProviderForm.modelType === 'chat'} label="Chat" onToggle={checked => updateCustomProviderForm('chat', checked)} />
                                      {customProviderForm.protocol !== 'anthropic' && <>
                                        <CapabilityToggle checked={customProviderForm.embedding} disabled={customProviderForm.modelType === 'embedding'} label="Embedding" onToggle={checked => updateCustomProviderForm('embedding', checked)} />
                                        <CapabilityToggle checked={customProviderForm.rerank} disabled={customProviderForm.modelType === 'rerank'} label="Rerank" onToggle={checked => updateCustomProviderForm('rerank', checked)} />
                                      </>}
                                    </div>
                                    <p className="mb-2 mt-4 text-[10px] font-bold text-gray-400">对话特性</p>
                                    <div className="grid grid-cols-2 gap-2">
                                      <CapabilityToggle checked={customProviderForm.supportsStreaming} disabled={!customProviderForm.chat} label="流式输出" ariaLabel="SSE 流式" onToggle={checked => updateCustomProviderForm('supportsStreaming', checked)} />
                                      <CapabilityToggle checked={customProviderForm.supportsReasoning} disabled={!customProviderForm.chat} label="思考参数" onToggle={checked => updateCustomProviderForm('supportsReasoning', checked)} />
                                    </div>
                                    {customProviderForm.supportsReasoning && (
                                      <details className="mt-3 border-t border-gray-100 pt-3">
                                        <summary className="cursor-pointer select-none text-[11px] font-bold text-gray-600 outline-none marker:text-gray-400">
                                          高级思考兼容
                                        </summary>
                                        <div className="mt-3 grid grid-cols-2 gap-3">
                                          <LabeledField label="原生协议">
                                            <GlassSelect aria-label="思考原生协议" value={customProviderForm.reasoningMode} onChange={e => updateCustomProviderForm('reasoningMode', e.target.value)}>
                                              <option value="">自动识别</option>
                                              <option value="openai_effort">OpenAI 档位</option>
                                              <option value="anthropic_adaptive">Anthropic 自适应</option>
                                              <option value="anthropic_budget">Anthropic 预算</option>
                                              <option value="gemini_level">Gemini 档位</option>
                                              <option value="gemini_budget">Gemini 预算</option>
                                              <option value="qwen_budget">Qwen 预算</option>
                                              <option value="thinking_toggle">思考开关</option>
                                              <option value="ollama_think">Ollama think</option>
                                              <option value="fixed">固定思考</option>
                                            </GlassSelect>
                                          </LabeledField>
                                          <LabeledField label="可用档位">
                                            <GlassSelect
                                              aria-label="思考可用档位"
                                              value={customProviderForm.reasoningOptions.join(',')}
                                              onChange={e => updateCustomProviderForm('reasoningOptions', e.target.value ? e.target.value.split(',') : [])}
                                            >
                                              <option value="">跟随协议</option>
                                               <option value="off,medium">off / medium</option>
                                               <option value="off,low,medium,high">off / low / medium / high</option>
                                               <option value="off,minimal,low,medium,high,xhigh,max">off / minimal / low / medium / high / xhigh / max</option>
                                               <option value="low,medium,high">low / medium / high</option>
                                               <option value="medium">medium</option>
                                            </GlassSelect>
                                          </LabeledField>
                                          <LabeledField label="默认档位">
                                            <GlassSelect aria-label="思考默认档位" value={customProviderForm.reasoningDefault} onChange={e => updateCustomProviderForm('reasoningDefault', e.target.value)}>
                                               <option value="">自动</option>
                                               <option value="off">off</option>
                                               <option value="minimal">minimal</option>
                                               <option value="low">low</option>
                                               <option value="medium">medium</option>
                                               <option value="high">high</option>
                                               <option value="xhigh">xhigh</option>
                                               <option value="max">max</option>
                                            </GlassSelect>
                                          </LabeledField>
                                          <LabeledField label="是否可关闭">
                                            <GlassSelect aria-label="思考是否始终开启" value={customProviderForm.reasoningAlwaysEnabled} onChange={e => updateCustomProviderForm('reasoningAlwaysEnabled', e.target.value)}>
                                              <option value="">自动</option>
                                              <option value="false">允许关闭</option>
                                              <option value="true">始终开启</option>
                                            </GlassSelect>
                                          </LabeledField>
                                          <LabeledField label="关闭参数">
                                            <GlassSelect aria-label="思考关闭参数" value={customProviderForm.reasoningOffControl} onChange={e => updateCustomProviderForm('reasoningOffControl', e.target.value)}>
                                              <option value="">跟随协议</option>
                                              <option value="reasoning_effort_none">reasoning_effort: none</option>
                                              <option value="thinking_disabled">thinking: disabled</option>
                                              <option value="enable_thinking_false">enable_thinking: false</option>
                                              <option value="gemini_budget_zero">thinkingBudget: 0</option>
                                              <option value="ollama_think_false">think: false</option>
                                            </GlassSelect>
                                          </LabeledField>
                                          <LabeledField label="开启参数">
                                            <GlassSelect aria-label="思考开启参数" value={customProviderForm.reasoningOnControl} onChange={e => updateCustomProviderForm('reasoningOnControl', e.target.value)}>
                                              <option value="">跟随协议</option>
                                              <option value="thinking_enabled">thinking: enabled</option>
                                              <option value="thinking_adaptive">thinking: adaptive</option>
                                              <option value="enable_thinking_true">enable_thinking: true</option>
                                              <option value="reasoning_split_true">reasoning_split: true（仅分离输出）</option>
                                              <option value="provider_default">不发送，使用上游默认</option>
                                            </GlassSelect>
                                          </LabeledField>
                                        </div>
                                      </details>
                                    )}
                                  </section>
                                </div>

                                <section className="mt-6 border-t border-gray-100 pt-5">
                                  <div className="mb-3 flex items-start gap-2.5">
                                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[10px] bg-[#F7F4F1] text-gray-500">
                                      <Key size={15} />
                                    </div>
                                    <div className="min-w-0">
                                      <div className="flex flex-wrap items-center gap-2">
                                        <p className="text-[12px] font-bold text-gray-800">认证方式</p>
                                        <span className="rounded-[7px] bg-gray-100 px-2 py-0.5 text-[9px] font-bold text-gray-500">通常无需修改</span>
                                      </div>
                                      <p className="mt-0.5 text-[10px] font-medium leading-relaxed text-gray-400">
                                        {customProviderForm.protocol === 'anthropic' ? 'Anthropic 默认使用 x-api-key，密钥前不加前缀' : 'OpenAI 兼容服务通常使用 Authorization: Bearer <key>'}
                                      </p>
                                    </div>
                                  </div>
                                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                                    <LabeledField label="密钥请求头">
                                      <GlassInput aria-label="API Key 请求头" value={customProviderForm.apiKeyHeader} onChange={e => updateCustomProviderForm('apiKeyHeader', e.target.value)} />
                                    </LabeledField>
                                    <LabeledField label="密钥前缀" hint={customProviderForm.protocol === 'anthropic' ? '留空' : '末尾保留空格'}>
                                      <GlassInput aria-label="API Key 前缀" placeholder={customProviderForm.protocol === 'anthropic' ? '无需填写' : 'Bearer '} value={customProviderForm.apiKeyPrefix} onChange={e => updateCustomProviderForm('apiKeyPrefix', e.target.value)} />
                                    </LabeledField>
                                  </div>
                                </section>
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>

                      {customProviderError && (
                        <div role="alert" aria-live="polite" className="mt-4 flex items-start gap-2 rounded-[12px] border border-red-200 bg-red-50 px-3 py-2.5 text-[12px] font-medium leading-relaxed text-red-700">
                          <AlertCircle size={15} className="mt-0.5 shrink-0" />
                          <span>{customProviderError}</span>
                        </div>
                      )}
                      {customProviderTestResult && (
                        <div role="status" aria-live="polite" className={`mt-4 flex items-start gap-2 rounded-[12px] border px-3 py-2.5 text-[12px] font-medium ${customProviderTestResult.success ? 'border-green-200 bg-green-50 text-green-700' : 'border-red-200 bg-red-50 text-red-700'}`}>
                          {customProviderTestResult.success ? <CheckCircle2 size={15} className="mt-0.5 shrink-0" /> : <AlertCircle size={15} className="mt-0.5 shrink-0" />}
                          <span>{customProviderTestResult.message}{Number.isFinite(customProviderTestResult.latency) ? ` · ${customProviderTestResult.latency}ms` : ''}</span>
                        </div>
                      )}
                    </div>

                    <div className="flex flex-wrap items-center justify-end gap-2 border-t border-gray-100 bg-gray-50/80 px-6 py-4">
                      <button
                        type="button"
                        onClick={handleTestCustomProvider}
                        disabled={customProviderTesting || customProviderSaving}
                        className="flex min-w-[124px] items-center justify-center gap-2 rounded-[13px] border border-gray-200 bg-white px-4 py-2.5 text-[13px] font-bold text-gray-700 transition-colors hover:border-[#F3B39D] hover:bg-[#FFF7F3] disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {customProviderTesting ? <RefreshCw size={15} className="animate-spin" /> : <Play size={15} />}
                        {customProviderTesting ? '测试中...' : '测试配置'}
                      </button>
                      <button
                        type="button"
                        onClick={handleAddCustomProvider}
                        disabled={customProviderSaving || customProviderTesting}
                        className="accent-cta flex min-w-[148px] items-center justify-center gap-2 rounded-[13px] px-5 py-2.5 text-[13px] font-bold disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {customProviderSaving ? <RefreshCw size={15} className="animate-spin" /> : <Check size={15} />}
                        {customProviderSaving ? '保存中...' : '保存并使用'}
                      </button>
                    </div>
                  </motion.section>
                ) : (
                  <>
                {/* Middle Column: Configuration */}
                <section className="flex min-w-0 w-full flex-col gap-4 overflow-visible pb-2 custom-scrollbar lg:overflow-y-auto lg:pr-2">
                  {/* Header Card */}
                  <div className="settings-card bg-white px-6 py-5 flex justify-between items-center shrink-0">
                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 bg-[#faf9f7] rounded-[18px] flex items-center justify-center ring-1 ring-black/[0.04] overflow-hidden">
                        <ProviderAvatar providerId={activeProvider?.id} size={32} />
                      </div>
                      <div>
                        <h2 className="text-[18px] font-bold text-gray-900">{activeProvider?.name || '未选择'}</h2>
                        <p className="text-[14px] text-gray-500 font-medium">{activeProvider?.id}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <label className="flex items-center gap-2.5 cursor-pointer select-none">
                        <span className={`text-[13px] font-bold ${activeProvider?.enabled ? 'text-gray-800' : 'text-gray-400'}`}>
                          {activeProvider?.enabled ? '已启用' : '已停用'}
                        </span>
                        <input
                          type="checkbox"
                          checked={!!activeProvider?.enabled}
                          onChange={e => handleProviderUpdate('enabled', e.target.checked)}
                          className="sr-only"
                        />
                        <span
                          aria-hidden="true"
                          className={`relative h-[22px] w-[38px] shrink-0 rounded-full transition-colors ${activeProvider?.enabled ? 'bg-[#F0653A]' : 'bg-gray-300'}`}
                        >
                          <span className={`absolute top-[3px] h-4 w-4 rounded-full bg-white shadow-sm transition-[left] duration-200 ${activeProvider?.enabled ? 'left-[18px]' : 'left-[3px]'}`} />
                        </span>
                      </label>
                    </div>
                  </div>

                  {/* API Settings Card */}
                  <div className="settings-card bg-white p-6 space-y-4 shrink-0">
                    <div className="space-y-1.5">
                      <label className="text-[12px] font-semibold text-gray-600 ml-1">API Key</label>
                      <GlassInput 
                        icon={Key} 
                        type={providerApiKeyVisible ? 'text' : 'password'}
                        placeholder="sk-... (多个 Key 用逗号分隔)" 
                        value={activeProvider?.apiKey || ''}
                        onChange={e => handleProviderUpdate('apiKey', e.target.value)}
                        trailing={(
                          <button
                            type="button"
                            onClick={() => setProviderApiKeyVisible(value => !value)}
                            aria-label={providerApiKeyVisible ? '隐藏 API Key' : '显示 API Key'}
                            className="flex h-8 w-8 items-center justify-center rounded-[9px] text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-700"
                          >
                            {providerApiKeyVisible ? <EyeOff size={16} /> : <Eye size={16} />}
                          </button>
                        )}
                      />
                    </div>
                    <div className="space-y-1.5 mt-2">
                      <label className="text-[12px] font-semibold text-gray-600 ml-1">API 地址</label>
                      <GlassInput 
                        icon={Link2} 
                        placeholder="https://api.openai.com/v1" 
                        value={activeProvider?.apiHost || ''}
                        onChange={e => handleProviderUpdate('apiHost', e.target.value)}
                      />
                    </div>
                    <div className="grid grid-cols-1 gap-3 pt-1 md:grid-cols-2">
                      <div className="space-y-1.5">
                        <label className="text-[12px] font-semibold text-gray-600 ml-1">API Key 请求头</label>
                        <GlassInput
                          placeholder="Authorization"
                          value={activeProvider?.apiConfig?.apiKeyHeader || ''}
                          onChange={e => handleProviderApiConfigUpdate('apiKeyHeader', e.target.value)}
                        />
                      </div>
                      <div className="space-y-1.5">
                        <span className="flex items-baseline justify-between gap-2 ml-1">
                          <label className="text-[12px] font-semibold text-gray-600">API Key 前缀</label>
                          <span className="text-[10px] font-medium text-gray-400">末尾需留空格</span>
                        </span>
                        <GlassInput
                          placeholder="Bearer "
                          value={activeProvider?.apiConfig?.apiKeyPrefix ?? ''}
                          onChange={e => handleProviderApiConfigUpdate('apiKeyPrefix', e.target.value)}
                        />
                      </div>
                    </div>
                    <div className="flex gap-4 pt-3">
                      <button
                        onClick={handleTest}
                        disabled={!activeProvider || testing}
                        className="accent-cta flex-1 disabled:opacity-60 disabled:cursor-not-allowed text-[14px] font-semibold py-3 rounded-[16px] flex items-center justify-center gap-2"
                      >
                        {testing ? <RefreshCw size={16} className="animate-spin" /> : <Play size={16} className="fill-current" />}
                        <span>测试连接</span>
                      </button>
                      <button 
                        onClick={handleSyncModels}
                        disabled={!activeProvider || isFetching}
                        className="flex-1 bg-[#faf9f7] hover:bg-[#f3efe9] active:bg-[#ece8e3] disabled:opacity-60 disabled:cursor-not-allowed transition-colors text-gray-700 text-[14px] font-semibold py-3 rounded-[16px] ring-1 ring-black/[0.06] flex items-center justify-center gap-2"
                      >
                        <RefreshCw size={16} className={isFetching ? "animate-spin" : ""} />
                        {isFetching ? '同步中...' : '同步模型'}
                      </button>
                    </div>

                    {testResult && (
                      <div className={`mt-2 rounded-[16px] p-3 text-[13px] border font-medium ${testResult.success ? 'border-green-200 bg-green-50/80 text-green-700' : 'border-red-200 bg-red-50/80 text-red-700'}`}>
                        {testResult.success
                          ? `${testResult.message || '连接成功'}${testResult.verifiedModel ? ` · ${testResult.verifiedModel}` : ''}${testResult.latency ? ` (${testResult.latency}ms)` : ''}`
                          : '连接失败'
                        } {testResult.message || testResult.error || ''}
                      </div>
                    )}
                    {fetchError && (
                      <div className="mt-2 rounded-[16px] p-3 text-[13px] border border-amber-200 bg-amber-50/80 text-amber-700 font-medium">
                        {fetchError}
                      </div>
                    )}
                  </div>

                  {/* Add Model Card */}
                  <div className="settings-card bg-white p-6 flex flex-col flex-1 shrink-0">
                    <h3 className="text-[15px] font-bold text-gray-900 mb-4 shrink-0">手动新增模型</h3>
                    <div className="space-y-4 pr-2">
                      <div className="flex gap-4 shrink-0">
                        <div className="flex-[2] space-y-1.5">
                          {/* 「必填」跟在标签后面而不是靠 justify-between 推到列尾：
                              推到列尾会紧贴右侧「类型」的标签，看起来像在修饰「类型」。 */}
                          <label className="ml-1 block text-[12px] font-semibold text-gray-600">
                            模型 ID
                            <span className="ml-1.5 text-[10px] font-medium text-[#C2705A]">必填</span>
                          </label>
                          <GlassInput 
                            placeholder="模型 ID (如 gpt-4)" 
                            value={addModelForm.id}
                            onChange={e => setAddModelForm({ ...addModelForm, id: e.target.value })}
                          />
                        </div>
                        <div className="flex-1 space-y-1.5">
                          <label className="block text-[12px] font-semibold text-gray-600 ml-1">类型</label>
                          <SoftSelect
                            ariaLabel="新增模型类型"
                            value={addModelForm.type}
                            onChange={next => setAddModelForm({ ...addModelForm, type: next })}
                            options={[
                              { value: 'chat', label: 'Chat' },
                              { value: 'embedding', label: 'Embedding' },
                              { value: 'rerank', label: 'Rerank' },
                              { value: 'image', label: 'Image' },
                            ]}
                          />
                        </div>
                      </div>
                      <div className="shrink-0 space-y-1.5">
                        <label className="ml-1 block text-[12px] font-semibold text-gray-600">
                          显示名称
                          <span className="ml-1.5 text-[10px] font-medium text-gray-400">可选，留空则用模型 ID</span>
                        </label>
                        <GlassInput
                          placeholder="例如 GPT-4 Turbo"
                          value={addModelForm.name}
                          onChange={e => setAddModelForm({ ...addModelForm, name: e.target.value })}
                        />
                      </div>
                      {addModelForm.type === 'embedding' && (
                        <div className="shrink-0">
                          <GlassInput
                            type="number"
                            min="1"
                            placeholder="向量维度（必填，如 1024）"
                            value={addModelForm.dimension}
                            onChange={e => setAddModelForm({ ...addModelForm, dimension: e.target.value })}
                          />
                        </div>
                      )}
                      
                      <div className="space-y-2 pt-1 shrink-0 pb-2">
                        <label className="text-[12px] font-semibold text-gray-600 ml-1">标签 (可选)</label>
                        <div className="flex flex-wrap gap-2.5">
                          {TAG_OPTIONS.map(tag => (
                            <Tag 
                              key={tag.value} 
                              text={tag.label} 
                              active={newModelTags.includes(tag.value)}
                              onClick={() => {
                                if (newModelTags.includes(tag.value)) {
                                  setNewModelTags(newModelTags.filter(t => t !== tag.value))
                                } else {
                                  setNewModelTags([...newModelTags, tag.value])
                                }
                              }}
                            />
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="pt-4 shrink-0">
                      <button
                        onClick={handleAddModel}
                        className="w-full flex items-center justify-center gap-2 rounded-[16px] bg-[#F3EDE8] py-3 text-[14px] font-semibold text-[#5c564f] transition-colors hover:bg-[#ece6df] active:bg-[#e4ddd5]"
                      >
                        <Plus size={18} />
                        <span>保存模型</span>
                      </button>
                      {addSuccess && (
                        <div className="mt-2 text-center text-[12px] font-medium text-[#4F7F63]">
                          {addSuccess}
                        </div>
                      )}
                    </div>
                  </div>
                </section>

                {/* Right Column: Model List */}
                <aside className="settings-card flex min-h-[360px] w-full min-w-0 flex-col bg-white p-6">
                  <h3 className="text-[16px] font-bold text-gray-900 mt-2">模型列表</h3>
                  <p className="text-[12px] text-gray-500 font-medium mt-1 mb-4">
                    {isCustomProvider ? '每个模型独立管理，可分别测试和设置默认模型' : '按类型分组: 对话 / 嵌入 / 重排'}
                  </p>

                  <div className="flex-1 overflow-y-auto custom-scrollbar pr-1 space-y-4">
                    {isCustomProvider ? (
                      providerModels.length > 0 ? (
                        <div className="space-y-2.5">
                          {providerModels.map(renderCustomModelCard)}
                        </div>
                      ) : (
                        <div className="flex flex-col items-center justify-center py-12 text-center">
                          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gray-100/80 text-gray-400">
                            <Cpu size={20} />
                          </div>
                          <p className="mt-3 text-[13px] font-semibold text-gray-500">还没有独立模型</p>
                          <p className="mt-1 text-[11px] leading-relaxed text-gray-400">
                            填好 API Key 后点「同步模型」自动添加，
                            <br />
                            或在「手动新增模型」中保存一个模型
                          </p>
                        </div>
                      )
                    ) : (
                      <>
                    {['chat', 'embedding', 'rerank', 'image'].every(t => !(modelsByType[t] || []).length) && (
                      <div className="flex flex-col items-center justify-center py-12 text-center">
                        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gray-100/80 text-gray-400">
                          <Cpu size={20} />
                        </div>
                        <p className="mt-3 text-[13px] font-semibold text-gray-500">这个平台还没有模型</p>
                        <p className="mt-1 text-[11px] leading-relaxed text-gray-400">
                          填好 API Key 后点「同步模型」自动拉取，
                          <br />
                          或在「手动新增模型」中添加
                        </p>
                      </div>
                    )}
                    {['chat', 'embedding', 'rerank', 'image'].map(type => {
                      const list = modelsByType[type] || [];
                      if (list.length === 0) return null;
                      const meta = TYPE_META[type] || { label: type };
                      const isCollapsed = !!collapsedTypes[type];
                      const defaultLabel = (() => {
                        const key = DEFAULT_TYPE_MAP[type];
                        if (!key) return '—';
                        const current = getDefaultModel(key);
                        return current || '未选择';
                      })();
                      
                      return (
                        <div key={type} className="settings-inset rounded-[18px] overflow-hidden">
                          <button
                            type="button"
                            aria-expanded={!isCollapsed}
                            onClick={() => toggleCollapse(type)}
                            className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors focus:outline-none"
                          >
                            <div className="flex items-center gap-2 shrink-0">
                              {type === 'chat' && <MessageSquare size={16} className="text-[#B85F47]" />}
                              {type === 'embedding' && <Cpu size={16} className="text-[#B85F47]" />}
                              {type === 'rerank' && <Zap size={16} className="text-[#B85F47]" />}
                              {type === 'image' && <Sparkles size={16} className="text-[#B85F47]" />}
                              <span className="text-[14px] font-bold text-gray-800 whitespace-nowrap">{meta.label}</span>
                              <span className="text-[11px] font-medium text-[#B85F47] bg-[#FFA07A]/10 px-1.5 py-0.5 rounded-full">{list.length}</span>
                            </div>
                            {/* 默认模型已由下方模型行的「默认」徽章标示，组头再放一次会被挤成「默认 ...」。 */}
                            <div className="flex items-center gap-2 shrink-0 ml-2" title={`默认: ${defaultLabel}`}>
                              <ChevronDown size={14} className={`text-gray-400 shrink-0 transition-transform ${isCollapsed ? '-rotate-90' : ''}`} />
                            </div>
                          </button>
                          
                          <AnimatePresence initial={false}>
                            {!isCollapsed && (
                              <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ type: 'spring', stiffness: 320, damping: 24 }}
                                className="overflow-hidden"
                              >
                                <div className="p-2 pt-0 space-y-1">
                                  {list.map(model => (
                                      <div key={model.id} className={`bg-white/80 hover:bg-white rounded-[16px] p-2.5 flex items-center justify-between ${lastAddedModelKey === `${model.providerId}:${model.id}` ? 'ring-1 ring-green-300' : ''} cursor-pointer transition-colors group`}>
                                        <div className="flex items-center space-x-2.5 overflow-hidden min-w-0 flex-1">
                                          <div className="w-8 h-8 rounded-[12px] bg-[#faf9f7] flex items-center justify-center shrink-0 ring-1 ring-black/[0.04]">
                                            <ProviderAvatar providerId={getIconProviderId(model)} size={20} />
                                          </div>
                                          <div className="flex flex-col flex-1 min-w-0 py-0.5">
                                            <div className="flex flex-wrap items-center gap-1.5 mb-1">
                                              {/* 不用 break-words：长模型名会在词内断开，把 "v2" 这类尾巴孤立到下一行。 */}
                                              <h4 className="text-[13px] font-bold text-gray-800 text-pretty line-clamp-2 leading-snug" title={model.name || model.id}>{model.name || model.id}</h4>
                                              {model.tags?.map(tag => (
                                                <span key={tag} className="shrink-0 text-[9px] px-1 py-0.5 rounded bg-gray-100 text-gray-500 border border-gray-200/70">
                                                  {TAG_LABELS[tag] || tag}
                                                </span>
                                              ))}
                                              {model.metadata?.dimension && (
                                                <span className="shrink-0 text-[9px] px-1 py-0.5 rounded bg-amber-50 text-amber-600 border border-amber-100">
                                                  {model.metadata.dimension}维
                                                </span>
                                              )}
                                            </div>
                                            <div className="flex items-start gap-1.5 mt-0.5">
                                               {isDefaultModel(model.type, model.id) && <span className="text-[10px] bg-[#FFA07A]/10 text-[#B85F47] px-1.5 py-0.5 rounded-sm font-bold shrink-0 mt-px">默认</span>}
                                               {/* ID 是次要信息：单行省略号比 break-all 的逐字符断行可读得多，完整值在 title 里。 */}
                                               <p className="min-w-0 flex-1 truncate text-[11px] font-medium text-gray-400" title={model.id}>{model.id}</p>
                                            </div>
                                          </div>
                                        </div>
                                        <div className="flex shrink-0 items-center gap-1.5 ml-2">
                                            {!isDefaultModel(model.type, model.id) && (
                                              <button
                                                type="button"
                                                onClick={(e) => { e.stopPropagation(); handleSetDefault(model.type, model.id); }}
                                                className="inline-flex items-center gap-1 rounded-full bg-[#F3EDE8] px-2.5 py-1 text-[11px] font-medium text-[#5c564f] transition-colors hover:bg-[#ece6df] hover:text-[#3f3a35]"
                                                title="设为默认模型"
                                                aria-label={`将 ${model.name || model.id} 设为默认模型`}
                                              >
                                                <CheckCircle2 size={13} strokeWidth={2.1} />
                                                设为默认
                                              </button>
                                            )}
                                            {model.isUserAdded && (
                                              <button onClick={(e) => { e.stopPropagation(); removeModelFromCollection(model.id, model.providerId); }} className="p-1.5 text-gray-400 hover:text-red-500 rounded-md hover:bg-red-50 opacity-0 transition-opacity group-hover:opacity-100" title="删除">
                                                <Trash2 size={16} />
                                              </button>
                                            )}
                                        </div>
                                      </div>
                                  ))}
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      );
                    })}
                    
                    {(!modelsByType || Object.keys(modelsByType).length === 0) && (
                      <div className="text-center text-sm text-gray-500 py-8 bg-gray-50 rounded-[16px] border border-dashed border-gray-300">
                        暂无模型
                      </div>
                    )}
                      </>
                    )}
                  </div>
                </aside>
                  </>
                )}

              </div>
              <style dangerouslySetInnerHTML={{__html: `
                @keyframes shimmer {
                  100% { left: 125%; }
                }
                .animate-shimmer {
                  animation: shimmer 1.5s infinite;
                }
                /* Hide scrollbar for clean UI but allow scrolling */
                .custom-scrollbar::-webkit-scrollbar {
                  width: 4px;
                }
                .custom-scrollbar::-webkit-scrollbar-track {
                  background: transparent;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb {
                  background: rgba(184, 95, 71, 0.12);
                  border-radius: 10px;
                }
                .custom-scrollbar:hover::-webkit-scrollbar-thumb {
                  background: rgba(184, 95, 71, 0.24);
                }
                .animate-blob {
                  animation: blob 7s infinite;
                }
                .animation-delay-2000 {
                  animation-delay: 2s;
                }
                @keyframes blob {
                  0% { transform: translate(0px, 0px) scale(1); }
                  33% { transform: translate(30px, -50px) scale(1.1); }
                  66% { transform: translate(-20px, 20px) scale(0.9); }
                  100% { transform: translate(0px, 0px) scale(1); }
                }
              `}} />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    )
  } catch (err) {
    console.error('EmbeddingSettings render error', err)
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/25 p-4 font-sans">
        <div className="settings-solid settings-shell bg-white border border-white/80 p-6 max-w-lg w-full">
          <div className="text-lg font-bold text-gray-900 mb-2">模型服务管理加载失败</div>
          <div className="text-sm text-gray-600 mb-4 bg-red-50 p-3 rounded-xl border border-red-100">{err?.message || '未知错误'}</div>
          <button
            onClick={onClose}
            className="accent-surface w-full text-[14px] font-bold py-3 rounded-[16px] transition-colors"
          >
            关闭
          </button>
        </div>
      </div>
    )
  }
}
