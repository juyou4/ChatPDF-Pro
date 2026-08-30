import React, { useEffect, useState, useCallback } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Download,
  Eye,
  EyeOff,
  FileSearch,
  FolderOpen,
  Globe,
  Key,
  Loader2,
  RefreshCw,
  Save,
  ScanText,
  SlidersHorizontal,
  Wifi,
  ChevronLeft,
  XCircle,
} from 'lucide-react'
import SettingsSegmentedControl from './SettingsSegmentedControl'
import {
  DEFAULT_PARSE_ROUTE,
  PARSE_ROUTE_OPTIONS as SHARED_PARSE_ROUTE_OPTIONS,
  VALID_PARSE_ROUTES,
} from '../utils/parseRouteUtils'

/**
 * localStorage 中 OCR 设置的键名
 */
const OCR_SETTINGS_KEY = 'ocrSettings'

/**
 * OCR 模式选项定义
 * 每个选项包含值、标签和描述
 */
const OCR_MODES = [
  {
    value: 'auto',
    label: '自动',
    description: '只给抽不出字的页面补识别',
  },
  {
    value: 'always',
    label: '始终',
    description: '每一页都重新识别',
  },
  {
    value: 'never',
    label: '关闭',
    description: '只用 PDF 里已有的文字层',
  },
]

const FIELD_INPUT_CLASS =
  'w-full pl-10 py-2.5 text-sm rounded-xl border border-gray-200 bg-white focus:border-[#ed8c68]/50 focus:ring-2 focus:ring-[#ed8c68]/20 outline-none transition-all placeholder:text-gray-300'

/**
 * 文档上传时的主解析路线。OCR、YOLO 等能力只作为路线内的增强，
 * 不应再被误解为另一套正文来源。
 */
const PARSE_ROUTE_ICONS = {
  auto: FileSearch,
  local: ScanText,
  mineru: Globe,
}

const PARSE_ROUTE_OPTIONS = SHARED_PARSE_ROUTE_OPTIONS.map((option) => ({
  ...option,
  icon: PARSE_ROUTE_ICONS[option.value],
}))


/**
 * 后端名称到中文显示名称的映射
 */
const BACKEND_LABELS = {
  tesseract: 'Tesseract',
  paddleocr: 'PaddleOCR',
}

/**
 * OCR 引擎选择选项定义
 * 每个选项包含值、标签和描述
 */
const BACKEND_OPTIONS = [
  {
    value: 'auto',
    label: '自动选择',
    description: '自动选可用引擎',
  },
  {
    value: 'tesseract',
    label: 'Tesseract',
    description: '本地 OCR 引擎，需安装 Tesseract',
  },
  {
    value: 'paddleocr',
    label: 'PaddleOCR',
    description: '本地 OCR 引擎，需安装 PaddleOCR',
  },
]

/**
 * API 基础地址（Vite 代理转发到后端）
 */
const API_BASE_URL = ''

/**
 * 合法的 OCR 模式值
 */
const VALID_MODES = ['auto', 'always', 'never']

/**
 * 合法的 OCR 引擎后端值
 */
const VALID_BACKENDS = ['auto', 'tesseract', 'paddleocr']

/**
 * 本地逐页 OCR 不可用时才展示安装指引。
 * MinerU 可用不能掩盖 Tesseract/PaddleOCR 缺失。
 */
export function shouldShowLocalOcrInstallGuide(ocrStatus) {
  if (!ocrStatus) return false
  if (typeof ocrStatus.local_available === 'boolean') {
    return ocrStatus.local_available === false
  }
  const backends = ocrStatus.backends || {}
  return !['tesseract', 'paddleocr'].some((name) => Boolean(backends[name]))
}

/** 底部状态栏文案：按当前路线显示，不把 OCR 说成第三条解析方式。 */
export function parseSettingsStatusText({
  parseRoute,
  mineruConfigured,
  mineruConnectionVerified = false,
  ocrModeLabel,
  ocrBackendLabel,
}) {
  if (parseRoute === 'local') {
    return `本地解析 · ${ocrModeLabel || '自动'} · ${ocrBackendLabel || '自动选择'}`
  }
  // 配置已保存不代表远端 Token 已经验证通过。此前这里写成“已连接”，
  // 会让填写错误 Token 的用户误以为可以直接上传。
  if (!mineruConfigured) return 'MinerU · 待配置'
  return mineruConnectionVerified ? 'MinerU · 已验证' : 'MinerU · 已配置，待验证'
}

function OptionToggle({ title, description, checked, onToggle }) {
  return (
    <div className="flex items-center justify-between gap-3 px-3 py-2.5 rounded-xl bg-gray-50/80 border border-gray-100">
      <span className="min-w-0">
        <span className="block text-xs text-gray-700">{title}</span>
        {description ? (
          <span className="block text-[10px] leading-4 text-gray-400 mt-0.5">{description}</span>
        ) : null}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={title}
        onClick={onToggle}
        className={`relative w-[42px] h-[24px] rounded-full transition-colors duration-200 flex-shrink-0 ${
          checked ? 'accent-control' : 'bg-gray-300'
        }`}
      >
        <span
          className={`absolute top-[2px] left-[2px] w-[20px] h-[20px] rounded-full bg-white shadow-sm transition-transform ${
            checked ? 'translate-x-[18px]' : 'translate-x-0'
          }`}
        />
      </button>
    </div>
  )
}

/**
 * 合法的主解析路线值
 */
const LEGACY_PAGE_OCR_BACKENDS = ['mineru', 'mistral', 'doc2x']

const DEFAULT_OCR_SETTINGS = {
  mode: 'auto',
  backend: 'auto',
  parseRoute: DEFAULT_PARSE_ROUTE,
  mineruFigureEnhance: true,
  figureRenderMode: 'raw',
}

/** 旧版速览图表模式，仅用于兼容读取已有 localStorage。 */
const VALID_FIGURE_RENDER_MODES = ['raw', 'yolo']

/**
 * 从 localStorage 读取 OCR 设置
 * @returns {object} OCR 与主解析路线设置
 */
export function loadOCRSettings() {
  try {
    const raw = localStorage.getItem(OCR_SETTINGS_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      const result = { ...DEFAULT_OCR_SETTINGS }
      // 校验 mode 值是否合法
      if (VALID_MODES.includes(parsed.mode)) {
        result.mode = parsed.mode
      }
      // 校验 backend 值是否合法
      if (VALID_BACKENDS.includes(parsed.backend)) {
        result.backend = parsed.backend
      }
      // 已下线的云 OCR 与旧版 MinerU 逐页 OCR 统一迁回自动本地 OCR。
      const migratedLegacyPageOcr = LEGACY_PAGE_OCR_BACKENDS.includes(parsed.backend)
      const migratedLegacyParseRoute = parsed.parseRoute === 'auto'
      // 校验上传主解析路线
      if (VALID_PARSE_ROUTES.includes(parsed.parseRoute)) {
        result.parseRoute = parsed.parseRoute === 'local' ? 'local' : DEFAULT_PARSE_ROUTE
      }
      // 旧版图表字段继续透传，避免保存其他设置时破坏历史数据。
      if (typeof parsed.mineruFigureEnhance === 'boolean') {
        result.mineruFigureEnhance = parsed.mineruFigureEnhance
      }
      if (VALID_FIGURE_RENDER_MODES.includes(parsed.figureRenderMode)) {
        result.figureRenderMode = parsed.figureRenderMode
      }
      if (migratedLegacyPageOcr || migratedLegacyParseRoute) {
        localStorage.setItem(
          OCR_SETTINGS_KEY,
          JSON.stringify({ ...parsed, backend: 'auto', parseRoute: result.parseRoute })
        )
      }
      return result
    }
  } catch (err) {
    console.error('读取 OCR 设置失败:', err)
  }
  return { ...DEFAULT_OCR_SETTINGS }
}

/**
 * 将 OCR 设置保存到 localStorage
 * @param {object} settings - OCR 设置对象，包含 mode 和 backend
 */
export function saveOCRSettings(settings) {
  try {
    const nextSettings = { ...settings }
    if (LEGACY_PAGE_OCR_BACKENDS.includes(nextSettings.backend)) {
      nextSettings.backend = 'auto'
    }
    if (!VALID_PARSE_ROUTES.includes(nextSettings.parseRoute) || nextSettings.parseRoute === 'auto') {
      nextSettings.parseRoute = DEFAULT_PARSE_ROUTE
    }
    localStorage.setItem(OCR_SETTINGS_KEY, JSON.stringify(nextSettings))
  } catch (err) {
    console.error('保存 OCR 设置失败:', err)
  }
}

const formatApiDetail = (data, fallback) => {
  const detail = data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail
  if (Array.isArray(detail)) {
    const joined = detail
      .map((item) => item?.msg || item?.message || (typeof item === 'string' ? item : ''))
      .filter(Boolean)
      .join('；')
    if (joined) return joined
  }
  return data?.message || fallback
}

const getApiErrorMessage = async (res, fallback) => {
  try {
    const data = await res.json()
    return formatApiDetail(data, fallback)
  } catch {
    return fallback
  }
}

/**
 * OCR 设置面板组件
 * 保持与 EmbeddingSettings.jsx 一致的毛玻璃卡片 UI 风格
 *
 * @param {object} props
 * @param {boolean} props.isOpen - 面板是否打开
 * @param {function} props.onClose - 关闭面板的回调
 */
export default function OCRSettingsPanel({ isOpen, onClose }) {
  // 首屏按路线展示对应配置；图表兜底默认收起。
  const [moreOpen, setMoreOpen] = useState(false)
  // OCR 模式状态
  const [mode, setMode] = useState('auto')
  // OCR 引擎后端选择状态
  const [backend, setBackend] = useState('auto')
  // 上传时固定的主解析路线
  const [parseRoute, setParseRoute] = useState(DEFAULT_PARSE_ROUTE)
  // 后端 OCR 状态数据
  const [ocrStatus, setOcrStatus] = useState(null)
  // 加载状态
  const [loading, setLoading] = useState(false)
  // 错误信息
  const [error, setError] = useState(null)

  // 已加载的 MinerU 配置
  const [onlineConfig, setOnlineConfig] = useState(null)

  // ---- MinerU OCR 配置状态 ----
  // MinerU 接入模式：worker（代理）或 direct（官方 API 直连）
  const [mineruAccessMode, setMineruAccessMode] = useState('worker')
  // MinerU 官方 API Base URL
  const [mineruBaseUrl, setMineruBaseUrl] = useState('https://mineru.net/api/v4')
  // MinerU Worker URL
  const [mineruWorkerUrl, setMineruWorkerUrl] = useState('')
  // MinerU Auth Key
  const [mineruAuthKey, setMineruAuthKey] = useState('')
  // MinerU Token
  const [mineruToken, setMineruToken] = useState('')
  // MinerU Token 模式：'frontend'（前端透传）或 'worker'（Worker 配置）
  const [mineruTokenMode, setMineruTokenMode] = useState('frontend')
  // MinerU OCR 选项
  const [mineruEnableOcr, setMineruEnableOcr] = useState(false)
  const [mineruEnableFormula, setMineruEnableFormula] = useState(true)
  const [mineruEnableTable, setMineruEnableTable] = useState(true)
  const [mineruModelVersion, setMineruModelVersion] = useState('vlm')
  // 是否显示 MinerU Auth Key 明文
  const [showMineruAuthKey, setShowMineruAuthKey] = useState(false)
  // 是否显示 MinerU Token 明文
  const [showMineruToken, setShowMineruToken] = useState(false)
  // MinerU 测试连接状态
  const [mineruValidating, setMineruValidating] = useState(false)
  // MinerU 测试连接结果
  const [mineruValidateStatus, setMineruValidateStatus] = useState(null)
  const [mineruValidateMessage, setMineruValidateMessage] = useState('')
  // MinerU 保存状态
  const [mineruSaving, setMineruSaving] = useState(false)
  const [mineruSaveMessage, setMineruSaveMessage] = useState('')
  // MinerU 配置卡片展开/折叠状态
  const [mineruExpanded, setMineruExpanded] = useState(false)
  // YOLO 资源状态
  const [yoloStatus, setYoloStatus] = useState(null)
  // YOLO 权重安装目录
  const [yoloInstallDir, setYoloInstallDir] = useState('')
  // 手动指定的 YOLO 权重路径
  const [yoloModelPath, setYoloModelPath] = useState('')
  // YOLO 资源操作状态
  const [yoloBusy, setYoloBusy] = useState(false)
  const [yoloMessage, setYoloMessage] = useState('')
  const [yoloMessageType, setYoloMessageType] = useState(null)

  /**
   * 从 localStorage 加载已保存的设置（mode 和 backend）
   */
  useEffect(() => {
    if (isOpen) {
      const settings = loadOCRSettings()
      setMode(settings.mode)
      setBackend(settings.backend)
      setParseRoute(settings.parseRoute)
      setMoreOpen(false)
    }
  }, [isOpen])

  /**
   * 调用后端 API 获取 OCR 状态
   */
  const fetchOCRStatus = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE_URL}/api/ocr/status`)
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const data = await res.json()
      setOcrStatus(data)
      if (data?.figure_preview?.yolo) {
        setYoloStatus(data.figure_preview.yolo)
        setYoloInstallDir((current) => current || data.figure_preview.yolo.default_install_dir || '')
        setYoloModelPath((current) => current || data.figure_preview.yolo.configured_model_path || data.figure_preview.yolo.model_path || '')
      }
    } catch (err) {
      console.error('获取 OCR 状态失败:', err)
      setError('无法获取 OCR 状态，请检查后端服务是否运行')
    } finally {
      setLoading(false)
    }
  }, [])

  /**
   * 单独获取 YOLO 资源状态
   */
  const fetchYoloStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/layout/yolo/status`)
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const data = await res.json()
      setYoloStatus(data)
      setYoloInstallDir((current) => current || data.default_install_dir || '')
      setYoloModelPath((current) => current || data.configured_model_path || data.model_path || '')
      return data
    } catch (err) {
      console.error('获取 YOLO 资源状态失败:', err)
      setYoloMessageType('error')
      setYoloMessage('无法获取 YOLO 资源状态')
      return null
    }
  }, [])

  /**
   * 加载已有的在线 OCR 配置
   */
  const fetchOnlineConfig = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/ocr/online-config`)
      if (!res.ok) return
      const data = await res.json()
      setOnlineConfig(data)
      // 加载 MinerU 已保存配置（回填非敏感字段）
      if (data?.mineru) {
        if (data.mineru.access_mode) {
          setMineruAccessMode(data.mineru.access_mode)
        }
        if (data.mineru.base_url) {
          setMineruBaseUrl(data.mineru.base_url)
        }
        if (data.mineru.worker_url) {
          setMineruWorkerUrl(data.mineru.worker_url)
        }
        if (data.mineru.token_mode) {
          setMineruTokenMode(data.mineru.token_mode)
        }
        if (data.mineru.enable_ocr !== undefined) {
          setMineruEnableOcr(data.mineru.enable_ocr)
        }
        if (data.mineru.enable_formula !== undefined) {
          setMineruEnableFormula(data.mineru.enable_formula)
        }
        if (data.mineru.enable_table !== undefined) {
          setMineruEnableTable(data.mineru.enable_table)
        }
        if (data.mineru.model_version) {
          setMineruModelVersion(data.mineru.model_version)
        }
        const configured = Boolean(data.mineru.worker_url || data.mineru.token_configured)
        if (!configured) setMineruExpanded(true)
      }
    } catch (err) {
      console.error('获取在线 OCR 配置失败:', err)
    }
  }, [])

  const clearMineruValidation = useCallback(() => {
    setMineruValidateStatus(null)
    setMineruValidateMessage('')
  }, [])

  /**
   * MinerU 测试连接：验证 Worker 可达性
   */
  const handleMineruValidate = useCallback(async () => {
    const hasDirectToken = Boolean(mineruToken.trim() || onlineConfig?.mineru?.token_configured)
    const hasWorker = Boolean(mineruWorkerUrl.trim() || onlineConfig?.mineru?.worker_url)
    if (mineruAccessMode === 'worker' && !hasWorker) {
      setMineruValidateStatus('error')
      setMineruValidateMessage('请先填写 Worker URL')
      return
    }
    if (mineruAccessMode === 'direct' && !hasDirectToken) {
      setMineruValidateStatus('error')
      setMineruValidateMessage('请先填写 MinerU Token')
      return
    }
    setMineruValidating(true)
    setMineruValidateStatus(null)
    setMineruValidateMessage('正在测试连接')
    setMineruSaveMessage('')
    try {
      const res = await fetch(`${API_BASE_URL}/api/ocr/validate-key`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: 'mineru',
          access_mode: mineruAccessMode,
          base_url: mineruBaseUrl.trim() || 'https://mineru.net/api/v4',
          worker_url: mineruWorkerUrl.trim(),
          auth_key: mineruAuthKey.trim(),
          token: mineruToken.trim(),
          token_mode: mineruTokenMode,
          model_version: mineruModelVersion,
        }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        setMineruValidateStatus('error')
        setMineruValidateMessage(formatApiDetail(data, `请求失败 (HTTP ${res.status})`))
        return
      }
      setMineruValidateStatus(data.valid ? 'success' : 'error')
      if (data.valid) {
        const saveRes = await fetch(`${API_BASE_URL}/api/ocr/online-config`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            provider: 'mineru',
            access_mode: mineruAccessMode,
            base_url: mineruBaseUrl.trim() || 'https://mineru.net/api/v4',
            worker_url: mineruWorkerUrl.trim(),
            auth_key: mineruAuthKey.trim(),
            token: mineruToken.trim(),
            token_mode: mineruTokenMode,
            model_version: mineruModelVersion,
            enable_ocr: mineruEnableOcr,
            enable_formula: mineruEnableFormula,
            enable_table: mineruEnableTable,
          }),
        })
        const saveData = await saveRes.json().catch(() => ({}))
        if (!saveRes.ok || !saveData.success) {
          setMineruValidateStatus('error')
          setMineruValidateMessage(formatApiDetail(saveData, '连接成功，但配置保存失败'))
          return
        }
        setMineruValidateMessage(`${data.message || '连接成功'}，已保存配置`)
        fetchOnlineConfig()
        fetchOCRStatus()
        setMineruAuthKey('')
        setMineruToken('')
      } else {
        setMineruValidateMessage(data.message || '连接失败')
      }
    } catch (err) {
      console.error('MinerU 测试连接失败:', err)
      setMineruValidateStatus('error')
      setMineruValidateMessage('无法连接到服务器，请检查后端服务')
    } finally {
      setMineruValidating(false)
    }
  }, [
    mineruAccessMode,
    mineruBaseUrl,
    mineruWorkerUrl,
    mineruAuthKey,
    mineruToken,
    mineruTokenMode,
    mineruModelVersion,
    mineruEnableOcr,
    mineruEnableFormula,
    mineruEnableTable,
    onlineConfig?.mineru?.token_configured,
    onlineConfig?.mineru?.worker_url,
    fetchOnlineConfig,
    fetchOCRStatus,
  ])

  /**
   * 保存 MinerU OCR 配置
   */
  const handleMineruSave = useCallback(async () => {
    setMineruSaving(true)
    setMineruSaveMessage('')
    try {
      const body = {
        provider: 'mineru',
        access_mode: mineruAccessMode,
        base_url: mineruBaseUrl.trim() || 'https://mineru.net/api/v4',
        worker_url: mineruWorkerUrl.trim(),
        auth_key: mineruAuthKey.trim(),
        token_mode: mineruTokenMode,
        token: mineruToken.trim(),
        model_version: mineruModelVersion,
        enable_ocr: mineruEnableOcr,
        enable_formula: mineruEnableFormula,
        enable_table: mineruEnableTable,
      }
      const res = await fetch(`${API_BASE_URL}/api/ocr/online-config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json().catch(() => ({}))
      if (res.ok && data.success) {
        setMineruSaveMessage('配置已保存，请测试连接确认 Token 有效性')
        // 重新加载配置和状态
        fetchOnlineConfig()
        fetchOCRStatus()
        // 清空敏感输入（已保存到后端）
        setMineruAuthKey('')
        setMineruToken('')
        setMineruValidateStatus(null)
        setMineruValidateMessage('')
      } else {
        setMineruSaveMessage(data.detail || data.message || `保存失败 (HTTP ${res.status})`)
      }
    } catch (err) {
      console.error('保存 MinerU 配置失败:', err)
      setMineruSaveMessage('保存失败，请检查后端服务')
    } finally {
      setMineruSaving(false)
      // 3 秒后清除保存消息
      setTimeout(() => setMineruSaveMessage(''), 3000)
    }
  }, [mineruAccessMode, mineruBaseUrl, mineruWorkerUrl, mineruAuthKey, mineruTokenMode, mineruToken, mineruModelVersion, mineruEnableOcr, mineruEnableFormula, mineruEnableTable, fetchOnlineConfig, fetchOCRStatus])

  /**
   * 选择 YOLO 权重安装目录（桌面端弹系统目录选择器，Web 端保留手动输入）
   */
  const handleSelectYoloInstallDir = useCallback(async () => {
    if (!window.chatpdfDesktop?.selectDirectory) return
    const dir = await window.chatpdfDesktop.selectDirectory()
    if (dir) setYoloInstallDir(dir)
  }, [])

  /**
   * 选择已有 YOLO 权重文件
   */
  const handleSelectYoloModelFile = useCallback(async () => {
    if (!window.chatpdfDesktop?.selectFile) return
    const filePath = await window.chatpdfDesktop.selectFile({
      filters: [{ name: 'PyTorch Weights', extensions: ['pt'] }],
    })
    if (filePath) setYoloModelPath(filePath)
  }, [])

  /**
   * 一键下载 YOLO 权重
   */
  const handleDownloadYoloModel = useCallback(async () => {
    setYoloBusy(true)
    setYoloMessageType(null)
    setYoloMessage('正在下载 YOLO 权重，请保持网络连接')
    try {
      const res = await fetch(`${API_BASE_URL}/api/layout/yolo/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          install_dir: yoloInstallDir.trim(),
        }),
      })
      if (!res.ok) {
        throw new Error(await getApiErrorMessage(res, `下载失败 (HTTP ${res.status})`))
      }
      const data = await res.json()
      setYoloStatus(data)
      setYoloModelPath(data.model_path || '')
      setYoloInstallDir(data.default_install_dir || yoloInstallDir)
      setYoloMessageType('success')
      setYoloMessage(data.downloaded === false ? '权重已存在，配置已启用' : 'YOLO 权重已下载并启用')
      fetchOCRStatus()
    } catch (err) {
      console.error('下载 YOLO 权重失败:', err)
      setYoloMessageType('error')
      setYoloMessage(err.message || '下载失败，请检查网络或手动指定权重路径')
    } finally {
      setYoloBusy(false)
    }
  }, [yoloInstallDir, fetchOCRStatus])

  /**
   * 保存手动指定的 YOLO 权重路径
   */
  const handleSaveYoloModelPath = useCallback(async () => {
    if (!yoloModelPath.trim()) return
    setYoloBusy(true)
    setYoloMessageType(null)
    setYoloMessage('')
    try {
      const res = await fetch(`${API_BASE_URL}/api/layout/yolo/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: yoloModelPath.trim() }),
      })
      if (!res.ok) {
        throw new Error(await getApiErrorMessage(res, `保存失败 (HTTP ${res.status})`))
      }
      const data = await res.json()
      setYoloStatus(data)
      setYoloModelPath(data.configured_model_path || data.model_path || yoloModelPath)
      setYoloMessageType('success')
      setYoloMessage('YOLO 权重路径已保存')
      fetchOCRStatus()
    } catch (err) {
      console.error('保存 YOLO 权重路径失败:', err)
      setYoloMessageType('error')
      setYoloMessage(err.message || '保存失败，请检查路径')
    } finally {
      setYoloBusy(false)
    }
  }, [yoloModelPath, fetchOCRStatus])

  /**
   * 清除手动 YOLO 路径配置
   */
  const handleResetYoloModelPath = useCallback(async () => {
    setYoloBusy(true)
    setYoloMessageType(null)
    setYoloMessage('')
    try {
      const res = await fetch(`${API_BASE_URL}/api/layout/yolo/reset`, {
        method: 'POST',
      })
      if (!res.ok) {
        throw new Error(await getApiErrorMessage(res, `重置失败 (HTTP ${res.status})`))
      }
      const data = await res.json()
      setYoloStatus(data)
      setYoloModelPath(data.model_path || '')
      setYoloInstallDir(data.default_install_dir || '')
      setYoloMessageType('success')
      setYoloMessage('已恢复默认 YOLO 权重目录')
      fetchOCRStatus()
    } catch (err) {
      console.error('重置 YOLO 权重路径失败:', err)
      setYoloMessageType('error')
      setYoloMessage(err.message || '重置失败')
    } finally {
      setYoloBusy(false)
    }
  }, [fetchOCRStatus])

  /**
   * 面板打开时获取 OCR 状态和在线配置
   */
  useEffect(() => {
    if (isOpen) {
      fetchOCRStatus()
      fetchOnlineConfig()
      fetchYoloStatus()
    }
  }, [isOpen, fetchOCRStatus, fetchOnlineConfig, fetchYoloStatus])

  /**
   * 切换 OCR 模式并持久化到 localStorage
   * @param {string} newMode - 新的 OCR 模式
   */
  const handleModeChange = (newMode) => {
    setMode(newMode)
    const settings = loadOCRSettings()
    saveOCRSettings({ ...settings, mode: newMode, backend })
  }

  /**
   * 切换 OCR 引擎后端并持久化到 localStorage
   * @param {string} newBackend - 新的 OCR 引擎后端
   */
  const handleBackendChange = (newBackend) => {
    setBackend(newBackend)
    const settings = loadOCRSettings()
    saveOCRSettings({ ...settings, mode, backend: newBackend })
  }

  /**
   * 切换上传主解析路线并持久化。路线只影响后续上传，不改写已上传文档。
   * @param {'auto'|'local'|'mineru'} newRoute - 主解析路线
   */
  const handleParseRouteChange = (newRoute) => {
    if (!VALID_PARSE_ROUTES.includes(newRoute)) return
    setParseRoute(newRoute)
    const settings = loadOCRSettings()
    saveOCRSettings({ ...settings, parseRoute: newRoute })
    if (newRoute !== 'local') {
      const configured = Boolean(
        onlineConfig?.mineru?.worker_url || onlineConfig?.mineru?.token_configured
      )
      if (!configured) setMineruExpanded(true)
    }
  }

  const yoloReady = yoloStatus?.available === true
  const yoloDependencyMissing = yoloStatus && yoloStatus.dependencies_available === false
  const yoloInstalled = yoloStatus?.model_installed === true
  const yoloStatusLabel = yoloReady
    ? '已就绪'
    : yoloDependencyMissing
      ? '依赖缺失'
      : yoloInstalled
        ? '待验证'
        : '未安装'
  const mineruConfigured = Boolean(
    onlineConfig?.mineru?.worker_url || onlineConfig?.mineru?.token_configured
  )
  // 只在当前弹窗会话里、且测试请求真实通过后才展示“已验证”。
  // 配置文件只保存密钥，不保存可长期信赖的连接状态，避免 Token 过期后
  // 重开页面仍被误标为已连接。
  const mineruConnectionVerified = mineruValidateStatus === 'success'
  const mineruCanTest = mineruAccessMode === 'direct'
    ? Boolean(mineruToken.trim() || onlineConfig?.mineru?.token_configured)
    : Boolean(mineruWorkerUrl.trim() || onlineConfig?.mineru?.worker_url)
  const localOcrBackends = Object.entries(ocrStatus?.backends || {}).filter(
    ([name]) => VALID_BACKENDS.includes(name) && name !== 'auto'
  )
  const localOcrAvailable = Boolean(
    typeof ocrStatus?.local_available === 'boolean'
      ? ocrStatus.local_available
      : localOcrBackends.some(([, available]) => available)
  )
  const recommendedLocalBackend = BACKEND_LABELS[ocrStatus?.recommended]
  const showLocalOcrInstallGuide = shouldShowLocalOcrInstallGuide(ocrStatus)

  return (
    <AnimatePresence initial={false}>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          aria-hidden={!isOpen}
          className={`fixed inset-0 bg-slate-950/25 z-50 flex items-center justify-center p-4 transition-all opacity-100 ${isOpen ? 'pointer-events-auto' : 'pointer-events-none'}`}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.95, opacity: 0, y: 10 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30, mass: 0.8 }}
            className="settings-solid settings-shell w-full max-w-[720px] max-h-[92vh] bg-[#f6f7f9] border border-white/80 overflow-hidden flex flex-col"
          >
            {/* 顶部标题栏 */}
            <div className="settings-chrome flex items-center px-6 py-5 sticky top-0 z-10 border-b border-gray-200">
              <div className="flex items-center gap-3">
                <button
                  onClick={onClose}
                  className="p-2 -ml-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full transition-colors"
                  title="返回设置中心"
                  aria-label="返回设置中心"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>
                <div className="w-10 h-10 bg-[#fcede8] rounded-[14px] flex items-center justify-center text-[#ed8c68]">
                  <ScanText className="w-5 h-5" />
                </div>
                <div>
                  <div className="text-[17px] font-bold text-gray-900 tracking-tight">
                    文档解析
                  </div>
                  <div className="text-[12px] font-medium text-gray-500">
                    先选上传路线，再配这条路线要用的项
                  </div>
                </div>
              </div>
            </div>

            {/* 主内容区（单栏） */}
            <div className="flex-1 overflow-y-auto p-6 space-y-5">
              {/* 错误提示 */}
              {error && (
                <div className="flex items-start gap-3 p-4 rounded-2xl border border-red-100 bg-red-50/60 text-red-700 text-sm">
                  <XCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                  <div>
                    <div className="font-medium">获取状态失败</div>
                    <div className="text-red-600 mt-0.5">{error}</div>
                  </div>
                </div>
              )}

              <section className="px-1" aria-labelledby="parse-route-heading">
                <div className="flex items-start justify-between gap-4 mb-4">
                  <div>
                    <div id="parse-route-heading" className="text-sm font-semibold text-gray-800">上传路线</div>
                    <p className="text-[12px] leading-5 text-gray-500 mt-1">
                      之后上传的文档都走这条路线，已上传的不受影响。
                    </p>
                  </div>
                  <span className="shrink-0 rounded-lg bg-gray-100 px-2 py-1 text-[10px] font-semibold text-gray-500">
                    影响后续上传
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  {PARSE_ROUTE_OPTIONS.map((option) => {
                    const Icon = option.icon
                    const isActive = parseRoute === option.value
                    const isMinerU = option.value === 'mineru'
                    return (
                      <button
                        key={option.value}
                        type="button"
                        aria-pressed={isActive}
                        onClick={() => handleParseRouteChange(option.value)}
                        className={`relative min-h-[132px] rounded-[16px] border p-4 text-left transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#ed8c68]/35 ${
                          isActive
                            ? 'border-[#ed8c68]/40 bg-[#fff8f5] shadow-[0_8px_24px_-20px_rgba(184,95,71,0.7)]'
                            : 'border-gray-200 bg-white hover:-translate-y-0.5 hover:border-[#ed8c68]/25 hover:bg-[#fffbf9]'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className={`grid h-9 w-9 place-items-center rounded-[11px] ${
                            isActive ? 'bg-[#fcede8] text-[#d96f50]' : 'bg-gray-100 text-gray-500'
                          }`}>
                            <Icon className="h-4 w-4" />
                          </div>
                          <span className={`rounded-md px-2 py-1 text-[10px] font-semibold ${
                            isMinerU
                              ? 'bg-[#fcede8] text-[#b85f47]'
                              : 'bg-gray-100 text-gray-500'
                          }`}>
                            {isMinerU ? '结构优先' : '文件不出机'}
                          </span>
                        </div>
                        <div className="mt-3 flex items-center gap-2 text-sm font-semibold text-gray-800">
                          {option.label}
                          {isActive && <CheckCircle2 className="h-4 w-4 text-[#d96f50]" />}
                        </div>
                        <p className="mt-1.5 text-[11px] leading-5 text-gray-500">
                          {isMinerU
                            ? '论文和扫描件更合适，公式表格更完整。'
                            : '适合文字层完整的普通 PDF。'}
                        </p>
                      </button>
                    )
                  })}
                </div>
              </section>

              {parseRoute !== 'local' && (
                <section className="space-y-4" aria-labelledby="mineru-settings-heading">
                  <div className="settings-card bg-white p-5 border border-gray-200/90">
                    <div className="flex items-start justify-between gap-3 mb-4">
                      <div>
                        <div id="mineru-settings-heading" className="text-sm font-semibold text-gray-800">MinerU 连接</div>
                        <p className="text-[11px] leading-5 text-gray-500 mt-1">
                          深度解析会把 PDF 发到 MinerU。
                        </p>
                      </div>
                      {mineruConfigured && (
                        <span className={`shrink-0 text-[11px] px-2 py-0.5 rounded-full border flex items-center gap-1 ${
                          mineruConnectionVerified
                            ? 'text-green-700 bg-green-50 border-green-100'
                            : 'text-amber-700 bg-amber-50 border-amber-100'
                        }`}>
                          {mineruConnectionVerified
                            ? <CheckCircle2 className="w-3 h-3" />
                            : <Key className="w-3 h-3" />}
                          {mineruConnectionVerified ? '已验证' : '已配置'}
                        </span>
                      )}
                    </div>

                    {mineruConfigured && !mineruExpanded && (
                      <div className="space-y-1.5 mb-4">
                        <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-gray-50/80 text-xs text-gray-600">
                          <Globe className="w-3.5 h-3.5 text-gray-400 shrink-0" />
                          <span>{onlineConfig.mineru.access_mode === 'direct' ? '直连 API' : 'Worker 代理'}</span>
                          <code className="font-mono text-gray-700 truncate">
                            {onlineConfig.mineru.access_mode === 'direct' ? onlineConfig.mineru.base_url : onlineConfig.mineru.worker_url}
                          </code>
                        </div>
                        {onlineConfig.mineru.token_configured && (
                          <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-gray-50/80 text-xs text-gray-600">
                            <Key className="w-3.5 h-3.5 text-gray-400 shrink-0" />
                            <span>Token {onlineConfig.mineru.token_preview}</span>
                          </div>
                        )}
                      </div>
                    )}

                    <button
                      type="button"
                      onClick={() => setMineruExpanded(!mineruExpanded)}
                      className="inline-flex items-center gap-1.5 text-[12px] font-semibold text-gray-600 hover:text-[#b85f47]"
                    >
                      <ChevronDown className={`w-4 h-4 transition-transform ${mineruExpanded ? 'rotate-180' : ''}`} />
                      {mineruExpanded ? '收起连接设置' : mineruConfigured ? '修改连接' : '填写连接信息'}
                    </button>

                    <AnimatePresence initial={false}>
                      {mineruExpanded && (
                        <motion.div
                          key="mineru-connection"
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.2, ease: 'easeInOut' }}
                          style={{ overflow: 'hidden' }}
                        >
                          <div className="mt-4 space-y-4">
                            <div>
                              <label className="text-xs font-medium text-gray-600 mb-1.5 block">接入方式</label>
                              <SettingsSegmentedControl
                                ariaLabel="MinerU 接入模式"
                                value={mineruAccessMode}
                                onChange={(value) => {
                                  setMineruAccessMode(value)
                                  clearMineruValidation()
                                }}
                                options={[
                                  { value: 'direct', label: '直连 API' },
                                  { value: 'worker', label: 'Worker 代理' },
                                ]}
                                buttonClassName="px-3 py-2 text-xs font-medium text-center rounded-[10px]"
                              />
                              <div className="text-xs text-gray-400 mt-1">
                                {mineruAccessMode === 'worker'
                                  ? '经你部署的代理转发 MinerU 请求'
                                  : '后端直接调用 MinerU 官方 API'}
                              </div>
                            </div>

                            {mineruAccessMode === 'direct' && (
                              <div>
                                <label className="text-xs font-medium text-gray-600 mb-1.5 block">API 地址</label>
                                <div className="relative">
                                  <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
                                    <Globe className="w-4 h-4" />
                                  </div>
                                  <input
                                    type="text"
                                    value={mineruBaseUrl}
                                    onChange={(e) => {
                                      setMineruBaseUrl(e.target.value)
                                      clearMineruValidation()
                                    }}
                                    placeholder="https://mineru.net/api/v4"
                                    className={FIELD_INPUT_CLASS}
                                  />
                                </div>
                              </div>
                            )}

                            {mineruAccessMode === 'worker' && (
                              <div>
                                <label className="text-xs font-medium text-gray-600 mb-1.5 block">Worker URL</label>
                                <div className="relative">
                                  <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
                                    <Globe className="w-4 h-4" />
                                  </div>
                                  <input
                                    type="text"
                                    value={mineruWorkerUrl}
                                    onChange={(e) => {
                                      setMineruWorkerUrl(e.target.value)
                                      clearMineruValidation()
                                    }}
                                    placeholder="https://your-worker.workers.dev"
                                    className={FIELD_INPUT_CLASS}
                                  />
                                </div>
                              </div>
                            )}

                            {mineruAccessMode === 'worker' && (
                              <div>
                                <label className="text-xs font-medium text-gray-600 mb-1.5 block">Auth Key（可选）</label>
                                <div className="relative">
                                  <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
                                    <Key className="w-4 h-4" />
                                  </div>
                                  <input
                                    type={showMineruAuthKey ? 'text' : 'password'}
                                    value={mineruAuthKey}
                                    onChange={(e) => {
                                      setMineruAuthKey(e.target.value)
                                      clearMineruValidation()
                                    }}
                                    placeholder={
                                      onlineConfig?.mineru?.auth_key_configured
                                        ? '输入新 Auth Key 以更新（留空保持不变）'
                                        : 'Worker 开启访问控制时填写'
                                    }
                                    className={`${FIELD_INPUT_CLASS} pr-10`}
                                  />
                                  <button
                                    type="button"
                                    onClick={() => setShowMineruAuthKey(!showMineruAuthKey)}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                                    aria-label={showMineruAuthKey ? '隐藏 Auth Key' : '显示 Auth Key'}
                                  >
                                    {showMineruAuthKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                  </button>
                                </div>
                              </div>
                            )}

                            {mineruAccessMode === 'worker' && (
                              <div>
                                <label className="text-xs font-medium text-gray-600 mb-1.5 block">Token 来源</label>
                                <SettingsSegmentedControl
                                  ariaLabel="MinerU Token 模式"
                                  value={mineruTokenMode}
                                  onChange={(value) => {
                                    setMineruTokenMode(value)
                                    clearMineruValidation()
                                  }}
                                  options={[
                                    { value: 'frontend', label: '在此填写' },
                                    { value: 'worker', label: 'Worker 环境变量' },
                                  ]}
                                  buttonClassName="px-3 py-2 text-xs font-medium text-center rounded-[10px]"
                                />
                              </div>
                            )}

                            {(mineruAccessMode === 'direct' || mineruTokenMode === 'frontend') && (
                              <div>
                                <label className="text-xs font-medium text-gray-600 mb-1.5 block">MinerU Token</label>
                                <div className="relative">
                                  <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
                                    <Key className="w-4 h-4" />
                                  </div>
                                  <input
                                    type={showMineruToken ? 'text' : 'password'}
                                    value={mineruToken}
                                    onChange={(e) => {
                                      setMineruToken(e.target.value)
                                      clearMineruValidation()
                                    }}
                                    placeholder={
                                      onlineConfig?.mineru?.token_configured
                                        ? '输入新 Token 以更新（留空保持不变）'
                                        : '输入 MinerU API Token'
                                    }
                                    className={`${FIELD_INPUT_CLASS} pr-10`}
                                  />
                                  <button
                                    type="button"
                                    onClick={() => setShowMineruToken(!showMineruToken)}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                                    aria-label={showMineruToken ? '隐藏 Token' : '显示 Token'}
                                  >
                                    {showMineruToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                  </button>
                                </div>
                              </div>
                            )}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>

                  <div className="settings-card bg-white p-5 border border-gray-200/90">
                    <div className="mb-4">
                      <div className="text-sm font-semibold text-gray-800">解析增强</div>
                      <p className="text-[11px] leading-5 text-gray-500 mt-1">
                        仅 MinerU 深度解析生效。扫描件抽字很差时仍会自动开 OCR。
                      </p>
                    </div>
                    <div className="space-y-2">
                      <OptionToggle
                        title="扫描件识别"
                        description="图片型 PDF 需要打开；普通论文可关"
                        checked={mineruEnableOcr}
                        onToggle={() => setMineruEnableOcr(!mineruEnableOcr)}
                      />
                      <OptionToggle
                        title="公式"
                        checked={mineruEnableFormula}
                        onToggle={() => setMineruEnableFormula(!mineruEnableFormula)}
                      />
                      <OptionToggle
                        title="表格"
                        checked={mineruEnableTable}
                        onToggle={() => setMineruEnableTable(!mineruEnableTable)}
                      />
                      <div className="px-3 py-2.5 rounded-xl bg-gray-50/80 border border-gray-100">
                        <div className="flex items-center justify-between gap-3">
                          <span>
                            <span className="block text-xs text-gray-700">解析模型</span>
                            <span className="block text-[10px] text-gray-400 mt-0.5">复杂版式用 VLM</span>
                          </span>
                          <SettingsSegmentedControl
                            ariaLabel="MinerU 解析模型"
                            value={mineruModelVersion}
                            onChange={setMineruModelVersion}
                            options={[
                              { value: 'vlm', label: 'VLM' },
                              { value: 'pipeline', label: 'Pipeline' },
                            ]}
                            className="w-[152px] rounded-lg"
                            buttonClassName="px-2 py-1 text-[11px] font-medium text-center rounded-md"
                            indicatorClassName="rounded-md"
                          />
                        </div>
                      </div>
                    </div>

                    {(mineruValidateMessage || mineruSaveMessage) && (
                      <div
                        className={`mt-3 flex items-center gap-2 px-3 py-2 rounded-xl text-xs ${
                          mineruValidating
                            ? 'bg-gray-50 border border-gray-100 text-gray-600'
                            : mineruValidateStatus === 'success'
                              ? 'bg-green-50/60 border border-green-100 text-green-700'
                              : mineruValidateStatus === 'error'
                                ? 'bg-red-50/60 border border-red-100 text-red-700'
                                : 'bg-amber-50/60 border border-amber-100 text-amber-800'
                        }`}
                      >
                        {mineruValidating ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
                        ) : mineruValidateStatus === 'success' ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-green-500 shrink-0" />
                        ) : mineruValidateStatus === 'error' ? (
                          <XCircle className="w-3.5 h-3.5 text-red-500 shrink-0" />
                        ) : (
                          <AlertTriangle className="w-3.5 h-3.5 text-amber-500 shrink-0" />
                        )}
                        <span>{mineruValidateMessage || mineruSaveMessage}</span>
                      </div>
                    )}

                    <div className="mt-4 flex items-center gap-2">
                      <button
                        type="button"
                        onClick={handleMineruValidate}
                        disabled={mineruValidating || mineruSaving || !mineruCanTest}
                        className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium rounded-xl border border-[#ed8c68]/30 bg-[#ed8c68]/5 text-[#ed8c68] hover:bg-[#ed8c68]/10 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                      >
                        {mineruValidating ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Wifi className="w-3.5 h-3.5" />}
                        {mineruValidating ? '测试中' : '测试连接'}
                      </button>
                      <button
                        type="button"
                        onClick={handleMineruSave}
                        disabled={mineruSaving || mineruValidating || !mineruCanTest}
                        className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium rounded-xl border border-green-200 bg-green-50/60 text-green-700 hover:bg-green-100/60 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                      >
                        {mineruSaving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
                        保存
                      </button>
                    </div>
                  </div>
                </section>
              )}

              {parseRoute === 'local' && (
                <section className="space-y-4" aria-labelledby="local-ocr-heading">
                  <div className="settings-card bg-white p-5 border border-gray-200/90">
                    <div className="mb-4">
                      <div id="local-ocr-heading" className="text-sm font-semibold text-gray-800">扫描页补字</div>
                      <p className="text-[11px] leading-5 text-gray-500 mt-1">
                        给本地解析补识别，默认自动即可。
                      </p>
                    </div>

                    <div className="mb-4">
                      <SettingsSegmentedControl
                        ariaLabel="扫描页补字时机"
                        value={mode}
                        onChange={handleModeChange}
                        options={OCR_MODES.map((option) => ({
                          value: option.value,
                          label: option.label,
                        }))}
                        buttonClassName="px-3 py-2 text-xs font-medium text-center rounded-[10px]"
                      />
                      <p className="text-[11px] text-gray-400 mt-2">
                        {OCR_MODES.find((option) => option.value === mode)?.description}
                      </p>
                    </div>

                    <div className="flex items-center justify-between gap-3 mb-3">
                      <span className="text-xs font-medium text-gray-600">本地引擎</span>
                      <span className={`text-[11px] ${localOcrAvailable ? 'text-green-700' : 'text-gray-500'}`}>
                        {loading
                          ? '正在检查'
                          : localOcrAvailable
                            ? `可用${recommendedLocalBackend ? ` · 推荐 ${recommendedLocalBackend}` : ''}`
                            : '未就绪'}
                      </span>
                    </div>

                    <div className="space-y-2">
                      {BACKEND_OPTIONS.map((option) => {
                        const isActive = backend === option.value
                        const localProviderUnavailable =
                          ['tesseract', 'paddleocr'].includes(option.value) &&
                          ocrStatus?.backends?.[option.value] === false
                        const disabled = Boolean(option.deprecatedForPageOcr || localProviderUnavailable)
                        const reason = option.value !== 'auto' ? ocrStatus?.diagnostics?.[option.value]?.reason : ''
                        return (
                          <button
                            key={option.value}
                            type="button"
                            disabled={disabled}
                            onClick={() => !disabled && handleBackendChange(option.value)}
                            className={`w-full flex items-start gap-3 px-3 py-2.5 rounded-xl border text-left transition-all ${
                              isActive
                                ? 'border-[#ed8c68]/30 bg-[#ed8c68]/5'
                                : disabled
                                  ? 'border-gray-100 bg-gray-50 cursor-not-allowed opacity-70'
                                  : 'border-gray-100 hover:border-[#ed8c68]/30 hover:bg-[#fffbf9]'
                            }`}
                          >
                            <div className="flex-1 min-w-0">
                              <div className={`text-sm font-medium ${isActive ? 'text-[#ed8c68]' : 'text-gray-700'}`}>
                                {option.label}
                              </div>
                              <div className="text-[11px] text-gray-400 mt-0.5">
                                {reason || option.description}
                              </div>
                            </div>
                            {isActive && <CheckCircle2 className="w-4 h-4 text-[#ed8c68] mt-0.5 shrink-0" />}
                          </button>
                        )
                      })}
                    </div>

                    {ocrStatus?.poppler_available === false && (
                      <div className="mt-3 text-[11px] leading-5 text-amber-700 bg-amber-50/80 border border-amber-100 rounded-xl px-3 py-2">
                        还缺 Poppler，PDF 无法转成图片，本地补字不能工作。
                      </div>
                    )}
                  </div>

                  {ocrStatus && showLocalOcrInstallGuide && (
                    <div className="settings-card bg-white p-5 border border-gray-200/90">
                      <div className="flex items-start gap-3">
                        <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                        <div>
                          <div className="text-sm font-semibold text-gray-800">本机还不能补字</div>
                          <p className="text-xs text-gray-500 mt-1 leading-relaxed">
                            需要 Tesseract 或 PaddleOCR。走 MinerU 深度解析则不必安装。
                          </p>
                          {ocrStatus.install_instructions && (
                            <div className="mt-3 space-y-1.5">
                              {Object.entries(ocrStatus.install_instructions).map(([key, instruction]) => (
                                <div key={key} className="text-xs bg-gray-50 border border-gray-100 rounded-lg px-3 py-2">
                                  <span className="font-medium text-gray-700">{key}：</span>
                                  <code className="text-gray-600 ml-1 break-all">{instruction}</code>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </section>
              )}

              <section className="settings-card bg-white border border-gray-200/90 overflow-hidden">
                <button
                  type="button"
                  onClick={() => setMoreOpen(!moreOpen)}
                  className="w-full flex items-center gap-2 px-5 py-4 text-left"
                  aria-expanded={moreOpen}
                >
                  <SlidersHorizontal className="w-4 h-4 text-gray-400" />
                  <span className="flex-1">
                    <span className="block text-sm font-semibold text-gray-800">图表兜底</span>
                    <span className="block text-[11px] text-gray-400 mt-0.5">可选。主解析没定位到图时，才用本地模型裁切</span>
                  </span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${
                    yoloReady
                      ? 'bg-green-50 border-green-100 text-green-700'
                      : yoloDependencyMissing
                        ? 'bg-red-50 border-red-100 text-red-700'
                        : 'bg-gray-50 border-gray-200 text-gray-500'
                  }`}>
                    {yoloStatusLabel}
                  </span>
                  <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${moreOpen ? 'rotate-180' : ''}`} />
                </button>

                <AnimatePresence initial={false}>
                  {moreOpen && (
                    <motion.div
                      key="figure-fallback"
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2, ease: 'easeInOut' }}
                      style={{ overflow: 'hidden' }}
                    >
                      <div className="px-5 pb-5 space-y-3">
                        <div className="rounded-[12px] border border-[#eadfd9] bg-[#faf7f5] px-4 py-3">
                          <p className="text-[11px] leading-5 text-gray-500">
                            MinerU 文档优先用结构化图表，本地文档优先用 PDF 原生结构。DocLayout-YOLO 只负责定位和裁切。
                          </p>
                        </div>

                        <div className="flex items-center justify-between gap-3">
                          <span className="text-xs font-medium text-gray-700">本地图表定位模型</span>
                          <button
                            type="button"
                            onClick={fetchYoloStatus}
                            disabled={yoloBusy}
                            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-medium rounded-lg border border-gray-200 bg-white text-gray-600 hover:bg-gray-100 disabled:opacity-40"
                          >
                            <RefreshCw className={`w-3 h-3 ${yoloBusy ? 'animate-spin' : ''}`} />
                            刷新
                          </button>
                        </div>

                        {yoloStatus?.model_path && (
                          <div className="rounded-lg bg-gray-50 border border-gray-100 px-3 py-2">
                            <div className="text-[10px] text-gray-400 mb-1">当前权重</div>
                            <div className="font-mono text-[11px] text-gray-600 break-all">{yoloStatus.model_path}</div>
                          </div>
                        )}

                        {yoloDependencyMissing && (
                          <div className="flex items-start gap-2 px-3 py-2 rounded-xl bg-red-50/70 border border-red-100 text-xs text-red-700">
                            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                            <span>当前后端缺少 YOLO 运行依赖。桌面完整包会保留运行库，权重需在软件内下载或指定。</span>
                          </div>
                        )}

                        <div className="space-y-2">
                          <label className="block text-[11px] font-medium text-gray-500">下载到目录</label>
                          <div className="flex gap-2">
                            <input
                              type="text"
                              value={yoloInstallDir}
                              onChange={(e) => setYoloInstallDir(e.target.value)}
                              placeholder={yoloStatus?.default_install_dir || '默认用户数据目录'}
                              className="flex-1 min-w-0 px-3 py-2 rounded-xl border border-gray-200 bg-white text-xs text-gray-700 focus:outline-none focus:border-[#ed8c68]/50"
                            />
                            {typeof window !== 'undefined' && window.chatpdfDesktop?.selectDirectory && (
                              <button
                                type="button"
                                onClick={handleSelectYoloInstallDir}
                                disabled={yoloBusy}
                                className="shrink-0 inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-xl border border-gray-200 bg-white text-gray-600 hover:bg-gray-100 disabled:opacity-40"
                              >
                                <FolderOpen className="w-3.5 h-3.5" />
                                选择
                              </button>
                            )}
                            <button
                              type="button"
                              onClick={handleDownloadYoloModel}
                              disabled={yoloBusy}
                              className="shrink-0 inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-xl border border-[#ed8c68]/30 bg-[#ed8c68]/5 text-[#ed8c68] hover:bg-[#ed8c68]/10 disabled:opacity-40"
                            >
                              {yoloBusy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />}
                              下载
                            </button>
                          </div>
                        </div>

                        <div className="space-y-2">
                          <label className="block text-[11px] font-medium text-gray-500">指定已有权重</label>
                          <div className="flex gap-2">
                            <input
                              type="text"
                              value={yoloModelPath}
                              onChange={(e) => setYoloModelPath(e.target.value)}
                              placeholder="选择或输入 doclayout_yolo_*.pt 路径"
                              className="flex-1 min-w-0 px-3 py-2 rounded-xl border border-gray-200 bg-white text-xs text-gray-700 focus:outline-none focus:border-[#ed8c68]/50"
                            />
                            {typeof window !== 'undefined' && window.chatpdfDesktop?.selectFile && (
                              <button
                                type="button"
                                onClick={handleSelectYoloModelFile}
                                disabled={yoloBusy}
                                className="shrink-0 inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-xl border border-gray-200 bg-white text-gray-600 hover:bg-gray-100 disabled:opacity-40"
                              >
                                <FolderOpen className="w-3.5 h-3.5" />
                                选择
                              </button>
                            )}
                            <button
                              type="button"
                              onClick={handleSaveYoloModelPath}
                              disabled={yoloBusy || !yoloModelPath.trim()}
                              className="shrink-0 inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-xl border border-green-200 bg-green-50/60 text-green-700 hover:bg-green-100/60 disabled:opacity-40"
                            >
                              <Save className="w-3.5 h-3.5" />
                              保存
                            </button>
                          </div>
                        </div>

                        <div className="flex items-center justify-between gap-3">
                          <span className="text-[11px] text-gray-400">
                            默认目录：{yoloStatus?.default_install_dir || '读取中'}
                          </span>
                          <button
                            type="button"
                            onClick={handleResetYoloModelPath}
                            disabled={yoloBusy}
                            className="text-[11px] text-gray-500 hover:text-gray-700 disabled:opacity-40"
                          >
                            恢复默认
                          </button>
                        </div>

                        {yoloMessage && (
                          <div
                            className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs ${
                              yoloMessageType === 'success'
                                ? 'bg-green-50/70 border border-green-100 text-green-700'
                                : yoloMessageType === 'error'
                                  ? 'bg-red-50/70 border border-red-100 text-red-700'
                                  : 'bg-[#ed8c68]/10 border border-[#ed8c68]/15 text-[#b85f47]'
                            }`}
                          >
                            {yoloMessageType === 'success' ? (
                              <CheckCircle2 className="w-3.5 h-3.5 text-green-500 shrink-0" />
                            ) : yoloMessageType === 'error' ? (
                              <XCircle className="w-3.5 h-3.5 text-red-500 shrink-0" />
                            ) : (
                              <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
                            )}
                            <span>{yoloMessage}</span>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </section>
            </div>

            <div className="settings-chrome px-6 py-3 border-t border-gray-200 flex items-center justify-between">
              <div className="text-xs text-gray-400">
                {parseRoute === 'local' ? '扫描页补字已保存到本机' : '连接信息保存在后端，路线选择保存在本机'}
              </div>
              <div className="text-xs font-medium text-gray-600">
                {parseSettingsStatusText({
                  parseRoute,
                  mineruConfigured,
                  mineruConnectionVerified,
                  ocrModeLabel: OCR_MODES.find((item) => item.value === mode)?.label,
                  ocrBackendLabel: BACKEND_OPTIONS.find((item) => item.value === backend)?.label,
                })}
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
