import React, { useEffect, useState, useCallback } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Crop,
  Download,
  Eye,
  EyeOff,
  FileSearch,
  FolderOpen,
  Globe,
  Info,
  Key,
  Loader2,
  RefreshCw,
  Save,
  ScanText,
  Wifi,
  WifiOff,
  ChevronLeft,
  XCircle,
} from 'lucide-react'
import SettingsSegmentedControl from './SettingsSegmentedControl'
import {
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
    description: '仅对质量较差的页面执行 OCR',
    icon: FileSearch,
  },
  {
    value: 'always',
    label: '始终',
    description: '对所有页面执行 OCR 处理',
    icon: Eye,
  },
  {
    value: 'never',
    label: '关闭',
    description: '不执行任何 OCR 处理',
    icon: EyeOff,
  },
]

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
 * 面板分区导航定义
 */
const PANEL_TABS = [
  { id: 'local', label: '扫描 OCR', icon: ScanText },
  { id: 'cloud', label: '云端解析', icon: Globe },
  { id: 'figure', label: '图表定位', icon: Crop },
]

/**
 * 后端名称到中文显示名称的映射
 */
const BACKEND_LABELS = {
  tesseract: 'Tesseract',
  paddleocr: 'PaddleOCR',
  mistral: 'Mistral OCR',
  mineru: 'MinerU 深度解析',
  doc2x: 'Doc2X OCR',
}

/**
 * OCR 引擎选择选项定义
 * 每个选项包含值、标签和描述
 */
const BACKEND_OPTIONS = [
  {
    value: 'auto',
    label: '自动选择',
    description: '根据可用性自动选择最佳引擎',
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
  {
    value: 'mistral',
    label: 'Mistral OCR',
    description: '在线 OCR 服务，需配置 API Key',
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
const VALID_BACKENDS = ['auto', 'tesseract', 'paddleocr', 'mistral']

/**
 * 合法的主解析路线值
 */
const LEGACY_PAGE_OCR_BACKENDS = ['mineru', 'doc2x']

const DEFAULT_OCR_SETTINGS = {
  mode: 'auto',
  backend: 'auto',
  parseRoute: 'auto',
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
      // 旧版本曾把 MinerU/Doc2X 当作逐页 OCR；读取时统一迁回自动 OCR。
      const migratedLegacyPageOcr = LEGACY_PAGE_OCR_BACKENDS.includes(parsed.backend)
      // 校验上传主解析路线
      if (VALID_PARSE_ROUTES.includes(parsed.parseRoute)) {
        result.parseRoute = parsed.parseRoute
      }
      // 旧版图表字段继续透传，避免保存其他设置时破坏历史数据。
      if (typeof parsed.mineruFigureEnhance === 'boolean') {
        result.mineruFigureEnhance = parsed.mineruFigureEnhance
      }
      if (VALID_FIGURE_RENDER_MODES.includes(parsed.figureRenderMode)) {
        result.figureRenderMode = parsed.figureRenderMode
      }
      if (migratedLegacyPageOcr) {
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
    if (!VALID_PARSE_ROUTES.includes(nextSettings.parseRoute)) {
      nextSettings.parseRoute = 'auto'
    }
    localStorage.setItem(OCR_SETTINGS_KEY, JSON.stringify(nextSettings))
  } catch (err) {
    console.error('保存 OCR 设置失败:', err)
  }
}

const getApiErrorMessage = async (res, fallback) => {
  try {
    const data = await res.json()
    return data?.detail || data?.message || fallback
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
  // 面板分区导航：local=扫描 OCR，cloud=云端解析，figure=图表兜底资源
  const [activePanelTab, setActivePanelTab] = useState('local')
  // OCR 模式状态
  const [mode, setMode] = useState('auto')
  // OCR 引擎后端选择状态
  const [backend, setBackend] = useState('auto')
  // 上传时固定的主解析路线
  const [parseRoute, setParseRoute] = useState('auto')
  // 后端 OCR 状态数据
  const [ocrStatus, setOcrStatus] = useState(null)
  // 加载状态
  const [loading, setLoading] = useState(false)
  // 错误信息
  const [error, setError] = useState(null)

  // ---- 在线 OCR 配置状态 ----
  // Mistral API Key
  const [mistralApiKey, setMistralApiKey] = useState('')
  // Mistral Base URL
  const [mistralBaseUrl, setMistralBaseUrl] = useState('https://api.mistral.ai')
  // 是否显示 API Key 明文
  const [showApiKey, setShowApiKey] = useState(false)
  // 测试连接状态：null=未测试, 'loading'=测试中, 'success'=成功, 'error'=失败
  const [validateStatus, setValidateStatus] = useState(null)
  // 测试连接结果消息
  const [validateMessage, setValidateMessage] = useState('')
  // 保存配置中
  const [saving, setSaving] = useState(false)
  // 保存结果消息
  const [saveMessage, setSaveMessage] = useState('')
  // 已加载的在线 OCR 配置
  const [onlineConfig, setOnlineConfig] = useState(null)
  // Mistral 配置卡片展开/折叠状态（与 MinerU/Doc2X 卡片保持一致）
  const [mistralExpanded, setMistralExpanded] = useState(false)

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

  // ---- Doc2X OCR 配置状态 ----
  // Doc2X Worker URL
  const [doc2xWorkerUrl, setDoc2xWorkerUrl] = useState('')
  // Doc2X Auth Key
  const [doc2xAuthKey, setDoc2xAuthKey] = useState('')
  // Doc2X Token
  const [doc2xToken, setDoc2xToken] = useState('')
  // Doc2X Token 模式：'frontend'（前端透传）或 'worker'（Worker 配置）
  const [doc2xTokenMode, setDoc2xTokenMode] = useState('frontend')
  // 是否显示 Doc2X Auth Key 明文
  const [showDoc2xAuthKey, setShowDoc2xAuthKey] = useState(false)
  // 是否显示 Doc2X Token 明文
  const [showDoc2xToken, setShowDoc2xToken] = useState(false)
  // Doc2X 测试连接状态
  const [doc2xValidating, setDoc2xValidating] = useState(false)
  // Doc2X 测试连接结果
  const [doc2xValidateStatus, setDoc2xValidateStatus] = useState(null)
  const [doc2xValidateMessage, setDoc2xValidateMessage] = useState('')
  // Doc2X 保存状态
  const [doc2xSaving, setDoc2xSaving] = useState(false)
  const [doc2xSaveMessage, setDoc2xSaveMessage] = useState('')
  // Doc2X 配置卡片展开/折叠状态
  const [doc2xExpanded, setDoc2xExpanded] = useState(false)

  /**
   * 从 localStorage 加载已保存的设置（mode 和 backend）
   */
  useEffect(() => {
    if (isOpen) {
      const settings = loadOCRSettings()
      setMode(settings.mode)
      setBackend(settings.backend)
      setParseRoute(settings.parseRoute)
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
      // 如果已有配置，填充 Base URL（API Key 不回填，仅显示脱敏预览）
      if (data?.mistral) {
        if (data.mistral.base_url) {
          setMistralBaseUrl(data.mistral.base_url)
        }
      }
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
      }
      // 加载 Doc2X 已保存配置（回填非敏感字段）
      if (data?.doc2x) {
        if (data.doc2x.worker_url) {
          setDoc2xWorkerUrl(data.doc2x.worker_url)
        }
        if (data.doc2x.token_mode) {
          setDoc2xTokenMode(data.doc2x.token_mode)
        }
      }
    } catch (err) {
      console.error('获取在线 OCR 配置失败:', err)
    }
  }, [])

  /**
   * 测试连接：验证 API Key 有效性
   */
  const handleValidateKey = useCallback(async () => {
    if (!mistralApiKey.trim() && !onlineConfig?.mistral?.api_key_configured) return
    setValidateStatus('loading')
    setValidateMessage('')
    try {
      const res = await fetch(`${API_BASE_URL}/api/ocr/validate-key`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: 'mistral',
          api_key: mistralApiKey.trim(),
          base_url: mistralBaseUrl.trim() || 'https://api.mistral.ai',
        }),
      })
      const data = await res.json()
      // 处理 HTTP 错误（如 400）：FastAPI 返回 {"detail": "..."}
      if (!res.ok) {
        setValidateStatus('error')
        setValidateMessage(data.detail || `请求失败 (HTTP ${res.status})`)
        return
      }
      setValidateStatus(data.valid ? 'success' : 'error')
      setValidateMessage(data.message || (data.valid ? '验证成功' : '验证失败'))
    } catch (err) {
      console.error('验证 API Key 失败:', err)
      setValidateStatus('error')
      setValidateMessage('无法连接到服务器，请检查后端服务')
    }
  }, [mistralApiKey, mistralBaseUrl, onlineConfig?.mistral?.api_key_configured])

  /**
   * 保存在线 OCR 配置
   */
  const handleSaveOnlineConfig = useCallback(async () => {
    setSaving(true)
    setSaveMessage('')
    try {
      const body = {
        provider: 'mistral',
        api_key: mistralApiKey.trim(),
        base_url: mistralBaseUrl.trim() || 'https://api.mistral.ai',
      }
      const res = await fetch(`${API_BASE_URL}/api/ocr/online-config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      if (data.success) {
        setSaveMessage('配置已保存')
        // 重新加载配置和状态
        fetchOnlineConfig()
        fetchOCRStatus()
        // 清空输入的 API Key（已保存到后端）
        setMistralApiKey('')
        setValidateStatus(null)
        setValidateMessage('')
      } else {
        setSaveMessage(data.message || '保存失败')
      }
    } catch (err) {
      console.error('保存在线 OCR 配置失败:', err)
      setSaveMessage('保存失败，请检查后端服务')
    } finally {
      setSaving(false)
      // 3 秒后清除保存消息
      setTimeout(() => setSaveMessage(''), 3000)
    }
  }, [mistralApiKey, mistralBaseUrl, fetchOnlineConfig, fetchOCRStatus])

  /**
   * MinerU 测试连接：验证 Worker 可达性
   */
  const handleMineruValidate = useCallback(async () => {
    if (mineruAccessMode === 'worker' && !mineruWorkerUrl.trim()) return
    if (mineruAccessMode === 'direct' && !mineruToken.trim() && !onlineConfig?.mineru?.token_configured) return
    setMineruValidating(true)
    setMineruValidateStatus(null)
    setMineruValidateMessage('')
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
      const data = await res.json()
      // 处理 HTTP 错误（如 400）：FastAPI 返回 {"detail": "..."}
      if (!res.ok) {
        setMineruValidateStatus('error')
        setMineruValidateMessage(data.detail || `请求失败 (HTTP ${res.status})`)
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
          setMineruValidateMessage(saveData.detail || saveData.message || '连接成功，但配置保存失败')
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
      const data = await res.json()
      if (data.success) {
        setMineruSaveMessage('配置已保存')
        // 重新加载配置和状态
        fetchOnlineConfig()
        fetchOCRStatus()
        // 清空敏感输入（已保存到后端）
        setMineruAuthKey('')
        setMineruToken('')
        setMineruValidateStatus(null)
        setMineruValidateMessage('')
      } else {
        setMineruSaveMessage(data.message || '保存失败')
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
   * Doc2X 测试连接：验证 Worker 可达性
   */
  const handleDoc2xValidate = useCallback(async () => {
    if (!doc2xWorkerUrl.trim()) return
    setDoc2xValidating(true)
    setDoc2xValidateStatus(null)
    setDoc2xValidateMessage('')
    try {
      const res = await fetch(`${API_BASE_URL}/api/ocr/validate-key`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: 'doc2x',
          worker_url: doc2xWorkerUrl.trim(),
          auth_key: doc2xAuthKey.trim(),
          token: doc2xToken.trim(),
          token_mode: doc2xTokenMode,
        }),
      })
      const data = await res.json()
      // 处理 HTTP 错误（如 400）：FastAPI 返回 {"detail": "..."}
      if (!res.ok) {
        setDoc2xValidateStatus('error')
        setDoc2xValidateMessage(data.detail || `请求失败 (HTTP ${res.status})`)
        return
      }
      setDoc2xValidateStatus(data.valid ? 'success' : 'error')
      setDoc2xValidateMessage(data.message || (data.valid ? '连接成功' : '连接失败'))
    } catch (err) {
      console.error('Doc2X 测试连接失败:', err)
      setDoc2xValidateStatus('error')
      setDoc2xValidateMessage('无法连接到服务器，请检查后端服务')
    } finally {
      setDoc2xValidating(false)
    }
  }, [doc2xWorkerUrl, doc2xAuthKey])

  /**
   * 保存 Doc2X OCR 配置
   */
  const handleDoc2xSave = useCallback(async () => {
    setDoc2xSaving(true)
    setDoc2xSaveMessage('')
    try {
      const body = {
        provider: 'doc2x',
        worker_url: doc2xWorkerUrl.trim(),
        auth_key: doc2xAuthKey.trim(),
        token_mode: doc2xTokenMode,
        token: doc2xToken.trim(),
      }
      const res = await fetch(`${API_BASE_URL}/api/ocr/online-config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      if (data.success) {
        setDoc2xSaveMessage('配置已保存')
        // 重新加载配置和状态
        fetchOnlineConfig()
        fetchOCRStatus()
        // 清空敏感输入（已保存到后端）
        setDoc2xAuthKey('')
        setDoc2xToken('')
        setDoc2xValidateStatus(null)
        setDoc2xValidateMessage('')
      } else {
        setDoc2xSaveMessage(data.message || '保存失败')
      }
    } catch (err) {
      console.error('保存 Doc2X 配置失败:', err)
      setDoc2xSaveMessage('保存失败，请检查后端服务')
    } finally {
      setDoc2xSaving(false)
      // 3 秒后清除保存消息
      setTimeout(() => setDoc2xSaveMessage(''), 3000)
    }
  }, [doc2xWorkerUrl, doc2xAuthKey, doc2xTokenMode, doc2xToken, fetchOnlineConfig, fetchOCRStatus])

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

  return (
    <AnimatePresence initial={false}>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-slate-950/25 z-50 flex items-center justify-center p-4 transition-all opacity-100"
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.95, opacity: 0, y: 10 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30, mass: 0.8 }}
            className="settings-solid settings-shell w-full max-w-[640px] max-h-[92vh] bg-[#f6f7f9] border border-white/80 overflow-hidden flex flex-col"
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
                    解析设置
                  </div>
                  <div className="text-[12px] font-medium text-gray-500">
                    配置文档识别、OCR 与 MinerU 深度解析
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

              {/* 主解析与能力配置分区展示；图表兜底不是第三条主解析路线。 */}
              <div className="settings-segment relative flex items-center p-1 rounded-2xl">
                <motion.div
                  className="settings-segment-indicator absolute inset-y-1 left-1 rounded-xl"
                  initial={false}
                  animate={{ x: `${PANEL_TABS.findIndex((t) => t.id === activePanelTab) * 100}%` }}
                  transition={{ type: 'spring', stiffness: 420, damping: 32, mass: 0.7 }}
                  style={{ width: 'calc((100% - 0.5rem) / 3)' }}
                />
                {PANEL_TABS.map(({ id, label, icon: TabIcon }) => {
                  const isActive = activePanelTab === id
                  return (
                    <button
                      key={id}
                      type="button"
                      onClick={() => setActivePanelTab(id)}
                      className={`relative z-10 flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl text-[13px] font-semibold transition-colors duration-200 ${
                        isActive
                          ? 'text-[#ed8c68]'
                          : 'text-gray-500 hover:text-gray-700'
                      }`}
                    >
                      <motion.span
                        className="flex items-center gap-1.5"
                        animate={{ scale: isActive ? 1 : 0.97 }}
                        transition={{ duration: 0.15 }}
                      >
                        <TabIcon className="w-3.5 h-3.5" />
                        {label}
                      </motion.span>
                    </button>
                  )
                })}
              </div>

              {/* 上传前确定一份文档的主解析来源，避免正文、索引和阅读块混用不同路线。 */}
              <div className="settings-card bg-white p-5 border border-gray-200/90">
                <div className="flex items-start gap-2 mb-4">
                  <FileSearch className="w-4 h-4 text-gray-500 mt-0.5" />
                  <div>
                    <span className="text-sm font-semibold text-gray-800">上传解析路线</span>
                    <p className="text-[11px] text-gray-400 mt-0.5">
                      仅影响下一次上传；同一份文档的正文、阅读、大纲、总结、翻译、速览和问答索引会使用同一主路线。
                    </p>
                  </div>
                </div>

                <div className="space-y-2">
                  {PARSE_ROUTE_OPTIONS.map((option) => {
                    const Icon = option.icon
                    const isActive = parseRoute === option.value
                    return (
                      <button
                        key={option.value}
                        type="button"
                        aria-pressed={isActive}
                        onClick={() => handleParseRouteChange(option.value)}
                        className={`settings-inset w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-all ${
                          isActive
                            ? 'accent-surface'
                            : 'hover:border-[#FFDCCF] hover:bg-[#FFF4EF] text-gray-700'
                        }`}
                      >
                        <div
                          className={`p-1.5 rounded-lg ${
                            isActive ? 'bg-[#fcede8] text-[#ed8c68]' : 'bg-gray-100 text-gray-500'
                          }`}
                        >
                          <Icon className="w-4 h-4" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium">{option.label}</div>
                          <div className={`text-xs mt-0.5 ${isActive ? 'text-[#d2633b]' : 'text-gray-400'}`}>
                            {option.description}
                          </div>
                        </div>
                        {isActive && <CheckCircle2 className="w-5 h-5 flex-shrink-0" />}
                      </button>
                    )
                  })}
                </div>

                <div className="mt-3 flex items-start gap-2 px-3 py-2 rounded-xl bg-gray-50/80 border border-gray-100 text-[11px] text-gray-500">
                  <Crop className="w-3.5 h-3.5 mt-0.5 shrink-0 text-gray-400" />
                  <span>DocLayout-YOLO 只定位并裁切图表区域，不读取图片内容；MinerU 仍是独立的全程解析路线。</span>
                </div>
              </div>

              <AnimatePresence mode="wait">
              {activePanelTab === 'local' && (
              <motion.div
                key="tab-local-top"
                className="space-y-5"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.18, ease: 'easeOut' }}
              >
              {/* OCR 可用状态卡片 */}
              <div className="settings-card bg-white p-5 border border-gray-200/90">
                <div className="flex items-center gap-2 mb-4">
                  <Info className="w-4 h-4 text-gray-500" />
                  <span className="text-sm font-semibold text-gray-800">
                    OCR 引擎状态
                  </span>
                  {loading && (
                    <span className="text-xs text-gray-400 ml-auto">
                      加载中...
                    </span>
                  )}
                </div>

                {ocrStatus ? (
                  <div className="space-y-3">
                    {/* 总体可用性 */}
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-2.5 h-2.5 rounded-full ${
                          ocrStatus.available
                            ? 'bg-emerald-500'
                            : 'bg-gray-300'
                        }`}
                      />
                      <span className="text-sm text-gray-700">
                        OCR 服务：
                        {ocrStatus.available ? (
                          <span className="text-green-600 font-medium">
                            可用
                          </span>
                        ) : (
                          <span className="text-gray-500 font-medium">
                            不可用
                          </span>
                        )}
                      </span>
                      {ocrStatus.recommended && (
                        <span className="ml-auto text-xs text-[#ed8c68] bg-purple-50 px-2 py-0.5 rounded-full border border-purple-100">
                          推荐：{BACKEND_LABELS[ocrStatus.recommended] || ocrStatus.recommended}
                        </span>
                      )}
                    </div>

                    {/* 各后端可用性 */}
                    {ocrStatus.backends && (
                      <div className="grid grid-cols-2 gap-2">
                        {Object.entries(ocrStatus.backends).map(
                          ([name, available]) => (
                            <div
                              key={name}
                              className="flex items-center gap-2 px-3 py-2 rounded-xl bg-gray-50/80 border border-gray-100"
                            >
                              <div
                                className={`w-2 h-2 rounded-full ${
                                  available ? 'bg-emerald-500' : 'bg-gray-300'
                                }`}
                              />
                              <span className="text-sm text-gray-700">
                                {BACKEND_LABELS[name] || name}
                              </span>
                              {available ? (
                                <CheckCircle2 className="w-3.5 h-3.5 text-green-500 ml-auto" />
                              ) : (
                                <XCircle className="w-3.5 h-3.5 text-gray-300 ml-auto" />
                              )}
                            </div>
                          )
                        )}
                      </div>
                    )}

                    {/* Poppler 状态 */}
                    {ocrStatus.poppler_available !== undefined && (
                      <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-gray-50/80 border border-gray-100">
                        <div
                          className={`w-2 h-2 rounded-full ${
                            ocrStatus.poppler_available
                              ? 'bg-emerald-500'
                              : 'bg-amber-400'
                          }`}
                        />
                        <span className="text-sm text-gray-700">
                          Poppler (PDF 转图像)
                        </span>
                        {ocrStatus.poppler_available ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-green-500 ml-auto" />
                        ) : (
                          <AlertTriangle className="w-3.5 h-3.5 text-amber-500 ml-auto" />
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  !loading &&
                  !error && (
                    <div className="text-sm text-gray-400 text-center py-4">
                      暂无状态信息
                    </div>
                  )
                )}
              </div>

              {/* OCR 模式选择 */}
              <div className="settings-card bg-white p-5 border border-gray-200/90">
                <div className="flex items-center gap-2 mb-4">
                  <ScanText className="w-4 h-4 text-gray-500" />
                  <span className="text-sm font-semibold text-gray-800">
                    OCR 模式
                  </span>
                </div>

                <div className="space-y-2">
                  {OCR_MODES.map((option) => {
                    const Icon = option.icon
                    const isActive = mode === option.value
                    return (
                      <button
                        key={option.value}
                        onClick={() => handleModeChange(option.value)}
                        className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl border text-left transition-all ${
                          isActive
                            ? 'border-[#ed8c68]/30 bg-[#ed8c68]/5 text-[#ed8c68] shadow-sm'
                            : 'border-gray-100 hover:border-[#ed8c68]/30 hover:bg-[#ed8c68]/5 text-gray-700'
                        }`}
                      >
                        <div
                          className={`p-1.5 rounded-lg ${
                            isActive
                              ? 'bg-purple-100 text-[#ed8c68]'
                              : 'bg-gray-100 text-gray-500'
                          }`}
                        >
                          <Icon className="w-4 h-4" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium">
                            {option.label}
                          </div>
                          <div
                            className={`text-xs mt-0.5 ${
                              isActive ? 'text-[#ed8c68]' : 'text-gray-400'
                            }`}
                          >
                            {option.description}
                          </div>
                        </div>
                        {isActive && (
                          <CheckCircle2 className="w-5 h-5 text-[#ed8c68] flex-shrink-0" />
                        )}
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* OCR 引擎选择 */}
              <div className="settings-card bg-white p-5 border border-gray-200/90">
                <div className="flex items-center gap-2 mb-4">
                  <ScanText className="w-4 h-4 text-gray-500" />
                  <span className="text-sm font-semibold text-gray-800">
                    OCR 引擎选择
                  </span>
                </div>

                <div className="space-y-2">
                  {BACKEND_OPTIONS.map((option) => {
                    const isActive = backend === option.value
                    const localProviderUnavailable =
                      ['tesseract', 'paddleocr'].includes(option.value) &&
                      ocrStatus?.backends?.[option.value] === false
                    const disabled = Boolean(option.deprecatedForPageOcr || localProviderUnavailable)
                    return (
                      <button
                        key={option.value}
                        type="button"
                        disabled={disabled}
                        onClick={() => !disabled && handleBackendChange(option.value)}
                        className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl border text-left transition-all ${
                          isActive
                            ? 'border-[#ed8c68]/30 bg-[#ed8c68]/5 text-[#ed8c68] shadow-sm'
                            : disabled
                              ? 'border-gray-100 bg-gray-50 text-gray-400 cursor-not-allowed opacity-70'
                              : 'border-gray-100 hover:border-[#ed8c68]/30 hover:bg-[#ed8c68]/5 text-gray-700'
                        }`}
                      >
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium">
                            {option.label}
                          </div>
                          <div
                            className={`text-xs mt-0.5 ${
                              isActive ? 'text-[#ed8c68]' : 'text-gray-400'
                            }`}
                          >
                            {option.description}
                          </div>
                        </div>
                        {isActive && (
                          <CheckCircle2 className="w-5 h-5 text-[#ed8c68] flex-shrink-0" />
                        )}
                      </button>
                    )
                  })}
                </div>
              </div>
              </motion.div>
              )}
              </AnimatePresence>

              <AnimatePresence mode="wait">
              {activePanelTab === 'cloud' && (
              <motion.div
                key="tab-cloud"
                className="space-y-5"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.18, ease: 'easeOut' }}
              >
              {/* MinerU only performs document-level deep parsing. Page OCR uses
                  local engines or the explicitly selected Mistral provider. */}
              <div className="flex items-start gap-2 px-4 py-3 rounded-2xl bg-[#ed8c68]/5 border border-[#ed8c68]/15 text-xs text-[#d2633b]">
                <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                <span>下方配置仅用于 MinerU 深度解析，包括阅读结构、大纲与速览图表；扫描页 OCR 请使用本地引擎或 Mistral OCR。</span>
              </div>

              {/* 在线 OCR 服务配置卡片（可折叠，与 MinerU/Doc2X 卡片风格一致） */}
              <div className="settings-card bg-white p-5 border border-gray-200/90">
                <button
                  onClick={() => setMistralExpanded(!mistralExpanded)}
                  className="w-full flex items-center gap-2"
                >
                  <Wifi className="w-4 h-4 text-gray-500" />
                  <span className="text-sm font-semibold text-gray-800">
                    Mistral OCR 服务
                  </span>
                  {/* 已配置状态指示 */}
                  {onlineConfig?.mistral?.api_key_configured && (
                    <span className="ml-auto mr-2 text-xs text-green-600 bg-green-50 px-2 py-0.5 rounded-full border border-green-100 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      已配置
                    </span>
                  )}
                  <ChevronDown
                    className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${
                      mistralExpanded ? 'rotate-180' : ''
                    } ${onlineConfig?.mistral?.api_key_configured ? '' : 'ml-auto'}`}
                  />
                </button>

                {/* 已配置状态预览（折叠时显示） */}
                {!mistralExpanded && onlineConfig?.mistral?.api_key_configured && (
                  <div className="mt-3 space-y-1.5">
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-50/80 text-xs text-gray-600">
                      <Key className="w-3 h-3 text-gray-400" />
                      <span>API Key：</span>
                      <code className="font-mono text-gray-700">{onlineConfig.mistral.api_key_preview}</code>
                    </div>
                  </div>
                )}

                <AnimatePresence initial={false}>
                {mistralExpanded && (
                <motion.div
                  key="mistral-expanded"
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.22, ease: 'easeInOut' }}
                  style={{ overflow: 'hidden' }}
                >
                <div className="mt-4 space-y-4">
                  {/* 已有配置预览 */}
                  {onlineConfig?.mistral?.api_key_configured && (
                    <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-green-50/60 border border-green-100">
                      <Key className="w-3.5 h-3.5 text-green-600" />
                      <span className="text-xs text-green-700">
                        当前 API Key：
                        <code className="font-mono ml-1">
                          {onlineConfig.mistral.api_key_preview}
                        </code>
                      </span>
                    </div>
                  )}

                  {/* Mistral API Key 输入 */}
                  <div>
                    <label className="text-xs font-medium text-gray-600 mb-1.5 block">
                      Mistral OCR API Key
                    </label>
                    <div className="relative">
                      <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
                        <Key className="w-4 h-4" />
                      </div>
                      <input
                        type={showApiKey ? 'text' : 'password'}
                        value={mistralApiKey}
                        onChange={(e) => {
                          setMistralApiKey(e.target.value)
                          // 输入变化时重置验证状态
                          setValidateStatus(null)
                          setValidateMessage('')
                        }}
                        placeholder={
                          onlineConfig?.mistral?.api_key_configured
                            ? '输入新 Key 以更新（留空保持不变）'
                            : '输入 Mistral API Key'
                        }
                        className="w-full pl-10 pr-10 py-2.5 text-sm rounded-xl border border-gray-200 bg-white/60 focus:border-purple-300 focus:ring-2 focus:ring-purple-100 outline-none transition-all placeholder:text-gray-300"
                      />
                      {/* 显示/隐藏 API Key 切换按钮 */}
                      <button
                        type="button"
                        onClick={() => setShowApiKey(!showApiKey)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                      >
                        {showApiKey ? (
                          <EyeOff className="w-4 h-4" />
                        ) : (
                          <Eye className="w-4 h-4" />
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Mistral Base URL 输入 */}
                  <div>
                    <label className="text-xs font-medium text-gray-600 mb-1.5 block">
                      Base URL
                    </label>
                    <div className="relative">
                      <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
                        <Globe className="w-4 h-4" />
                      </div>
                      <input
                        type="text"
                        value={mistralBaseUrl}
                        onChange={(e) => setMistralBaseUrl(e.target.value)}
                        placeholder="https://api.mistral.ai"
                        className="w-full pl-10 pr-4 py-2.5 text-sm rounded-xl border border-gray-200 bg-white/60 focus:border-purple-300 focus:ring-2 focus:ring-purple-100 outline-none transition-all placeholder:text-gray-300"
                      />
                    </div>
                  </div>

                  {/* 测试连接结果 */}
                  {validateStatus && validateStatus !== 'loading' && (
                    <div
                      className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs ${
                        validateStatus === 'success'
                          ? 'bg-green-50/60 border border-green-100 text-green-700'
                          : 'bg-red-50/60 border border-red-100 text-red-700'
                      }`}
                    >
                      {validateStatus === 'success' ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
                      ) : (
                        <XCircle className="w-3.5 h-3.5 text-red-500" />
                      )}
                      <span>{validateMessage}</span>
                    </div>
                  )}

                  {/* 保存结果消息 */}
                  {saveMessage && (
                    <div
                      className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs ${
                        saveMessage === '配置已保存'
                          ? 'bg-green-50/60 border border-green-100 text-green-700'
                          : 'bg-red-50/60 border border-red-100 text-red-700'
                      }`}
                    >
                      {saveMessage === '配置已保存' ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
                      ) : (
                        <XCircle className="w-3.5 h-3.5 text-red-500" />
                      )}
                      <span>{saveMessage}</span>
                    </div>
                  )}

                  {/* 操作按钮区域 */}
                  <div className="flex items-center gap-2">
                    {/* 测试连接按钮 */}
                    <button
                      onClick={handleValidateKey}
                      disabled={(!mistralApiKey.trim() && !onlineConfig?.mistral?.api_key_configured) || validateStatus === 'loading'}
                      className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium rounded-xl border border-[#ed8c68]/30 bg-[#ed8c68]/5 text-[#ed8c68] hover:bg-[#ed8c68]/10 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                    >
                      {validateStatus === 'loading' ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <Wifi className="w-3.5 h-3.5" />
                      )}
                      测试连接
                    </button>

                    {/* 保存配置按钮 */}
                    <button
                      onClick={handleSaveOnlineConfig}
                      disabled={saving || (!mistralApiKey.trim() && !onlineConfig?.mistral?.api_key_configured)}
                      className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium rounded-xl border border-green-200 bg-green-50/60 text-green-700 hover:bg-green-100/60 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                    >
                      {saving ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <Save className="w-3.5 h-3.5" />
                      )}
                      保存配置
                    </button>
                  </div>
                </div>
                </motion.div>
                )}
                </AnimatePresence>
              </div>

              {/* MinerU OCR 配置卡片（可折叠） */}
              <div className="settings-card bg-white p-5 border border-gray-200/90">
                {/* 卡片标题栏（点击展开/折叠） */}
                <button
                  onClick={() => setMineruExpanded(!mineruExpanded)}
                  className="w-full flex items-center gap-2"
                >
                  <Globe className="w-4 h-4 text-gray-500" />
                  <span className="text-sm font-semibold text-gray-800">
                    MinerU 深度解析服务
                  </span>
                  {/* 已配置状态指示 */}
                  {(onlineConfig?.mineru?.worker_url || onlineConfig?.mineru?.token_configured) && (
                    <span className="ml-auto mr-2 text-xs text-green-600 bg-green-50 px-2 py-0.5 rounded-full border border-green-100 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      已配置
                    </span>
                  )}
                  <ChevronDown
                    className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${
                      mineruExpanded ? 'rotate-180' : ''
                    } ${(onlineConfig?.mineru?.worker_url || onlineConfig?.mineru?.token_configured) ? '' : 'ml-auto'}`}
                  />
                </button>

                {/* 已配置状态预览（折叠时显示） */}
                {!mineruExpanded && (onlineConfig?.mineru?.worker_url || onlineConfig?.mineru?.token_configured) && (
                  <div className="mt-3 space-y-1.5">
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-50/80 text-xs text-gray-600">
                      <Globe className="w-3 h-3 text-gray-400" />
                      <span>模式：</span>
                      <code className="font-mono text-gray-700">{onlineConfig.mineru.access_mode === 'direct' ? '直连 MinerU API' : 'Worker 代理'}</code>
                    </div>
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-50/80 text-xs text-gray-600">
                      <Globe className="w-3 h-3 text-gray-400" />
                      <span>{onlineConfig.mineru.access_mode === 'direct' ? 'Base URL：' : 'Worker URL：'}</span>
                      <code className="font-mono text-gray-700">{onlineConfig.mineru.access_mode === 'direct' ? onlineConfig.mineru.base_url : onlineConfig.mineru.worker_url}</code>
                    </div>
                    {onlineConfig.mineru.auth_key_configured && (
                      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-50/80 text-xs text-gray-600">
                        <Key className="w-3 h-3 text-gray-400" />
                        <span>Auth Key：</span>
                        <code className="font-mono text-gray-700">{onlineConfig.mineru.auth_key_preview}</code>
                      </div>
                    )}
                    {onlineConfig.mineru.token_configured && (
                      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-50/80 text-xs text-gray-600">
                        <Key className="w-3 h-3 text-gray-400" />
                        <span>Token：</span>
                        <code className="font-mono text-gray-700">{onlineConfig.mineru.token_preview}</code>
                        <span className="text-gray-400 ml-1">
                          ({onlineConfig.mineru.token_mode === 'worker' ? 'Worker 配置' : '前端透传'})
                        </span>
                      </div>
                    )}
                  </div>
                )}

                {/* 展开的配置表单 */}
                <AnimatePresence initial={false}>
                {mineruExpanded && (
                  <motion.div
                    key="mineru-expanded"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.22, ease: 'easeInOut' }}
                    style={{ overflow: 'hidden' }}
                  >
                  <div className="mt-4 space-y-4">
                    {/* 已有配置预览 */}
                    {(onlineConfig?.mineru?.worker_url || onlineConfig?.mineru?.token_configured) && (
                      <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-green-50/60 border border-green-100">
                        <Globe className="w-3.5 h-3.5 text-green-600" />
                        <span className="text-xs text-green-700">
                          当前模式：
                          <code className="font-mono ml-1">
                            {onlineConfig.mineru.access_mode === 'direct' ? '直连 MinerU API' : 'Worker 代理'}
                          </code>
                          {onlineConfig.mineru.access_mode === 'direct' ? ' · Base URL：' : ' · Worker URL：'}
                          <code className="font-mono ml-1">
                            {onlineConfig.mineru.access_mode === 'direct' ? onlineConfig.mineru.base_url : onlineConfig.mineru.worker_url}
                          </code>
                        </span>
                      </div>
                    )}

                    {/* 接入模式选择 */}
                    <div>
                      <label className="text-xs font-medium text-gray-600 mb-1.5 block">
                        接入模式
                      </label>
                      <SettingsSegmentedControl
                        ariaLabel="MinerU 接入模式"
                        value={mineruAccessMode}
                        onChange={setMineruAccessMode}
                        options={[
                          { value: 'worker', label: 'Worker 代理' },
                          { value: 'direct', label: '直连 API' },
                        ]}
                        buttonClassName="px-3 py-2 text-xs font-medium text-center rounded-[10px]"
                      />
                      <div className="text-xs text-gray-400 mt-1">
                        {mineruAccessMode === 'worker'
                          ? '通过你部署的 pb-ocr-proxy 转发 MinerU 请求'
                          : '后端直接调用 MinerU 官方 API，仍会上传当前 PDF 到 MinerU'}
                      </div>
                    </div>

                    {mineruAccessMode === 'direct' && (
                      <div>
                        <label className="text-xs font-medium text-gray-600 mb-1.5 block">
                          MinerU API Base URL
                        </label>
                        <div className="relative">
                          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
                            <Globe className="w-4 h-4" />
                          </div>
                          <input
                            type="text"
                            value={mineruBaseUrl}
                            onChange={(e) => setMineruBaseUrl(e.target.value)}
                            placeholder="https://mineru.net/api/v4"
                            className="w-full pl-10 pr-4 py-2.5 text-sm rounded-xl border border-gray-200 bg-white/60 focus:border-purple-300 focus:ring-2 focus:ring-purple-100 outline-none transition-all placeholder:text-gray-300"
                          />
                        </div>
                      </div>
                    )}

                    {/* Worker URL 输入框 */}
                    {mineruAccessMode === 'worker' && (
                    <div>
                      <label className="text-xs font-medium text-gray-600 mb-1.5 block">
                        Worker URL
                      </label>
                      <div className="relative">
                        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
                          <Globe className="w-4 h-4" />
                        </div>
                        <input
                          type="text"
                          value={mineruWorkerUrl}
                          onChange={(e) => setMineruWorkerUrl(e.target.value)}
                          placeholder="https://your-worker.workers.dev"
                          className="w-full pl-10 pr-4 py-2.5 text-sm rounded-xl border border-gray-200 bg-white/60 focus:border-purple-300 focus:ring-2 focus:ring-purple-100 outline-none transition-all placeholder:text-gray-300"
                        />
                      </div>
                    </div>
                    )}

                    {/* Auth Key 输入框（可选，带显示/隐藏切换） */}
                    {mineruAccessMode === 'worker' && (
                    <div>
                      <label className="text-xs font-medium text-gray-600 mb-1.5 block">
                        Auth Key（可选）
                      </label>
                      <div className="relative">
                        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
                          <Key className="w-4 h-4" />
                        </div>
                        <input
                          type={showMineruAuthKey ? 'text' : 'password'}
                          value={mineruAuthKey}
                          onChange={(e) => {
                            setMineruAuthKey(e.target.value)
                            setMineruValidateStatus(null)
                            setMineruValidateMessage('')
                          }}
                          placeholder={
                            onlineConfig?.mineru?.auth_key_configured
                              ? '输入新 Auth Key 以更新（留空保持不变）'
                              : '如果 Worker 启用了访问控制，填写这里'
                          }
                          className="w-full pl-10 pr-10 py-2.5 text-sm rounded-xl border border-gray-200 bg-white/60 focus:border-purple-300 focus:ring-2 focus:ring-purple-100 outline-none transition-all placeholder:text-gray-300"
                        />
                        <button
                          type="button"
                          onClick={() => setShowMineruAuthKey(!showMineruAuthKey)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                        >
                          {showMineruAuthKey ? (
                            <EyeOff className="w-4 h-4" />
                          ) : (
                            <Eye className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </div>
                    )}

                    {/* Token Mode 选择 */}
                    {mineruAccessMode === 'worker' && (
                    <div>
                      <label className="text-xs font-medium text-gray-600 mb-1.5 block">
                        Token 模式
                      </label>
                      <SettingsSegmentedControl
                        ariaLabel="MinerU Token 模式"
                        value={mineruTokenMode}
                        onChange={setMineruTokenMode}
                        options={[
                          { value: 'frontend', label: '前端透传' },
                          { value: 'worker', label: 'Worker 配置' },
                        ]}
                        buttonClassName="px-3 py-2 text-xs font-medium text-center rounded-[10px]"
                      />
                      <div className="text-xs text-gray-400 mt-1">
                        {mineruTokenMode === 'frontend'
                          ? '由前端传递 Token 到 Worker'
                          : 'Token 在 Worker 环境变量中配置，无需前端提供'}
                      </div>
                    </div>
                    )}

                    {/* Token 输入框（仅 frontend 模式显示，带显示/隐藏切换） */}
                    {(mineruAccessMode === 'direct' || mineruTokenMode === 'frontend') && (
                      <div>
                        <label className="text-xs font-medium text-gray-600 mb-1.5 block">
                          MinerU Token
                        </label>
                        <div className="relative">
                          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
                            <Key className="w-4 h-4" />
                          </div>
                          <input
                            type={showMineruToken ? 'text' : 'password'}
                            value={mineruToken}
                            onChange={(e) => setMineruToken(e.target.value)}
                            placeholder={
                              onlineConfig?.mineru?.token_configured
                                ? '输入新 Token 以更新（留空保持不变）'
                                : '输入 MinerU API Token'
                            }
                            className="w-full pl-10 pr-10 py-2.5 text-sm rounded-xl border border-gray-200 bg-white/60 focus:border-purple-300 focus:ring-2 focus:ring-purple-100 outline-none transition-all placeholder:text-gray-300"
                          />
                          <button
                            type="button"
                            onClick={() => setShowMineruToken(!showMineruToken)}
                            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                          >
                            {showMineruToken ? (
                              <EyeOff className="w-4 h-4" />
                            ) : (
                              <Eye className="w-4 h-4" />
                            )}
                          </button>
                        </div>
                      </div>
                    )}

                    {/* OCR 选项开关 */}
                    <div>
                      <label className="text-xs font-medium text-gray-600 mb-2 block">
                        OCR 处理选项
                      </label>
                      <div className="space-y-2">
                    {/* 深度解析中的扫描件识别 */}
                    <label className="flex items-center justify-between px-3 py-2 rounded-xl bg-gray-50/80 border border-gray-100 cursor-pointer">
                      <span>
                        <span className="block text-xs text-gray-700">扫描件 OCR</span>
                        <span className="block text-[10px] text-gray-400 mt-0.5">仅影响 MinerU 深度解析；普通论文建议关闭，扫描 PDF 再开启</span>
                          </span>
                          <div
                            onClick={() => setMineruEnableOcr(!mineruEnableOcr)}
                            className={`relative w-[42px] h-[24px] rounded-full transition-colors duration-200 outline-none flex-shrink-0 cursor-pointer ${
                              mineruEnableOcr ? 'accent-control' : 'bg-gray-300'
                            }`}
                          >
                            <div
                              className={`absolute top-[2px] left-[2px] w-[20px] h-[20px] rounded-full bg-white shadow-sm transition-transform ${
                                mineruEnableOcr ? 'translate-x-[18px]' : 'translate-x-0'
                              }`}
                            />
                          </div>
                        </label>
                        {/* 启用公式识别 */}
                        <label className="flex items-center justify-between px-3 py-2 rounded-xl bg-gray-50/80 border border-gray-100 cursor-pointer">
                          <span className="text-xs text-gray-700">启用公式识别</span>
                          <div
                            onClick={() => setMineruEnableFormula(!mineruEnableFormula)}
                            className={`relative w-[42px] h-[24px] rounded-full transition-colors duration-200 outline-none flex-shrink-0 cursor-pointer ${
                              mineruEnableFormula ? 'accent-control' : 'bg-gray-300'
                            }`}
                          >
                            <div
                              className={`absolute top-[2px] left-[2px] w-[20px] h-[20px] rounded-full bg-white shadow-sm transition-transform ${
                                mineruEnableFormula ? 'translate-x-[18px]' : 'translate-x-0'
                              }`}
                            />
                          </div>
                        </label>
                        {/* 启用表格识别 */}
                        <label className="flex items-center justify-between px-3 py-2 rounded-xl bg-gray-50/80 border border-gray-100 cursor-pointer">
                          <span className="text-xs text-gray-700">启用表格识别</span>
                          <div
                            onClick={() => setMineruEnableTable(!mineruEnableTable)}
                            className={`relative w-[42px] h-[24px] rounded-full transition-colors duration-200 outline-none flex-shrink-0 cursor-pointer ${
                              mineruEnableTable ? 'accent-control' : 'bg-gray-300'
                            }`}
                          >
                            <div
                              className={`absolute top-[2px] left-[2px] w-[20px] h-[20px] rounded-full bg-white shadow-sm transition-transform ${
                                mineruEnableTable ? 'translate-x-[18px]' : 'translate-x-0'
                              }`}
                            />
                          </div>
                        </label>
                        <div className="px-3 py-2 rounded-xl bg-gray-50/80 border border-gray-100">
                          <div className="flex items-center justify-between gap-3">
                            <span>
                              <span className="block text-xs text-gray-700">解析模型</span>
                              <span className="block text-[10px] text-gray-400 mt-0.5">VLM 适合复杂版式和表格；Pipeline 可作为兼容回退</span>
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
                    </div>

                    {/* MinerU 测试连接结果 */}
                    {mineruValidateStatus && (
                      <div
                        className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs ${
                          mineruValidateStatus === 'success'
                            ? 'bg-green-50/60 border border-green-100 text-green-700'
                            : 'bg-red-50/60 border border-red-100 text-red-700'
                        }`}
                      >
                        {mineruValidateStatus === 'success' ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
                        ) : (
                          <XCircle className="w-3.5 h-3.5 text-red-500" />
                        )}
                        <span>{mineruValidateMessage}</span>
                      </div>
                    )}

                    {/* MinerU 保存结果消息 */}
                    {mineruSaveMessage && (
                      <div
                        className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs ${
                          mineruSaveMessage === '配置已保存'
                            ? 'bg-green-50/60 border border-green-100 text-green-700'
                            : 'bg-red-50/60 border border-red-100 text-red-700'
                        }`}
                      >
                        {mineruSaveMessage === '配置已保存' ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
                        ) : (
                          <XCircle className="w-3.5 h-3.5 text-red-500" />
                        )}
                        <span>{mineruSaveMessage}</span>
                      </div>
                    )}

                    {/* MinerU 操作按钮区域 */}
                    <div className="flex items-center gap-2">
                      {/* 测试连接按钮 */}
                      <button
                        onClick={handleMineruValidate}
                        disabled={
                          mineruValidating
                          || (mineruAccessMode === 'worker' && !mineruWorkerUrl.trim())
                          || (mineruAccessMode === 'direct' && !mineruToken.trim() && !onlineConfig?.mineru?.token_configured)
                        }
                        className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium rounded-xl border border-[#ed8c68]/30 bg-[#ed8c68]/5 text-[#ed8c68] hover:bg-[#ed8c68]/10 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                      >
                        {mineruValidating ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <Wifi className="w-3.5 h-3.5" />
                        )}
                        测试连接
                      </button>

                      {/* 保存配置按钮 */}
                      <button
                        onClick={handleMineruSave}
                        disabled={
                          mineruSaving
                          || (mineruAccessMode === 'worker' && !mineruWorkerUrl.trim())
                          || (mineruAccessMode === 'direct' && !mineruToken.trim() && !onlineConfig?.mineru?.token_configured)
                        }
                        className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium rounded-xl border border-green-200 bg-green-50/60 text-green-700 hover:bg-green-100/60 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                      >
                        {mineruSaving ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <Save className="w-3.5 h-3.5" />
                        )}
                        保存配置
                      </button>
                    </div>
                  </div>
                  </motion.div>
                )}
                </AnimatePresence>
              </div>

              {/* Doc2X OCR 配置卡片（可折叠） */}
              {false && (
              <div className="settings-card bg-white p-5 border border-gray-200/90">
                {/* 卡片标题栏（点击展开/折叠） */}
                <button
                  onClick={() => setDoc2xExpanded(!doc2xExpanded)}
                  className="w-full flex items-center gap-2"
                >
                  <Globe className="w-4 h-4 text-gray-500" />
                  <span className="text-sm font-semibold text-gray-800">
                    Doc2X OCR 服务
                  </span>
                  {/* 已配置状态指示 */}
                  {onlineConfig?.doc2x?.worker_url && (
                    <span className="ml-auto mr-2 text-xs text-green-600 bg-green-50 px-2 py-0.5 rounded-full border border-green-100 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      已配置
                    </span>
                  )}
                  <ChevronDown
                    className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${
                      doc2xExpanded ? 'rotate-180' : ''
                    } ${onlineConfig?.doc2x?.worker_url ? '' : 'ml-auto'}`}
                  />
                </button>

                {/* 已配置状态预览（折叠时显示） */}
                {!doc2xExpanded && onlineConfig?.doc2x?.worker_url && (
                  <div className="mt-3 space-y-1.5">
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-50/80 text-xs text-gray-600">
                      <Globe className="w-3 h-3 text-gray-400" />
                      <span>Worker URL：</span>
                      <code className="font-mono text-gray-700">{onlineConfig.doc2x.worker_url}</code>
                    </div>
                    {onlineConfig.doc2x.auth_key_configured && (
                      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-50/80 text-xs text-gray-600">
                        <Key className="w-3 h-3 text-gray-400" />
                        <span>Auth Key：</span>
                        <code className="font-mono text-gray-700">{onlineConfig.doc2x.auth_key_preview}</code>
                      </div>
                    )}
                    {onlineConfig.doc2x.token_configured && (
                      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-50/80 text-xs text-gray-600">
                        <Key className="w-3 h-3 text-gray-400" />
                        <span>Token：</span>
                        <code className="font-mono text-gray-700">{onlineConfig.doc2x.token_preview}</code>
                        <span className="text-gray-400 ml-1">
                          ({onlineConfig.doc2x.token_mode === 'worker' ? 'Worker 配置' : '前端透传'})
                        </span>
                      </div>
                    )}
                  </div>
                )}

                {/* 展开的配置表单 */}
                <AnimatePresence initial={false}>
                {doc2xExpanded && (
                  <motion.div
                    key="doc2x-expanded"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.22, ease: 'easeInOut' }}
                    style={{ overflow: 'hidden' }}
                  >
                  <div className="mt-4 space-y-4">
                    {/* 已有配置预览 */}
                    {onlineConfig?.doc2x?.worker_url && (
                      <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-green-50/60 border border-green-100">
                        <Globe className="w-3.5 h-3.5 text-green-600" />
                        <span className="text-xs text-green-700">
                          当前 Worker URL：
                          <code className="font-mono ml-1">
                            {onlineConfig.doc2x.worker_url}
                          </code>
                        </span>
                      </div>
                    )}

                    {/* Worker URL 输入框 */}
                    <div>
                      <label className="text-xs font-medium text-gray-600 mb-1.5 block">
                        Worker URL
                      </label>
                      <div className="relative">
                        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
                          <Globe className="w-4 h-4" />
                        </div>
                        <input
                          type="text"
                          value={doc2xWorkerUrl}
                          onChange={(e) => setDoc2xWorkerUrl(e.target.value)}
                          placeholder="https://your-worker.workers.dev"
                          className="w-full pl-10 pr-4 py-2.5 text-sm rounded-xl border border-gray-200 bg-white/60 focus:border-purple-300 focus:ring-2 focus:ring-purple-100 outline-none transition-all placeholder:text-gray-300"
                        />
                      </div>
                    </div>

                    {/* Auth Key 输入框（可选，带显示/隐藏切换） */}
                    <div>
                      <label className="text-xs font-medium text-gray-600 mb-1.5 block">
                        Auth Key（可选）
                      </label>
                      <div className="relative">
                        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
                          <Key className="w-4 h-4" />
                        </div>
                        <input
                          type={showDoc2xAuthKey ? 'text' : 'password'}
                          value={doc2xAuthKey}
                          onChange={(e) => {
                            setDoc2xAuthKey(e.target.value)
                            setDoc2xValidateStatus(null)
                            setDoc2xValidateMessage('')
                          }}
                          placeholder={
                            onlineConfig?.doc2x?.auth_key_configured
                              ? '输入新 Auth Key 以更新（留空保持不变）'
                              : '如果 Worker 启用了访问控制，填写这里'
                          }
                          className="w-full pl-10 pr-10 py-2.5 text-sm rounded-xl border border-gray-200 bg-white/60 focus:border-purple-300 focus:ring-2 focus:ring-purple-100 outline-none transition-all placeholder:text-gray-300"
                        />
                        <button
                          type="button"
                          onClick={() => setShowDoc2xAuthKey(!showDoc2xAuthKey)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                        >
                          {showDoc2xAuthKey ? (
                            <EyeOff className="w-4 h-4" />
                          ) : (
                            <Eye className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </div>

                    {/* Token Mode 选择 */}
                    <div>
                      <label className="text-xs font-medium text-gray-600 mb-1.5 block">
                        Token 模式
                      </label>
                      <SettingsSegmentedControl
                        ariaLabel="Doc2X Token 模式"
                        value={doc2xTokenMode}
                        onChange={setDoc2xTokenMode}
                        options={[
                          { value: 'frontend', label: '前端透传' },
                          { value: 'worker', label: 'Worker 配置' },
                        ]}
                        buttonClassName="px-3 py-2 text-xs font-medium text-center rounded-[10px]"
                      />
                      <div className="text-xs text-gray-400 mt-1">
                        {doc2xTokenMode === 'frontend'
                          ? '由前端传递 Token 到 Worker'
                          : 'Token 在 Worker 环境变量中配置，无需前端提供'}
                      </div>
                    </div>

                    {/* Token 输入框（仅 frontend 模式显示，带显示/隐藏切换） */}
                    {doc2xTokenMode === 'frontend' && (
                      <div>
                        <label className="text-xs font-medium text-gray-600 mb-1.5 block">
                          Doc2X Token
                        </label>
                        <div className="relative">
                          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
                            <Key className="w-4 h-4" />
                          </div>
                          <input
                            type={showDoc2xToken ? 'text' : 'password'}
                            value={doc2xToken}
                            onChange={(e) => setDoc2xToken(e.target.value)}
                            placeholder={
                              onlineConfig?.doc2x?.token_configured
                                ? '输入新 Token 以更新（留空保持不变）'
                                : '输入 Doc2X API Token'
                            }
                            className="w-full pl-10 pr-10 py-2.5 text-sm rounded-xl border border-gray-200 bg-white/60 focus:border-purple-300 focus:ring-2 focus:ring-purple-100 outline-none transition-all placeholder:text-gray-300"
                          />
                          <button
                            type="button"
                            onClick={() => setShowDoc2xToken(!showDoc2xToken)}
                            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                          >
                            {showDoc2xToken ? (
                              <EyeOff className="w-4 h-4" />
                            ) : (
                              <Eye className="w-4 h-4" />
                            )}
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Doc2X 测试连接结果 */}
                    {doc2xValidateStatus && (
                      <div
                        className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs ${
                          doc2xValidateStatus === 'success'
                            ? 'bg-green-50/60 border border-green-100 text-green-700'
                            : 'bg-red-50/60 border border-red-100 text-red-700'
                        }`}
                      >
                        {doc2xValidateStatus === 'success' ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
                        ) : (
                          <XCircle className="w-3.5 h-3.5 text-red-500" />
                        )}
                        <span>{doc2xValidateMessage}</span>
                      </div>
                    )}

                    {/* Doc2X 保存结果消息 */}
                    {doc2xSaveMessage && (
                      <div
                        className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs ${
                          doc2xSaveMessage === '配置已保存'
                            ? 'bg-green-50/60 border border-green-100 text-green-700'
                            : 'bg-red-50/60 border border-red-100 text-red-700'
                        }`}
                      >
                        {doc2xSaveMessage === '配置已保存' ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
                        ) : (
                          <XCircle className="w-3.5 h-3.5 text-red-500" />
                        )}
                        <span>{doc2xSaveMessage}</span>
                      </div>
                    )}

                    {/* Doc2X 操作按钮区域 */}
                    <div className="flex items-center gap-2">
                      {/* 测试连接按钮 */}
                      <button
                        onClick={handleDoc2xValidate}
                        disabled={!doc2xWorkerUrl.trim() || doc2xValidating}
                        className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium rounded-xl border border-[#ed8c68]/30 bg-[#ed8c68]/5 text-[#ed8c68] hover:bg-[#ed8c68]/10 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                      >
                        {doc2xValidating ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <Wifi className="w-3.5 h-3.5" />
                        )}
                        测试连接
                      </button>

                      {/* 保存配置按钮 */}
                      <button
                        onClick={handleDoc2xSave}
                        disabled={doc2xSaving || !doc2xWorkerUrl.trim()}
                        className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium rounded-xl border border-green-200 bg-green-50/60 text-green-700 hover:bg-green-100/60 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                      >
                        {doc2xSaving ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <Save className="w-3.5 h-3.5" />
                        )}
                        保存配置
                      </button>
                    </div>
                  </div>
                  </motion.div>
                )}
                </AnimatePresence>
              </div>
              )}
              </motion.div>
              )}
              </AnimatePresence>

              <AnimatePresence mode="wait">
              {activePanelTab === 'figure' && (
              <motion.div
                key="tab-figure"
                className="space-y-5"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.18, ease: 'easeOut' }}
              >
              {/* 图表定位遵循主解析路线，视觉模型只在结构结果缺失时兜底。 */}
              <div className="settings-card bg-white p-5 border border-gray-200/90">
                <div className="flex items-center gap-2 mb-4">
                  <Crop className="w-4 h-4 text-gray-500" />
                  <span className="text-sm font-semibold text-gray-800">
                    图表兜底
                  </span>
                </div>

                <div className="rounded-[12px] border border-[#eadfd9] bg-[#faf7f5] px-4 py-3">
                  <div className="flex items-start gap-2.5">
                    <Info className="mt-0.5 h-4 w-4 shrink-0 text-[#b85f47]" />
                    <div>
                      <div className="text-xs font-semibold text-gray-700">自动跟随当前文档的主解析路线</div>
                      <p className="mt-1 text-[11px] leading-5 text-gray-500">
                        MinerU 文档优先使用结构化图表，本地文档优先使用 PDF 原生结构；只有主结果未定位到图表时，才调用本地图表定位兜底。
                      </p>
                    </div>
                  </div>
                </div>

                <div className="mt-4 px-1">
                  <div className="mt-3 rounded-2xl border border-gray-100 bg-gray-50/70 p-3 space-y-3">
                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-medium text-gray-700">本地图表定位模型</span>
                          <span
                            className={`text-[10px] px-1.5 py-0.5 rounded-full border ${
                              yoloReady
                                ? 'bg-green-50 border-green-100 text-green-700'
                                : yoloDependencyMissing
                                  ? 'bg-red-50 border-red-100 text-red-700'
                                  : 'bg-amber-50 border-amber-100 text-amber-700'
                            }`}
                          >
                            {yoloStatusLabel}
                          </span>
                        </div>
                        <p className="text-[11px] text-gray-400 mt-0.5">
                          DocLayout-YOLO 仅负责版面定位和裁切；图表内容理解在设置中心的「阅读」中单独选择
                        </p>
                      </div>
                      <button
                        onClick={fetchYoloStatus}
                        disabled={yoloBusy}
                        className="shrink-0 inline-flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-medium rounded-lg border border-gray-200 bg-white text-gray-600 hover:bg-gray-100 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                      >
                        <RefreshCw className={`w-3 h-3 ${yoloBusy ? 'animate-spin' : ''}`} />
                        刷新
                      </button>
                    </div>

                    {yoloStatus?.model_path && (
                      <div className="min-w-0 rounded-lg bg-white/80 border border-gray-100 px-3 py-2">
                        <div className="text-[10px] text-gray-400 mb-1">当前权重路径</div>
                        <div className="font-mono text-[11px] text-gray-600 break-all">{yoloStatus.model_path}</div>
                      </div>
                    )}

                    {yoloDependencyMissing && (
                      <div className="flex items-start gap-2 px-3 py-2 rounded-xl bg-red-50/70 border border-red-100 text-xs text-red-700">
                        <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                        <span>当前后端缺少 YOLO 运行依赖。桌面完整包会保留运行库，但权重需在软件内下载或指定。</span>
                      </div>
                    )}

                    <div className="space-y-2">
                      <label className="block text-[11px] font-medium text-gray-500">一键下载到目录</label>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={yoloInstallDir}
                          onChange={(e) => setYoloInstallDir(e.target.value)}
                          placeholder={yoloStatus?.default_install_dir || '默认用户数据目录'}
                          className="flex-1 min-w-0 px-3 py-2 rounded-xl border border-gray-200 bg-white text-xs text-gray-700 focus:outline-none focus:border-[#ed8c68]/50"
                        />
                        {window.chatpdfDesktop?.selectDirectory && (
                          <button
                            onClick={handleSelectYoloInstallDir}
                            disabled={yoloBusy}
                            className="shrink-0 inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-xl border border-gray-200 bg-white text-gray-600 hover:bg-gray-100 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                          >
                            <FolderOpen className="w-3.5 h-3.5" />
                            选择
                          </button>
                        )}
                        <button
                          onClick={handleDownloadYoloModel}
                          disabled={yoloBusy}
                          className="shrink-0 inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-xl border border-[#ed8c68]/30 bg-[#ed8c68]/5 text-[#ed8c68] hover:bg-[#ed8c68]/10 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                        >
                          {yoloBusy ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          ) : (
                            <Download className="w-3.5 h-3.5" />
                          )}
                          下载
                        </button>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <label className="block text-[11px] font-medium text-gray-500">手动指定已有权重</label>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={yoloModelPath}
                          onChange={(e) => setYoloModelPath(e.target.value)}
                          placeholder="选择或输入 doclayout_yolo_*.pt 路径"
                          className="flex-1 min-w-0 px-3 py-2 rounded-xl border border-gray-200 bg-white text-xs text-gray-700 focus:outline-none focus:border-[#ed8c68]/50"
                        />
                        {window.chatpdfDesktop?.selectFile && (
                          <button
                            onClick={handleSelectYoloModelFile}
                            disabled={yoloBusy}
                            className="shrink-0 inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-xl border border-gray-200 bg-white text-gray-600 hover:bg-gray-100 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                          >
                            <FolderOpen className="w-3.5 h-3.5" />
                            选择
                          </button>
                        )}
                        <button
                          onClick={handleSaveYoloModelPath}
                          disabled={yoloBusy || !yoloModelPath.trim()}
                          className="shrink-0 inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-xl border border-green-200 bg-green-50/60 text-green-700 hover:bg-green-100/60 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
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
                        onClick={handleResetYoloModelPath}
                        disabled={yoloBusy}
                        className="text-[11px] text-gray-500 hover:text-gray-700 disabled:opacity-40 disabled:cursor-not-allowed"
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
                              : 'bg-blue-50/70 border border-blue-100 text-blue-700'
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
                </div>
              </div>
              </motion.div>
              )}
              </AnimatePresence>

              <AnimatePresence mode="wait">
              {activePanelTab === 'local' && (
              <motion.div
                key="tab-local-bottom"
                className="space-y-5"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.18, ease: 'easeOut' }}
              >
              {/* Poppler 不可用时的安装指引 */}
              {ocrStatus && !ocrStatus.poppler_available && (
                <div className="settings-card bg-white p-5 border border-gray-200/90">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <div className="text-sm font-semibold text-amber-800">
                        Poppler 未安装
                      </div>
                      <div className="text-xs text-amber-700 mt-1 leading-relaxed">
                        Poppler 用于将 PDF 页面转换为图像以进行 OCR
                        处理。未安装时 OCR 功能将不可用。
                      </div>
                      {ocrStatus.install_instructions && (
                        <div className="mt-3 space-y-1.5">
                          {Object.entries(ocrStatus.install_instructions).map(
                            ([platform, instruction]) => (
                              <div
                                key={platform}
                                className="text-xs bg-amber-50 border border-amber-100 rounded-lg px-3 py-2"
                              >
                                <span className="font-medium text-amber-800">
                                  {platform}：
                                </span>
                                <code className="text-amber-700 ml-1">
                                  {instruction}
                                </code>
                              </div>
                            )
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* OCR 后端不可用时的安装指引 */}
              {ocrStatus && !ocrStatus.available && (
                <div className="settings-card bg-white p-5 border border-gray-200/90">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <div className="text-sm font-semibold text-red-800">
                        无可用 OCR 引擎
                      </div>
                      <div className="text-xs text-red-700 mt-1 leading-relaxed">
                        未检测到任何可用的 OCR 后端。请安装 Tesseract 或
                        PaddleOCR 以启用 OCR 功能。
                      </div>
                      {ocrStatus.install_instructions && (
                        <div className="mt-3 space-y-1.5">
                          {Object.entries(ocrStatus.install_instructions).map(
                            ([key, instruction]) => (
                              <div
                                key={key}
                                className="text-xs bg-red-50 border border-red-100 rounded-lg px-3 py-2"
                              >
                                <span className="font-medium text-red-800">
                                  {key}：
                                </span>
                                <code className="text-red-700 ml-1">
                                  {instruction}
                                </code>
                              </div>
                            )
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
              </motion.div>
              )}
              </AnimatePresence>
            </div>

            {/* 底部状态栏 */}
            <div className="settings-chrome px-6 py-3 border-t border-gray-200 flex items-center justify-between">
              <div className="text-xs text-gray-400">
                设置已自动保存到本地
              </div>
              <div className="text-xs text-gray-400">
                当前模式：
                <span className="font-medium text-gray-600">
                  {OCR_MODES.find((m) => m.value === mode)?.label || mode}
                </span>
                {' · '}
                引擎：
                <span className="font-medium text-gray-600">
                  {BACKEND_OPTIONS.find((b) => b.value === backend)?.label || backend}
                </span>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}



