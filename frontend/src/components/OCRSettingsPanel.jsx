import React, { useEffect, useState, useCallback } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Eye,
  EyeOff,
  FileSearch,
  Globe,
  Info,
  Key,
  Loader2,
  Save,
  ScanText,
  Wifi,
  WifiOff,
  X,
  XCircle,
} from 'lucide-react'

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
 * 后端名称到中文显示名称的映射
 */
const BACKEND_LABELS = {
  tesseract: 'Tesseract',
  paddleocr: 'PaddleOCR',
  mistral: 'Mistral OCR',
  mineru: 'MinerU OCR',
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
  {
    value: 'mineru',
    label: 'MinerU OCR',
    description: '在线 OCR 服务，通过 Worker 代理，支持公式和表格识别',
  },
  {
    value: 'doc2x',
    label: 'Doc2X OCR',
    description: '在线 OCR 服务，通过 Worker 代理，支持 Dollar 公式模式',
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
const VALID_BACKENDS = ['auto', 'tesseract', 'paddleocr', 'mistral', 'mineru', 'doc2x']

/**
 * 从 localStorage 读取 OCR 设置
 * @returns {object} OCR 设置对象，包含 mode 和 backend
 */
export function loadOCRSettings() {
  try {
    const raw = localStorage.getItem(OCR_SETTINGS_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      const result = { mode: 'auto', backend: 'auto' }
      // 校验 mode 值是否合法
      if (VALID_MODES.includes(parsed.mode)) {
        result.mode = parsed.mode
      }
      // 校验 backend 值是否合法
      if (VALID_BACKENDS.includes(parsed.backend)) {
        result.backend = parsed.backend
      }
      return result
    }
  } catch (err) {
    console.error('读取 OCR 设置失败:', err)
  }
  return { mode: 'auto', backend: 'auto' }
}

/**
 * 将 OCR 设置保存到 localStorage
 * @param {object} settings - OCR 设置对象，包含 mode 和 backend
 */
export function saveOCRSettings(settings) {
  try {
    localStorage.setItem(OCR_SETTINGS_KEY, JSON.stringify(settings))
  } catch (err) {
    console.error('保存 OCR 设置失败:', err)
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
  // OCR 模式状态
  const [mode, setMode] = useState('auto')
  // OCR 引擎后端选择状态
  const [backend, setBackend] = useState('auto')
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

  // ---- MinerU OCR 配置状态 ----
  // MinerU Worker URL
  const [mineruWorkerUrl, setMineruWorkerUrl] = useState('')
  // MinerU Auth Key
  const [mineruAuthKey, setMineruAuthKey] = useState('')
  // MinerU Token
  const [mineruToken, setMineruToken] = useState('')
  // MinerU Token 模式：'frontend'（前端透传）或 'worker'（Worker 配置）
  const [mineruTokenMode, setMineruTokenMode] = useState('frontend')
  // MinerU OCR 选项
  const [mineruEnableOcr, setMineruEnableOcr] = useState(true)
  const [mineruEnableFormula, setMineruEnableFormula] = useState(true)
  const [mineruEnableTable, setMineruEnableTable] = useState(true)
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
    } catch (err) {
      console.error('获取 OCR 状态失败:', err)
      setError('无法获取 OCR 状态，请检查后端服务是否运行')
    } finally {
      setLoading(false)
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
    if (!mistralApiKey.trim()) return
    setValidateStatus('loading')
    setValidateMessage('')
    try {
      const res = await fetch(`${API_BASE_URL}/api/ocr/validate-key`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: 'mistral',
          api_key: mistralApiKey.trim(),
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
  }, [mistralApiKey])

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
    if (!mineruWorkerUrl.trim()) return
    setMineruValidating(true)
    setMineruValidateStatus(null)
    setMineruValidateMessage('')
    try {
      const res = await fetch(`${API_BASE_URL}/api/ocr/validate-key`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: 'mineru',
          worker_url: mineruWorkerUrl.trim(),
          auth_key: mineruAuthKey.trim(),
          token: mineruToken.trim(),
          token_mode: mineruTokenMode,
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
      setMineruValidateMessage(data.message || (data.valid ? '连接成功' : '连接失败'))
    } catch (err) {
      console.error('MinerU 测试连接失败:', err)
      setMineruValidateStatus('error')
      setMineruValidateMessage('无法连接到服务器，请检查后端服务')
    } finally {
      setMineruValidating(false)
    }
  }, [mineruWorkerUrl, mineruAuthKey])

  /**
   * 保存 MinerU OCR 配置
   */
  const handleMineruSave = useCallback(async () => {
    setMineruSaving(true)
    setMineruSaveMessage('')
    try {
      const body = {
        provider: 'mineru',
        worker_url: mineruWorkerUrl.trim(),
        auth_key: mineruAuthKey.trim(),
        token_mode: mineruTokenMode,
        token: mineruToken.trim(),
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
  }, [mineruWorkerUrl, mineruAuthKey, mineruTokenMode, mineruToken, mineruEnableOcr, mineruEnableFormula, mineruEnableTable, fetchOnlineConfig, fetchOCRStatus])

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
   * 面板打开时获取 OCR 状态和在线配置
   */
  useEffect(() => {
    if (isOpen) {
      fetchOCRStatus()
      fetchOnlineConfig()
    }
  }, [isOpen, fetchOCRStatus, fetchOnlineConfig])

  /**
   * 切换 OCR 模式并持久化到 localStorage
   * @param {string} newMode - 新的 OCR 模式
   */
  const handleModeChange = (newMode) => {
    setMode(newMode)
    saveOCRSettings({ mode: newMode, backend })
  }

  /**
   * 切换 OCR 引擎后端并持久化到 localStorage
   * @param {string} newBackend - 新的 OCR 引擎后端
   */
  const handleBackendChange = (newBackend) => {
    setBackend(newBackend)
    saveOCRSettings({ mode, backend: newBackend })
  }

  return (
    <AnimatePresence initial={false}>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/25 backdrop-blur-sm p-4"
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.95, opacity: 0, y: 10 }}
            transition={{
              type: 'spring',
              stiffness: 300,
              damping: 30,
              mass: 0.8,
            }}
            className="w-full max-w-xl max-h-[92vh] bg-white/80 backdrop-blur-2xl border border-white/50 shadow-[0_30px_80px_-35px_rgba(15,23,42,0.6)] rounded-[32px] overflow-hidden flex flex-col"
          >
            {/* 顶部标题栏 */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-amber-50 text-amber-700">
                  <ScanText className="w-5 h-5" />
                </div>
                <div>
                  <div className="text-lg font-bold text-gray-900">
                    OCR 设置
                  </div>
                  <div className="text-xs text-gray-500">
                    配置文档识别与文字提取
                  </div>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-2 rounded-full hover:bg-gray-100 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
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

              {/* OCR 可用状态卡片 */}
              <div className="soft-card rounded-[24px] p-5">
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
                            ? 'bg-green-500'
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
                        <span className="ml-auto text-xs text-purple-600 bg-purple-50 px-2 py-0.5 rounded-full border border-purple-100">
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
                                  available ? 'bg-green-500' : 'bg-gray-300'
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
                              ? 'bg-green-500'
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
              <div className="soft-card rounded-[24px] p-5">
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
                            ? 'border-purple-200 bg-purple-50/60 text-purple-700 shadow-sm'
                            : 'border-gray-100 hover:border-purple-200 hover:bg-purple-50/30 text-gray-700'
                        }`}
                      >
                        <div
                          className={`p-1.5 rounded-lg ${
                            isActive
                              ? 'bg-purple-100 text-purple-600'
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
                              isActive ? 'text-purple-500' : 'text-gray-400'
                            }`}
                          >
                            {option.description}
                          </div>
                        </div>
                        {isActive && (
                          <CheckCircle2 className="w-5 h-5 text-purple-500 flex-shrink-0" />
                        )}
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* OCR 引擎选择 */}
              <div className="soft-card rounded-[24px] p-5">
                <div className="flex items-center gap-2 mb-4">
                  <ScanText className="w-4 h-4 text-gray-500" />
                  <span className="text-sm font-semibold text-gray-800">
                    OCR 引擎选择
                  </span>
                </div>

                <div className="space-y-2">
                  {BACKEND_OPTIONS.map((option) => {
                    const isActive = backend === option.value
                    return (
                      <button
                        key={option.value}
                        onClick={() => handleBackendChange(option.value)}
                        className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl border text-left transition-all ${
                          isActive
                            ? 'border-purple-200 bg-purple-50/60 text-purple-700 shadow-sm'
                            : 'border-gray-100 hover:border-purple-200 hover:bg-purple-50/30 text-gray-700'
                        }`}
                      >
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium">
                            {option.label}
                          </div>
                          <div
                            className={`text-xs mt-0.5 ${
                              isActive ? 'text-purple-500' : 'text-gray-400'
                            }`}
                          >
                            {option.description}
                          </div>
                        </div>
                        {isActive && (
                          <CheckCircle2 className="w-5 h-5 text-purple-500 flex-shrink-0" />
                        )}
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* 在线 OCR 服务配置卡片 */}
              <div className="soft-card rounded-[24px] p-5">
                <div className="flex items-center gap-2 mb-4">
                  <Wifi className="w-4 h-4 text-gray-500" />
                  <span className="text-sm font-semibold text-gray-800">
                    在线 OCR 服务
                  </span>
                  {/* 已配置状态指示 */}
                  {onlineConfig?.mistral?.api_key_configured && (
                    <span className="ml-auto text-xs text-green-600 bg-green-50 px-2 py-0.5 rounded-full border border-green-100 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      已配置
                    </span>
                  )}
                </div>

                <div className="space-y-4">
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
                      disabled={!mistralApiKey.trim() || validateStatus === 'loading'}
                      className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium rounded-xl border border-purple-200 bg-purple-50/60 text-purple-700 hover:bg-purple-100/60 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
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
              </div>

              {/* MinerU OCR 配置卡片（可折叠） */}
              <div className="soft-card rounded-[24px] p-5">
                {/* 卡片标题栏（点击展开/折叠） */}
                <button
                  onClick={() => setMineruExpanded(!mineruExpanded)}
                  className="w-full flex items-center gap-2"
                >
                  <Globe className="w-4 h-4 text-gray-500" />
                  <span className="text-sm font-semibold text-gray-800">
                    MinerU OCR 服务
                  </span>
                  {/* 已配置状态指示 */}
                  {onlineConfig?.mineru?.worker_url && (
                    <span className="ml-auto mr-2 text-xs text-green-600 bg-green-50 px-2 py-0.5 rounded-full border border-green-100 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      已配置
                    </span>
                  )}
                  <ChevronDown
                    className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${
                      mineruExpanded ? 'rotate-180' : ''
                    } ${onlineConfig?.mineru?.worker_url ? '' : 'ml-auto'}`}
                  />
                </button>

                {/* 已配置状态预览（折叠时显示） */}
                {!mineruExpanded && onlineConfig?.mineru?.worker_url && (
                  <div className="mt-3 space-y-1.5">
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-50/80 text-xs text-gray-600">
                      <Globe className="w-3 h-3 text-gray-400" />
                      <span>Worker URL：</span>
                      <code className="font-mono text-gray-700">{onlineConfig.mineru.worker_url}</code>
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
                {mineruExpanded && (
                  <div className="mt-4 space-y-4">
                    {/* 已有配置预览 */}
                    {onlineConfig?.mineru?.worker_url && (
                      <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-green-50/60 border border-green-100">
                        <Globe className="w-3.5 h-3.5 text-green-600" />
                        <span className="text-xs text-green-700">
                          当前 Worker URL：
                          <code className="font-mono ml-1">
                            {onlineConfig.mineru.worker_url}
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
                          value={mineruWorkerUrl}
                          onChange={(e) => setMineruWorkerUrl(e.target.value)}
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

                    {/* Token Mode 选择 */}
                    <div>
                      <label className="text-xs font-medium text-gray-600 mb-1.5 block">
                        Token 模式
                      </label>
                      <div className="flex gap-2">
                        <button
                          onClick={() => setMineruTokenMode('frontend')}
                          className={`flex-1 px-3 py-2 text-xs font-medium rounded-xl border transition-all ${
                            mineruTokenMode === 'frontend'
                              ? 'border-purple-200 bg-purple-50/60 text-purple-700 shadow-sm'
                              : 'border-gray-200 bg-white/60 text-gray-600 hover:border-purple-200 hover:bg-purple-50/30'
                          }`}
                        >
                          前端透传
                        </button>
                        <button
                          onClick={() => setMineruTokenMode('worker')}
                          className={`flex-1 px-3 py-2 text-xs font-medium rounded-xl border transition-all ${
                            mineruTokenMode === 'worker'
                              ? 'border-purple-200 bg-purple-50/60 text-purple-700 shadow-sm'
                              : 'border-gray-200 bg-white/60 text-gray-600 hover:border-purple-200 hover:bg-purple-50/30'
                          }`}
                        >
                          Worker 配置
                        </button>
                      </div>
                      <div className="text-xs text-gray-400 mt-1">
                        {mineruTokenMode === 'frontend'
                          ? '由前端传递 Token 到 Worker'
                          : 'Token 在 Worker 环境变量中配置，无需前端提供'}
                      </div>
                    </div>

                    {/* Token 输入框（仅 frontend 模式显示，带显示/隐藏切换） */}
                    {mineruTokenMode === 'frontend' && (
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
                        {/* 启用 OCR */}
                        <label className="flex items-center justify-between px-3 py-2 rounded-xl bg-gray-50/80 border border-gray-100 cursor-pointer">
                          <span className="text-xs text-gray-700">启用 OCR</span>
                          <div
                            onClick={() => setMineruEnableOcr(!mineruEnableOcr)}
                            className={`relative w-9 h-5 rounded-full transition-colors cursor-pointer ${
                              mineruEnableOcr ? 'bg-purple-500' : 'bg-gray-300'
                            }`}
                          >
                            <div
                              className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
                                mineruEnableOcr ? 'translate-x-4' : 'translate-x-0'
                              }`}
                            />
                          </div>
                        </label>
                        {/* 启用公式识别 */}
                        <label className="flex items-center justify-between px-3 py-2 rounded-xl bg-gray-50/80 border border-gray-100 cursor-pointer">
                          <span className="text-xs text-gray-700">启用公式识别</span>
                          <div
                            onClick={() => setMineruEnableFormula(!mineruEnableFormula)}
                            className={`relative w-9 h-5 rounded-full transition-colors cursor-pointer ${
                              mineruEnableFormula ? 'bg-purple-500' : 'bg-gray-300'
                            }`}
                          >
                            <div
                              className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
                                mineruEnableFormula ? 'translate-x-4' : 'translate-x-0'
                              }`}
                            />
                          </div>
                        </label>
                        {/* 启用表格识别 */}
                        <label className="flex items-center justify-between px-3 py-2 rounded-xl bg-gray-50/80 border border-gray-100 cursor-pointer">
                          <span className="text-xs text-gray-700">启用表格识别</span>
                          <div
                            onClick={() => setMineruEnableTable(!mineruEnableTable)}
                            className={`relative w-9 h-5 rounded-full transition-colors cursor-pointer ${
                              mineruEnableTable ? 'bg-purple-500' : 'bg-gray-300'
                            }`}
                          >
                            <div
                              className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
                                mineruEnableTable ? 'translate-x-4' : 'translate-x-0'
                              }`}
                            />
                          </div>
                        </label>
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
                        disabled={!mineruWorkerUrl.trim() || mineruValidating}
                        className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium rounded-xl border border-purple-200 bg-purple-50/60 text-purple-700 hover:bg-purple-100/60 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
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
                        disabled={mineruSaving || !mineruWorkerUrl.trim()}
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
                )}
              </div>

              {/* Doc2X OCR 配置卡片（可折叠） */}
              <div className="soft-card rounded-[24px] p-5">
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
                {doc2xExpanded && (
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
                      <div className="flex gap-2">
                        <button
                          onClick={() => setDoc2xTokenMode('frontend')}
                          className={`flex-1 px-3 py-2 text-xs font-medium rounded-xl border transition-all ${
                            doc2xTokenMode === 'frontend'
                              ? 'border-purple-200 bg-purple-50/60 text-purple-700 shadow-sm'
                              : 'border-gray-200 bg-white/60 text-gray-600 hover:border-purple-200 hover:bg-purple-50/30'
                          }`}
                        >
                          前端透传
                        </button>
                        <button
                          onClick={() => setDoc2xTokenMode('worker')}
                          className={`flex-1 px-3 py-2 text-xs font-medium rounded-xl border transition-all ${
                            doc2xTokenMode === 'worker'
                              ? 'border-purple-200 bg-purple-50/60 text-purple-700 shadow-sm'
                              : 'border-gray-200 bg-white/60 text-gray-600 hover:border-purple-200 hover:bg-purple-50/30'
                          }`}
                        >
                          Worker 配置
                        </button>
                      </div>
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
                        className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium rounded-xl border border-purple-200 bg-purple-50/60 text-purple-700 hover:bg-purple-100/60 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
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
                )}
              </div>

              {/* Poppler 不可用时的安装指引 */}
              {ocrStatus && !ocrStatus.poppler_available && (
                <div className="soft-card rounded-[24px] p-5">
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
                <div className="soft-card rounded-[24px] p-5">
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
            </div>

            {/* 底部状态栏 */}
            <div className="px-6 py-3 border-t border-gray-100 flex items-center justify-between">
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
