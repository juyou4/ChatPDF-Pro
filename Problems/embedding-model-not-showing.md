# 问题：新增 Embedding 模型后列表不显示

## 现象

在 `EmbeddingSettings` 页面点击"保存模型"后：
- ✅ 成功 toast 弹出（"已添加: xxx"）
- ✅ 对应类型分组自动展开
- ❌ 新增的模型在列表中不出现

刷新页面后模型可以正常显示（说明持久化没问题，是 UI 实时渲染问题）。

---

## 根因

`EmbeddingSettings.jsx` 中的 `modelsByType` 原先通过 `getModelsByProvider(activeProvider.id)` 计算，该函数从 `ModelContext` 中获取，依赖 context 的传播时序。

```js
// ❌ 问题代码：依赖派生函数，不是基础 state
const { getModelsByProvider } = useModel()

const modelsByType = useMemo(() => {
  const list = getModelsByProvider(activeProvider.id)  // 函数引用可能没变
  ...
}, [activeProvider, getModelsByProvider])  // 函数引用不稳定
```

### 具体失败路径

1. `handleAddModel` 调用 `addModelToCollection(model)` → `ModelProvider` 中的 `setUserCollection` 触发
2. 同一 batch 内 `EmbeddingSettings` 自身也有多个 state 变化（`setCollapsedTypes`、`setAddSuccess` 等）
3. React 处理批量更新时，`EmbeddingSettings` 可能在 `ModelProvider` context 新值传播前就完成重渲染
4. `useMemo([..., getModelsByProvider])` 的函数引用未变 → 跳过重算 → 列表不更新

---

## 修复方案

### 方案一：直接订阅基础 state（Cherry Studio 对齐）

将 `modelsByType` 的数据来源从"派生函数"改为直接订阅 `userCollection` 和 `systemModels` 两个基础 state：

```js
// ✅ 修复后：直接依赖基础 state，引用变化即重算
const { userCollection, systemModels } = useModel()

const modelsByType = useMemo(() => {
  const sysForProvider = systemModels.filter(m => m.providerId === activeProvider.id)
  const userForProvider = userCollection.filter(m => m.providerId === activeProvider.id)
  const map = new Map()
  sysForProvider.forEach(m => map.set(m.id, m))
  userForProvider.forEach(m => map.set(m.id, m))
  return Array.from(map.values())
}, [activeProvider, systemModels, userCollection])
```

### 方案二：Optimistic Update（最终采用）

在 `EmbeddingSettings` 内维护 `locallyAddedModels` 本地 state，保存时立即写入，不依赖 context 传播：

```js
const [locallyAddedModels, setLocallyAddedModels] = useState([])

const handleAddModel = () => {
  const newModel = { id, providerId: activeProvider.id, type, ... }
  // 同时写入本地（立即显示）和 context（持久化）
  setLocallyAddedModels(prev => [...prev, newModel])
  addModelToCollection(newModel)
}

const modelsByType = useMemo(() => {
  ...
  locallyAddedModels.filter(m => m.providerId === activeProvider.id).forEach(m => map.set(m.id, m))
  ...
}, [activeProvider, systemModels, userCollection, locallyAddedModels, lastAddedModelKey])
```

`setLocallyAddedModels` 是 `EmbeddingSettings` 自身的 state 变化，**必然触发自身重渲染**，不受 context 传播时序影响。

---

## 快速排查方法（下次遇到类似问题）

### 第一步：确认 state 是否在更新

在组件内临时加一个固定悬浮显示：

```jsx
<div style={{position:'fixed',top:0,right:0,zIndex:9999,background:'red',color:'white'}}>
  {userCollection.length}
</div>
```

- 数字变了 → state 在更新，但 `useMemo` 没有重算（依赖数组有问题）
- 数字没变 → 上游 setter 没有触发（检查 setter 调用路径）

### 第二步：检查 `useMemo` 依赖数组

如果依赖的是**函数**（`getXxx()`、`computeXxx()`）而非**具体值**（`array`、`string`），大概率是问题所在。  
函数引用每次可能相同（来自 context 或 useCallback），导致 memo 跳过重算。

### 第三步：用 IIFE 快速验证

把 `useMemo` 临时改为 IIFE：

```js
// const modelsByType = useMemo(() => {...}, [...])
const modelsByType = (() => { ... })()
```

如果列表立即显示 → 确认是 `useMemo` 依赖缺失问题。

---

## 核心原则

> **`useMemo` 的依赖数组必须放基础 state 值，不能放捕获了 state 的函数引用。**

参考 Cherry Studio 的模式：`useProvider(id)` 直接返回 `provider.models`（基础 state），列表直接订阅它，不经过任何派生函数。

---

## 涉及文件

- `frontend/src/components/EmbeddingSettings.jsx` — 添加 `locallyAddedModels` state，修改 `modelsByType` 依赖
- `frontend/src/contexts/ModelContext.tsx` — 移除调试 `console.log`
