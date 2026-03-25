# 问题：引文与回答耦合过深，导致相关性差、编号乱序、前后端互相污染

## 现象

### 1. 引文相关性不高

- 回答中的某些句子虽然带了 `[1]`、`[2]`，但对应证据和句子本身并不匹配
- 常见表现是多个独立事实都被标成同一个编号
- 也会出现正文里有编号，但证据面板中的引用与正文不一致

### 2. 引用编号不是连续展示编号

- 用户看到的不是 `[1][2][3]`
- 而是类似“正文第一处是 `[1]`，第二处直接变成 `[5]`”
- 这会让用户误以为中间引用丢失，或者点击联动关系有问题

### 3. 改正文容易把引文搞坏，改引文也容易把正文搞坏

- 前端既负责展示，又在最终收口阶段本地重写正文引用
- 同时又根据正文里的编号再过滤/重排 citations
- 结果是“正文”和“证据列表”互相作为对方输入，形成耦合闭环

---

## 根因分析

问题不是单点 bug，而是“谁负责最终引文结果”这个边界不清。

### 1. 后端不是唯一真源，前端还在本地二次改写正文

旧链路里，后端虽然会返回 `answer + citations`，但前端还会继续做这些事：

- 本地纠正文中引用
- 本地补引用
- 本地按正文编号过滤 citations
- 本地重新决定展示顺序

这样一来：

- 后端改了 citation 结构，前端可能把它重新改坏
- 前端为了修正文案，又会影响 citations 的排序和联动

**文件**:
- `frontend/src/hooks/useMessageState.js`

### 2. 引文只有一个 `ref`，混用了“原始证据编号”和“展示编号”

旧逻辑里 `ref` 同时承担两种职责：

- 作为后端检索命中的原始来源编号
- 作为用户在正文和面板里看到的显示编号

这会直接导致：

- 正文中按原始编号显示，出现 `[1][5]`
- 面板和正文必须猜“这个 ref 到底是原始编号还是展示编号”

### 3. 引文后处理分散在多处，顺序不稳定

此前涉及引文修复的逻辑分散在多个位置：

- 坏格式修复
- 单一错误引用纠偏
- 无引用时补引用
- 前端按正文再过滤

这些逻辑前后都有，最终收口顺序不固定，导致一个问题修了，另一个问题又被重新引入。

### 4. 结构化引文模式下，正文和 `CITATION LIST` 的收口不在一个权威流程里

参考 kotaemon 的结构化引文设计时，系统已经具备：

- `FINAL ANSWER`
- `CITATION LIST`
- `start_phrase / end_phrase`

但“最终正文怎么重排展示编号”“最终 citations 如何投影成用户可见顺序”之前没有彻底收口到同一个后处理流程里。

---

## 修复方案

### 修复 1：后端统一输出最终展示版 `answer + citations`

新增后端统一收口流程：

```python
_prepare_answer_and_citations_for_display(answer, citations)
```

这个流程负责：

- 标准化 citation 记录
- 修复坏格式引用
- 纠正“整篇只有一个引用编号”的错误情况
- 按句子与证据的支撑关系优化引用
- 无内联引用时自动注入
- 按正文首次出现顺序投影展示编号
- 重写正文中的内联引用编号

这样最终交给前端的就是：

- 最终正文
- 最终 citations

前端不再需要重新决定“正确答案是什么”。

**文件**:
- `backend/routes/chat_routes.py`

### 修复 2：引文拆成 `source_ref` 和 `display_ref`

每条 citation 现在至少保留两套编号：

- `source_ref`: 后端原始证据编号，用于追踪检索来源、PDF 联动
- `display_ref`: 按正文首次出现顺序重排后的展示编号，从 `1..N` 连续递增

同时：

- `ref` 在前端消费时统一视作展示编号
- 正文中的内联引用最终也会被改写成 `display_ref`

因此用户看到的编号现在应该是连续的 `[1][2][3]`，而不是原始检索编号。

### 修复 3：前端停止本地重写最终正文

前端最终收口改成只做两件事：

- 接收后端最终正文
- 规范化 citations 结构

不再由前端负责：

- 重排正文引用
- 补正文引用
- 再次根据正文决定 citations 顺序

这一步本质上把“引文正确性”从前端搬回了后端。

**文件**:
- `frontend/src/hooks/useMessageState.js`

### 修复 4：展示层统一按 `display_ref` 渲染

正文、引用链接、证据面板统一优先使用：

```js
display_ref ?? ref
```

这样可以保证：

- 正文中用户看到的是连续编号
- 面板排序和正文一致
- `source_ref` 继续保留给内部联动使用，不直接暴露给用户

**文件**:
- `frontend/src/components/StreamingMarkdown.jsx`
- `frontend/src/components/EvidencePanel.jsx`

### 修复 5：保留结构化引文匹配，但把最终显示投影收口到后端

继续沿用 kotaemon 风格的结构化输出：

- `FINAL ANSWER`
- `CITATION LIST`
- `START_PHRASE / END_PHRASE`

但最终显示时：

- `highlight_text`
- `page_range`
- `display_ref / source_ref`
- 正文中的内联编号

都由后端同一条 display-prep 流程统一生成。

---

## 修复后的行为

### 已改善

- 引文编号会按正文出现顺序重排为连续编号
- 正文和证据面板使用同一套展示编号
- 证据点击跳转仍然保留原始来源信息，不受展示编号影响
- 前端不再本地重写最终正文，避免“改正文带坏引文 / 改引文带坏正文”

### 仍需注意

- 流式阶段允许先显示未最终整理的内容
- 最终 `done` 收口时，必须以后端权威版 `final_content + retrieval_meta.citations` 为准

也就是说：

> 流式阶段可以暂时不完美，但最终展示结果必须由后端统一定稿。

---

## 测试覆盖

后端补充了针对以下场景的测试：

- 坏引文格式修复
- 单一错误引用纠偏
- `display_ref / source_ref` 投影
- 无编号时自动注入引用
- `FINAL ANSWER / CITATION LIST` 跨 chunk 到达

前端补充了以下回归：

- 最终收口时不再本地改写正文
- 证据面板按 `display_ref` 排序
- 保留 `source_ref` 供证据联动

---

## 涉及文件

- `backend/routes/chat_routes.py` — 引文 display-prep 总收口、编号投影、正文编号重写
- `backend/tests/test_citation_relevance.py` — 引文编号与相关性回归测试
- `frontend/src/hooks/useMessageState.js` — 前端停止本地重写最终正文
- `frontend/src/components/StreamingMarkdown.jsx` — 正文引用展示按 `display_ref`
- `frontend/src/components/EvidencePanel.jsx` — 证据面板排序和展示按 `display_ref`
- `frontend/src/components/__tests__/EvidencePanel.test.jsx` — 证据面板展示编号回归
