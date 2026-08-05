# ChatPDF Pro v3.1.0

<div align="center">

![ChatPDF Logo](https://img.shields.io/badge/ChatPDF_Pro-3.1.0-blue?style=for-the-badge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![React](https://img.shields.io/badge/React-18.3-61dafb?style=for-the-badge&logo=react)](https://reactjs.org)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org)

**Smart Document Assistant - Chat with your PDFs** · [中文](README.md)

[Quick Start](#quick-start) • [Features](#core-features) • [Changelog](#changelog) • [Tech Stack](#tech-stack) • [Architecture](#architecture)

</div>

---

## Overview

ChatPDF Pro is a local-first AI reading assistant tailored for academic papers and long technical documents. The native PDF viewer on the left and the AI chat panel on the right, combined with a retrieval pipeline built on **Semantic Groups + three-tier granularity + dual-index RRF**, lets the model handle both high-level summarisation and pinpoint lookups down to a single table row. Chat history and runtime data stay local by default. Choosing MinerU, a remote model, or web search sends the relevant document or request to the configured service.

---

## App Preview

> The two blocks below showcase the main UI and a chat example. If you build from source, just take your own screenshots and overwrite `docs/preview_overview.png` and `docs/preview_chat.png` — no README edits required.

### One-Click Overview · `docs/preview_overview.png`

<div align="center">

<!-- Overview screenshot: right pane switched to the "速览 / Overview" tab. Ideally the frame shows both the "Speed-Read" and "Key Figure Analysis" cards. -->
<img src="docs/preview_overview.png" alt="ChatPDF Pro one-click overview panel" width="880" />

</div>

**Left:** the native PDF reader (PDF.js + text-selection toolbar). **Right:** a toggle between the **Overview (速览)** and **Chat (对话)** tabs. The overview is generated in the background as soon as a PDF is uploaded and is composed of five structured cards:

- **Abstract-level summary** — one paragraph describing what the paper does and why it matters.
- **Terminology** — core concepts lifted from the document with inline definitions.
- **Speed-Read** — three-bullet breakdown: method / experimental design / problem solved.
- **Key Figure Analysis** — page figures extracted automatically (**MinerU-enhanced**, **PDF-native**, or **caption-only (vector figures)** sources are supported) with AI commentary beyond the original caption.
- **Paper Summary** — strengths / innovations plus suggested future work.

The screenshot shows the Speed-Read and Key Figure Analysis cards for arXiv:2603.15031 *Attention Residuals*. Overview cards share the same document context with the Chat tab; tapping any term or figure reference jumps back to the matching page in the PDF.

### Chat in Action · `docs/preview_chat.png`

<div align="center">

<!-- Chat screenshot: capture one full round-trip showing [1][2] citations, the collapsible thinking block, and follow-up suggestions. -->
<img src="docs/preview_chat.png" alt="ChatPDF Pro chat example" width="880" />

</div>

Inline `[1]` `[2]` citations jump straight to the matching PDF page. Inline / block math renders live, deep-thinking blocks are collapsed by default, and each reply is followed by 3-5 suggested follow-up questions plus an optional hallucination-critic banner.

### Standalone Desktop Client

A self-contained Windows desktop application built with Electron. The Python backend is fully integrated and packaged using PyInstaller, ensuring it works **out of the box without any Python or Node.js environment configuration.**

- **Privacy-safe distribution** - The installer contains only the renderer, Electron main process, and frozen backend runtime. It never ships uploaded documents, chat history, generated overview/outline/summary caches, indexes, logs, `.env` files, or API keys.
- **Separate runtime storage** - On first launch, the desktop app creates its runtime data in the operating system's user-data location. Installing or upgrading does not delete existing local data, and that data is never copied back into the installer.

---

## Release Highlights (v3.1.0)

### Desktop Architecture (Electron)
- **Standalone Application** - Windows desktop client built on Electron 28, breaking free from browser limitations.
- **One-Click Installation** - Provided as an NSIS installer. Just double-click to install and run.
- **Embedded Backend** - FastAPI backend packaged with PyInstaller. The app's process manager automatically finds an available port and spawns the backend service upon startup.

### Deep Thinking Mode
- **Reasoning Visualization** - Real-time display of the AI's ThinkingBlock in the chat area, with manual collapse/expand support.
- **Adjustable Intensity** - Dynamically adjust reasoning intensity (Low, Medium, High) in chat settings.
- **Smooth Streaming** - Both thinking processes and final responses support RequestAnimationFrame-based smooth character-by-character rendering.

### Math Formula Rendering
- **Dual Engines** - Built-in KaTeX and MathJax engines. Users can switch between them or disable rendering entirely via settings.
- **Single Dollar Support** - Renders inline math with `$...$`, resolving conflicts between plain text and formula syntax.
- **LaTeX Bracket Conversion** - Employs a balanced matching algorithm to automatically convert `\[...\]` and `\(...\)` into standard Markdown math syntax.

### Connectivity & UI Optimization
- **Web Search** - Allows the AI to fetch real-time internet information, displaying clear source links at the bottom of the response.
- **Render Performance** - Implements Virtual List scrolling, maintaining 60fps performance even with extensive, text-heavy conversation histories.
- **DOM Direct-Write** - Bypasses React state updates during streaming output by directly modifying DOM nodes via refs, significantly reducing memory and CPU footprint.

---

## Core Features

### PDF Document Processing
- **Native Rendering** - High-fidelity document display via PDF.js with smooth zooming, pagination, and text selection.
- **Character-level text extraction** - The primary pipeline is **PyMuPDF `get_text("dict")`** with adaptive coordinate thresholds (line-break / whitespace detection); on failure it falls back to **pdfplumber**'s chars API. A heuristic rebuild pass repairs hyphenated line breaks and zh/en punctuation artefacts.
- **Table structuring** - `services/table_aware_service.py` uses **PyMuPDF `find_tables()`** to detect table regions and convert them into `[TABLE]`-tagged Markdown, which the chunker treats as protected regions.
- **Two explicit parse routes** - Choose **MinerU deep parsing** or **local parsing** before upload. Local Tesseract / PaddleOCR is only a low-quality-page supplement and never silently replaces the selected route.

### Intelligent Retrieval (RAG v3.0+)
- **Semantic Groups** - Aggregates scattered text chunks into semantically coherent units of ~5000 characters, respecting page, heading, and table boundaries.
- **Three-Level Granularity** - Automatically generates Summary (80 chars), Digest (1000 chars), and Full text representations for every semantic group.
- **Dynamic Granularity Matching** - Leverages LLMs to infer user intent (e.g., overview, extraction, specific data) during retrieval, automatically returning the optimal text granularity.
- **Token Budget Control** - Estimates token counts accurately based on target models and character properties (Chinese vs. English). Triggers intelligent granularity degradation instead of hard truncation when approaching context limits.
- **Dual-Index Retrieval** - Queries both chunk-level and group-level FAISS vector indexes simultaneously, combining with BM25 algorithms and RRF (Reciprocal Rank Fusion) for reranking.
- **Numeric-Table Specialisation** - A dedicated retrieval branch for numeric comparison queries ("second-best method", "Table 7 DiffuLT"); when a table row is hit, sibling rows are back-filled as contrastive context. Toggled by a feature flag.
- **BM25 Synonym Expansion** - Query-time expansion using a bundled zh/en synonym dictionary plus fine-grained Chinese tokenisation to boost recall.
- **Dual-Model Strategy (`cheap_model`)** - Non-core LLM tasks (query rewriting, sub-question decomposition, follow-up suggestions, answer critic) are routed to a cheaper model, saving 40-60% tokens without touching the primary answer model.
- **LLM Query Rewriting** - Resolves co-references across multi-turn dialogue ("it", "this method"); long queries skip rewriting entirely.
- **Answer Critic** - After the final answer, `cheap_model` cross-checks the response against the retrieved snippets; hallucinations are surfaced as a red warning banner (flag-gated).

### One-Click Overview Panel
- **Five structured cards** - As soon as a PDF is uploaded, the overview pane auto-generates *Abstract Summary · Terminology · Speed-Read · Key Figure Analysis · Paper Summary*. It shares the same document context with the Chat tab.
- **Figure-extraction adapter chain** - The default path is **PDF-native** (PyMuPDF image objects + Figure caption spatial matching, with 1a/1b sub-figure grouping); vector-figure PDFs fall onto a **caption-only** path that crops the figure from caption coordinates; if **MinerU cloud OCR** (via a Cloudflare Worker proxy) is enabled, its `middle.json` layout analysis is used first — ideal for scanned / image-based PDFs.
- **AI figure commentary** - Every extracted figure is run through a vision model to produce an explanation that goes beyond the original caption, instead of just showing the raw picture.
- **Adjustable depth** - The `depth` switch accepts `brief` / `standard` / `detailed`, which drive per-card character caps (150/400/600), term counts (3/5/8) and figure counts (2/3/5).

### AI Chat Capabilities
- **Multi-Model Support** - Native integration with OpenAI, Anthropic, Google Gemini, Grok, and local Ollama models.
- **Precise Citations** - Automatically generates [1], [2] inline citations. Clicking a citation highlights the source in the PDF view and scrolls smoothly to the exact page.
- **Text Selection Toolbar** - Selecting text in the PDF triggers a floating toolbar for instant AI explanation, translation, or inclusion as context for the next query.
- **Visual Diagrams** - Automatically parses and renders Mermaid code blocks generated by the AI, perfect for flowcharts and mind maps.

---

## Quick Start

### Option 1: Download Desktop Client (Recommended)

Download the latest `.exe` installer directly from the [Releases](https://github.com/juyou4/ChatPDF-Pro/releases) page.
Install and double-click the desktop icon to run. No environment setup required.

### Build a Clean Windows Installer

The following writes a build identity manifest, rebuilds the frontend, freezes the Python backend, and produces an x64 NSIS installer:

```bash
scripts/build-all.bat
```

The default output is in `electron/release/`. Installer filenames include the app version and Git short SHA, and the build writes `release-manifest.json` plus `.sha256` checksum files. Packaging rules exclude `data/`, `uploads/`, `logs/`, `cache/`, `history/`, `memory/`, vector and semantic indexes, test/course/evaluation directories, PDFs, database/serialized artifacts, `.env` files, logs, and API-key-named files at both the PyInstaller and Electron resource-copy stages. Do not manually copy a local user-data directory or development `data/` directory into the release output.

`version.json` at the repository root is the single release metadata source. `/version`, `/health`, `/capabilities`, the Electron package version, and the frontend public version must match it. Before release, run:

```bash
python scripts/release_metadata.py --check
```

### Option 2: Run from Source (Web Mode)

**1. Backend Service (Python 3.10+)**
```bash
cd backend
python -m pip install -r requirements.txt
python app.py
```

**2. Frontend Service (Node.js 20.19+, or 22.12+)**
```bash
cd frontend
npm install
npm run dev
```
Visit `http://localhost:3000` to start using the application.

---

## Tech Stack

### Frontend
- **Core**: React 18 + Vite 7 + Tailwind CSS
- **PDF Rendering**: react-pdf 9.0 + PDF.js
- **UI Animation**: Framer Motion
- **Markdown**: ReactMarkdown + rehype-katex / rehype-mathjax
- **Desktop Environment**: Electron 28 + electron-builder

### Backend
- **Framework**: FastAPI 0.115 (Uvicorn async driven)
- **PDF Processing**: PyMuPDF 1.24 (primary) + pdfplumber 0.11 (fallback)
- **Vector Database**: FAISS 1.9
- **Retrieval Architecture**: Semantic Groups + Dual-Index RRF Fusion
- **Model SDKs**: OpenAI, Anthropic, Google Generative AI
- **Parsing and OCR**: MinerU (Worker proxy or direct API) / local Tesseract and PaddleOCR supplements

---

## Architecture

```text
ChatPDF/
├── frontend/                    # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatPDF.jsx          # Main application component
│   │   │   ├── PDFViewer.jsx        # PDF rendering core
│   │   │   ├── StreamingMarkdown.jsx # Markdown + Math + Mermaid rendering
│   │   │   ├── ThinkingBlock.jsx    # Deep thinking visualizer
│   │   │   ├── ChatSettings.jsx     # Chat parameter configuration
│   │   │   ├── VirtualMessageList.jsx # Virtualized scroll list
│   │   │   ├── PresetQuestions.jsx   # Quick action buttons
│   │   │   └── CitationLink.jsx     # Interactive citation links
│   │   ├── contexts/
│   │   │   ├── ChatParamsContext.jsx # Chat parameters (incl. Math Engine)
│   │   │   ├── GlobalSettingsContext.jsx
│   │   │   └── WebSearchContext.jsx  # Web search state
│   │   ├── hooks/
│   │   │   ├── useMessageState.js    # Message state & streaming requests
│   │   │   └── useSmoothStream.js    # Smooth streaming orchestrator
│   │   └── utils/
│   │       └── processLatexBrackets.js # Balanced LaTeX bracket parser
│   ├── package.json
│   └── vite.config.js
├── backend/                     # FastAPI backend
│   ├── app.py                   # Main application entry
│   ├── desktop_entry.py         # PyInstaller frozen entry point
│   ├── routes/                  # API routing layer
│   ├── services/
│   │   ├── semantic_group_service.py  # Semantic group chunking
│   │   ├── hybrid_search.py           # Hybrid retrieval + RRF fusion
│   │   ├── context_builder.py         # Prompt assembly & citation generation
│   │   ├── chat_service.py            # AI chat logic & thinking stream handler
│   │   ├── web_search_service.py      # Internet search engine integration
│   │   ├── embedding_service.py       # Embedding calculation & FAISS indexing
│   │   └── rerank_service.py          # Cross-encoder reranking
│   └── requirements.txt
├── electron/                    # Electron desktop environment
│   ├── src/main.ts              # Main process: Window mgmt, Backend spawner
│   └── package.json
├── scripts/                     # Cross-platform build scripts
└── README.md
```

---

## FAQ

**Q: The desktop client launches with a blank screen or throws an error?**
A: Ensure no system proxy is intercepting localhost traffic, or try running as Administrator. On first launch, the app silently starts the Python engine in the background, which may take a few seconds.

**Q: Why can I still see old documents or chat history after installing a new build?**
A: The installer does not contain user history. When it is installed or upgraded under the same Windows account, the desktop app intentionally continues using that account's existing application-data directory, so locally stored records remain visible. A new operating-system account or a machine with no prior application data starts blank.

**Q: PDF doesn't display in Web mode?**
A: Verify the backend service is running (default port 8000). Check the browser console for CORS or network interception errors.

**Q: API calls fail or timeout?**
A: Check if your API Key format is correct. Ensure your network environment can reach the provider's endpoint (e.g., OpenAI may require specific network routing or a custom base URL proxy).

**Q: Connection refused when using local models (Ollama)?**
A: Ensure the Ollama background service is running and you have set the system environment variable `OLLAMA_ORIGINS="*"` to allow Cross-Origin requests.

**Q: Math formulas are rendering as garbage text?**
A: Switch between KaTeX and MathJax in the settings panel (bottom left). KaTeX is faster, while MathJax offers better compatibility for complex nested LaTeX.

---

## Changelog

### v3.1.0
- MinerU full-route parsing identity, downstream cache invalidation, and shared visual assets.
- Reading workspace UI refinements, persistent font settings, markdown note editing, and toolbar layout improvements.
- Engineering identity cleanup: unified version metadata, reproducible build manifest, privacy-safe packaging checks, and runtime log directory alignment.

### v3.0.2
- **Numeric-Table Specialisation**: New retrieval branch for "Table N" / numeric-comparison queries, unified under a single feature flag.
- **BM25 Synonym Expansion**: Built-in zh/en synonym dictionary + fine-grained tokenisation, measurably improves recall on long Chinese queries.
- **Dual-Model Strategy (`cheap_model`)**: Query rewriting, decomposition, follow-up suggestions, and the answer critic each take an independent cheap model.
- **LLM Query Rewriting + Answer Critic**: Multi-turn co-reference resolution; post-hoc hallucination detection with a red warning banner.
- **Per-Request Feature-Flag Overrides**: New "Retrieval Tuning" panel in GlobalSettings with tri-state switches (Auto / On / Off) — no backend restart required.

### v3.0.1
- **Desktop Client Release**: Full Windows standalone application packaged via Electron 28 and PyInstaller.
- **Deep Thinking Enhancements**: Introduced `ThinkingBlock` for multi-tier reasoning visualization and smooth collapsing.
- **Math Engine Iteration**: Support for hot-swapping between KaTeX and MathJax, resolving rendering crashes on complex LaTeX.
- **Render Optimization**: Rewrote `StreamingMarkdown`'s underlying logic to use DOM Ref direct-writes, bypassing React reconciliation overhead. Added virtual lists to eliminate lag in extensive histories.

### v3.0.0
- **RAG Architecture Rewrite**: Introduced Semantic Groups and a three-tier granularity (Full/Digest/Summary) degradation strategy.
- **Precise Token Accounting**: Dynamic budget system based on language character properties.
- **Dual-Track Retrieval**: Combined group-level and chunk-level FAISS vector retrieval with RRF fusion.

---

## Acknowledgments

The RAG retrieval pipeline in this project, specifically the concepts of "Semantic Groups" and "Multi-level Granularity Auto-degradation," was inspired by the design philosophy of [Paper Burner X](https://github.com/Feather-2/paper-burner-x). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
