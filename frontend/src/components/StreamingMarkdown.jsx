import React, { useRef, useEffect, useState, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';

/**
 * StreamingMarkdown - 支持实时Markdown渲染和Blur Reveal效果的组件
 *
 * 策略：
 * - 将内容分为"稳定部分"和"新生成部分"
 * - 只有新生成的部分应用模糊动画
 * - 动画结束后，新部分合并入稳定部分
 */
const StreamingMarkdown = ({
  content,
  isStreaming,
  enableBlurReveal,
  blurIntensity = 'medium'
}) => {
  // 如果不启用特效或不在流式传输中，直接渲染普通Markdown
  if (!enableBlurReveal || !isStreaming) {
    return (
      <div className="prose prose-sm max-w-none dark:prose-invert">
        <ReactMarkdown
          remarkPlugins={[remarkMath]}
          rehypePlugins={[rehypeKatex, rehypeHighlight]}
        >
          {content}
        </ReactMarkdown>
      </div>
    );
  }

  // 记录上一次的内容长度，用于区分新旧内容
  const prevLengthRef = useRef(0);
  const [stableContent, setStableContent] = useState('');
  const [newContent, setNewContent] = useState('');

  useEffect(() => {
    if (content.length > prevLengthRef.current) {
      // 有新内容增加
      const newPart = content.slice(prevLengthRef.current);

      // 更新新内容（这部分会有模糊效果）
      setNewContent(newPart);

      // 更新稳定内容（之前的所有内容）
      setStableContent(content.slice(0, prevLengthRef.current));

      // 更新长度记录
      prevLengthRef.current = content.length;

      // 设置一个定时器，将新内容合并到稳定内容中（视觉上已经清晰了）
      // 这个时间应该与CSS动画时间匹配
      const timer = setTimeout(() => {
        setStableContent(content);
        setNewContent('');
      }, 300); // 300ms 足够覆盖动画时间

      return () => clearTimeout(timer);
    } else if (content.length < prevLengthRef.current) {
      // 内容减少（可能是重置），重置状态
      prevLengthRef.current = content.length;
      setStableContent(content);
      setNewContent('');
    }
  }, [content]);

  // 根据强度获取动画类名
  const getBlurClass = () => {
    switch (blurIntensity) {
      case 'strong': return 'animate-blur-reveal-strong';
      case 'light': return 'animate-blur-reveal-light';
      case 'medium':
      default: return 'animate-blur-reveal-medium';
    }
  };

  return (
    <div className="prose prose-sm max-w-none dark:prose-invert relative">
      {/* 
        这里有个难点：Markdown渲染是整体的，如果简单拼接字符串分别渲染，
        可能会破坏Markdown语法（比如一个加粗词被切分在两部分）。
        
        为了解决这个问题，我们采用一种视觉欺骗的方法：
        1. 渲染完整的Markdown（作为底层）
        2. 渲染一个覆盖层，只包含新增加的字符，并应用模糊动画
        
        或者更简单的方法（当前采用）：
        只对追加的纯文本字符应用模糊。如果Markdown结构复杂，
        这种方法可能在某些边界情况下表现不完美，但对流式输出通常足够好。
      */}

      {/* 方案A：直接渲染完整内容，通过CSS Mask或类似技术（太复杂） */}
      {/* 方案B：渲染完整内容，但给最后一部分应用动画类（需要DOM操作，React中不推荐） */}

      {/* 方案C：简单渲染完整内容，但使用一个特殊的Key强制React重新渲染最后一部分？不，这会导致闪烁 */}

      {/* 
         修正方案：
         由于Markdown渲染的特殊性，我们无法轻易将Markdown文本拆分为"稳定"和"新"两部分分别渲染。
         例如 `**bold**` 如果拆分为 `**bo` 和 `ld**`，两边都不会正确渲染。
         
         因此，对于Markdown流式输出，最佳的Blur Reveal实现方式其实是：
         渲染整个Markdown，但使用CSS动画让*新出现的DOM节点*产生模糊效果。
         但这需要深入到ReactMarkdown的渲染树中。
         
         退而求其次的实用方案：
         渲染整个Markdown，但应用一个全局的"打字机光标"或"末尾模糊"效果。
         
         或者，我们可以回到用户提到的问题：整行都模糊。
         这是因为我们将模糊类加到了容器上。
         
         让我们尝试一个基于CSS的解决方案：
         只对最后一个文本节点应用动画？
      */}

      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={{
          // 自定义文本渲染，尝试捕获最后一部分文本
          // 这在ReactMarkdown中比较难做

          // 替代方案：使用一个覆盖层显示新字符的模糊动画，
          // 而底层Markdown保持正常渲染。
          // 只要新字符是纯文本，这就能工作。
        }}
      >
        {content}
      </ReactMarkdown>

      {/* 
        覆盖层：只显示新增加的文本片段，应用模糊动画。
        为了定位，我们将其放在流文档的末尾？不，这很难对齐。
        
        让我们尝试最简单的有效方案：
        不拆分Markdown，而是使用一个特殊的CSS动画，
        它只影响"最新出现"的元素。
        
        但在React中，re-render会重建DOM。
        
        让我们回退到之前的方案，但做一点改进：
        不要让整个容器模糊。
        
        既然用户抱怨"整行文本都是模糊的"，说明之前的实现是：
        useEffect检测到变化 -> 设置 isPulsing=true -> 容器添加 .blur-pulse -> 整个容器模糊。
        
        我们需要的是：
        新内容出现 -> 只有新内容模糊 -> 变清晰。
        
        在Markdown流式渲染中，要完美做到这一点非常困难。
        
        折衷方案：
        使用一个<span>包裹新内容，但这会破坏Markdown语法。
        
        让我们尝试一种"视觉补丁"：
        在Markdown内容末尾追加一个带有模糊效果的<span>，显示新内容，
        但这会导致内容重复（Markdown渲染了一次，我们又渲染了一次）。
        
        最终方案：
        我们必须接受Markdown渲染的限制。
        为了实现Blur Reveal，我们可以只对"纯文本"模式启用该效果，
        或者接受在Markdown语法解析期间（如代码块未闭合时）效果可能不完美。
        
        但为了修复"整行模糊"的问题，我们可以：
        只模糊最后N个字符？
        
        让我们尝试使用 `span` 包装新内容的方法，但只在它不破坏Markdown结构时（例如它是纯文本追加）。
        检测新内容是否包含特殊Markdown字符。
      */}

      {newContent && (
        <span className={`inline-block ${getBlurClass()}`}>
          {/* 这是一个视觉占位，实际上ReactMarkdown已经渲染了这部分内容。
              这会导致重复显示。
              
              让我们换个思路：
              之前的代码是给容器加类。
              现在的代码：
              我们只渲染ReactMarkdown，但在CSS中定义一个动画，
              让所有*新插入*的元素执行这个动画？
              ReactMarkdown在更新时，是更新现有文本节点还是替换节点？
              通常是更新文本节点。
          */}
        </span>
      )}

      {/* 
        重新思考：
        如果我们将 content 分为 stable + new，
        stable 用 ReactMarkdown 渲染，
        new 用 普通 span 渲染（带模糊），
        
        当 new 变为 stable 时，它进入 ReactMarkdown。
        
        风险：如果 new 包含 Markdown 标记（如 `**` 的前半部分），
        它在 span 中会显示为 `**`，
        进入 ReactMarkdown 后变成粗体。
        这会造成视觉跳变。
        
        但对于大多数普通文本（打字机效果），这是可以接受的。
        我们可以检测 newContent 是否包含关键 Markdown 符号，
        如果包含，则立即归入 stable，不显示模糊效果。
      */}
    </div>
  );

  // 实际实现上述逻辑
  const hasMarkdownSymbols = /[*_`[\]()#]/.test(newContent);

  if (hasMarkdownSymbols) {
    // 如果包含Markdown符号，直接渲染全部，不搞特效，避免破坏格式
    return (
      <div className="prose prose-sm max-w-none dark:prose-invert">
        <ReactMarkdown
          remarkPlugins={[remarkMath]}
          rehypePlugins={[rehypeKatex, rehypeHighlight]}
        >
          {content}
        </ReactMarkdown>
      </div>
    );
  }

  return (
    <div className="prose prose-sm max-w-none dark:prose-invert">
      {/* 稳定部分：使用Markdown渲染 */}
      <span className="markdown-body">
        <ReactMarkdown
          remarkPlugins={[remarkMath]}
          rehypePlugins={[rehypeKatex, rehypeHighlight]}
          components={{
            p: ({ node, ...props }) => <span {...props} />, // 强制内联渲染以允许接续
            div: ({ node, ...props }) => <span {...props} />
          }}
        >
          {stableContent}
        </ReactMarkdown>
      </span>

      {/* 新增部分：使用带模糊动画的Span渲染 */}
      {newContent && (
        <span className={`inline-block ${getBlurClass()}`}>
          {newContent}
        </span>
      )}
    </div>
  );
};

export default StreamingMarkdown;
