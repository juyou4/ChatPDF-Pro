        const processSseEvent = (eventText) => {
          // eventText 为不含末尾空行的单个 event
          const lines = eventText.split(/\r?\n/);
          const dataLines = [];
          for (const ln of lines) {
            // 使用正则匹配前缀，忽略可能的行首空格，并容忍没有空格的 data:
            const match = ln.match(/^\s*data:\s*(.*)$/);
            if (match) {
              dataLines.push(match[1]);
            }
          }
          if (dataLines.length === 0) return;

          // SSE 允许一个 event 多行 data:，拼接时用 \n
          const data = dataLines.join('\n');
          if (data === '[DONE]') {
            sseDone = true;
            return;
          }

          try {
            const parsed = JSON.parse(data);
            if (parsed.error) {
              throw new Error(parsed.error);
            }

            // 后端可能会插入检索进度事件（非 content/done 结构），这里忽略
            if (parsed.type === 'retrieval_progress') {
              return;
            }

            // 兼容 OpenAI 格式 (choices[0].delta) 和自定义格式 (content/reasoning_content)
            const delta = parsed.choices?.[0]?.delta || {};
            const chunkContent = delta.content || parsed.content || '';
            const chunkThinking = delta.reasoning_content || parsed.reasoning_content || '';

            // 核心修复：无论是否结束标志，只要 chunk 里有内容就必须追加，防止 finish_reason 丢失最后一个 token
            if (chunkContent) {
              currentText += chunkContent;
              contentStream.addChunk(chunkContent);
              if (thinkingStartTime && !thinkingEndTime) {
                thinkingEndTime = Date.now();
              }
            }
            if (chunkThinking) {
              if (!thinkingStartTime) thinkingStartTime = Date.now();
              currentThinking += chunkThinking;
              thinkingStream.addChunk(chunkThinking);
            }

            // 判定是否结束：解耦提取与终止逻辑
            if (parsed.done || parsed.choices?.[0]?.finish_reason) {
              if (parsed.retrieval_meta && parsed.retrieval_meta.citations) {
                streamCitationsRef.current = parsed.retrieval_meta.citations;
              }
              sseDone = true;
            }
          } catch (e) {
            console.error('SSE解析失败:', e, data);
          }
        };