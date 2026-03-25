export const shouldStreamAssistantContent = (message, streamingMessageId) =>
  Boolean(message?.isStreaming && streamingMessageId === message?.id);
