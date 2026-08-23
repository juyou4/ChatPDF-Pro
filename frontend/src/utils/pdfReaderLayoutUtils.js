export const PDF_READER_FLOW_MODES = Object.freeze({
  PAGED: 'paged',
  CONTINUOUS: 'continuous',
});

export const PDF_READER_FLOW_MODE_STORAGE_KEY = 'chatpdf_pdf_reader_flow_mode_v1';

export function normalizePdfReaderFlowMode(value) {
  return value === PDF_READER_FLOW_MODES.CONTINUOUS
    ? PDF_READER_FLOW_MODES.CONTINUOUS
    : PDF_READER_FLOW_MODES.PAGED;
}

export function readPdfReaderFlowMode(storage = globalThis.localStorage) {
  try {
    return normalizePdfReaderFlowMode(storage?.getItem?.(PDF_READER_FLOW_MODE_STORAGE_KEY));
  } catch {
    return PDF_READER_FLOW_MODES.PAGED;
  }
}

export function writePdfReaderFlowMode(value, storage = globalThis.localStorage) {
  const next = normalizePdfReaderFlowMode(value);
  try {
    storage?.setItem?.(PDF_READER_FLOW_MODE_STORAGE_KEY, next);
  } catch {
    // 隐私模式或配额满时仍返回规范化后的值，当前会话可以继续用。
  }
  return next;
}

export const PDF_READER_LAYOUTS = Object.freeze({
  SINGLE: 'single',
  DOUBLE: 'double',
  COVER: 'cover',
});

const clampPageNumber = (value, totalPages) => {
  const total = Math.max(1, Math.floor(Number(totalPages) || 1));
  return Math.min(total, Math.max(1, Math.floor(Number(value) || 1)));
};

const getKnownTotalPages = (totalPages) => {
  const total = Math.floor(Number(totalPages));
  return Number.isFinite(total) && total > 0 ? total : null;
};

export const normalizePdfReaderRotation = (value) => {
  const normalized = ((Math.round(Number(value) || 0) % 360) + 360) % 360;
  return [0, 90, 180, 270].includes(normalized) ? normalized : 0;
};

export const rotatePdfReader = (rotation, direction = 1) => (
  normalizePdfReaderRotation(normalizePdfReaderRotation(rotation) + Number(direction || 0) * 90)
);

export const getPdfReaderSpreadStart = ({
  totalPages,
  pageNumber,
  layout = PDF_READER_LAYOUTS.SINGLE,
} = {}) => {
  const total = getKnownTotalPages(totalPages);
  const current = total
    ? clampPageNumber(pageNumber, total)
    : Math.max(1, Math.floor(Number(pageNumber) || 1));

  if (layout === PDF_READER_LAYOUTS.DOUBLE) {
    return Math.floor((current - 1) / 2) * 2 + 1;
  }

  if (layout === PDF_READER_LAYOUTS.COVER && current > 1) {
    return current % 2 === 0 ? current : current - 1;
  }

  return current;
};

export const getPdfReaderDisplayPages = ({
  totalPages,
  pageNumber,
  flowMode = PDF_READER_FLOW_MODES.PAGED,
  layout = PDF_READER_LAYOUTS.SINGLE,
} = {}) => {
  const total = getKnownTotalPages(totalPages);
  const current = total
    ? clampPageNumber(pageNumber, total)
    : Math.max(1, Math.floor(Number(pageNumber) || 1));

  if (!total) return [current];

  if (flowMode === PDF_READER_FLOW_MODES.CONTINUOUS) {
    return Array.from({ length: total }, (_, index) => index + 1);
  }

  if (layout === PDF_READER_LAYOUTS.SINGLE) return [current];

  const spreadStart = getPdfReaderSpreadStart({ totalPages: total, pageNumber: current, layout });
  if (layout === PDF_READER_LAYOUTS.COVER && spreadStart === 1) return [1];
  return [spreadStart, spreadStart + 1].filter((item) => item <= total);
};

export const getPdfReaderNavigationTarget = ({
  totalPages,
  pageNumber,
  flowMode = PDF_READER_FLOW_MODES.PAGED,
  layout = PDF_READER_LAYOUTS.SINGLE,
  direction = 1,
} = {}) => {
  const total = getKnownTotalPages(totalPages);
  const current = total
    ? clampPageNumber(pageNumber, total)
    : Math.max(1, Math.floor(Number(pageNumber) || 1));
  const stepDirection = Number(direction) < 0 ? -1 : 1;

  if (!total) return current;

  if (flowMode === PDF_READER_FLOW_MODES.CONTINUOUS || layout === PDF_READER_LAYOUTS.SINGLE) {
    return clampPageNumber(current + stepDirection, total);
  }

  const spreadStart = getPdfReaderSpreadStart({ totalPages: total, pageNumber: current, layout });
  const target = stepDirection > 0
    ? (layout === PDF_READER_LAYOUTS.COVER && spreadStart === 1 ? 2 : spreadStart + 2)
    : (layout === PDF_READER_LAYOUTS.COVER && spreadStart === 2 ? 1 : spreadStart - 2);

  if (target < 1 || target > total) {
    return current;
  }

  return target;
};

export const getPdfReaderRotationTransform = ({ rotation, width, height } = {}) => {
  const normalizedRotation = normalizePdfReaderRotation(rotation);
  const safeWidth = Math.max(0, Number(width) || 0);
  const safeHeight = Math.max(0, Number(height) || 0);

  if (normalizedRotation === 0 || !safeWidth || !safeHeight) {
    return {
      rotation: normalizedRotation,
      stageWidth: safeWidth,
      stageHeight: safeHeight,
      transform: undefined,
    };
  }

  if (normalizedRotation === 90) {
    return {
      rotation: normalizedRotation,
      stageWidth: safeHeight,
      stageHeight: safeWidth,
      transform: `translateX(${safeHeight}px) rotate(90deg)`,
    };
  }

  if (normalizedRotation === 180) {
    return {
      rotation: normalizedRotation,
      stageWidth: safeWidth,
      stageHeight: safeHeight,
      transform: `translate(${safeWidth}px, ${safeHeight}px) rotate(180deg)`,
    };
  }

  return {
    rotation: normalizedRotation,
    stageWidth: safeHeight,
    stageHeight: safeWidth,
    transform: `translateY(${safeWidth}px) rotate(270deg)`,
  };
};

export const mapPdfReaderDisplayPointToPage = ({
  x,
  y,
  width,
  height,
  rotation,
} = {}) => {
  const safeX = Number(x) || 0;
  const safeY = Number(y) || 0;
  const safeWidth = Math.max(0, Number(width) || 0);
  const safeHeight = Math.max(0, Number(height) || 0);

  switch (normalizePdfReaderRotation(rotation)) {
    case 90:
      return { x: safeY, y: safeHeight - safeX };
    case 180:
      return { x: safeWidth - safeX, y: safeHeight - safeY };
    case 270:
      return { x: safeWidth - safeY, y: safeX };
    default:
      return { x: safeX, y: safeY };
  }
};

export const mapPdfReaderDisplayRectToPage = ({
  left,
  top,
  width,
  height,
  pageWidth,
  pageHeight,
  rotation,
} = {}) => {
  const safeLeft = Number(left) || 0;
  const safeTop = Number(top) || 0;
  const safeWidth = Math.max(0, Number(width) || 0);
  const safeHeight = Math.max(0, Number(height) || 0);
  const safePageWidth = Math.max(0, Number(pageWidth) || 0);
  const safePageHeight = Math.max(0, Number(pageHeight) || 0);

  switch (normalizePdfReaderRotation(rotation)) {
    case 90:
      return {
        left: safeTop,
        top: safePageHeight - safeLeft - safeWidth,
        width: safeHeight,
        height: safeWidth,
      };
    case 180:
      return {
        left: safePageWidth - safeLeft - safeWidth,
        top: safePageHeight - safeTop - safeHeight,
        width: safeWidth,
        height: safeHeight,
      };
    case 270:
      return {
        left: safePageWidth - safeTop - safeHeight,
        top: safeLeft,
        width: safeHeight,
        height: safeWidth,
      };
    default:
      return {
        left: safeLeft,
        top: safeTop,
        width: safeWidth,
        height: safeHeight,
      };
  }
};

export const mapPdfReaderPageRectToDisplay = ({
  left,
  top,
  width,
  height,
  pageWidth,
  pageHeight,
  rotation,
} = {}) => {
  const safeLeft = Number(left) || 0;
  const safeTop = Number(top) || 0;
  const safeWidth = Math.max(0, Number(width) || 0);
  const safeHeight = Math.max(0, Number(height) || 0);
  const safePageWidth = Math.max(0, Number(pageWidth) || 0);
  const safePageHeight = Math.max(0, Number(pageHeight) || 0);

  switch (normalizePdfReaderRotation(rotation)) {
    case 90:
      return {
        left: safePageHeight - safeTop - safeHeight,
        top: safeLeft,
        width: safeHeight,
        height: safeWidth,
      };
    case 180:
      return {
        left: safePageWidth - safeLeft - safeWidth,
        top: safePageHeight - safeTop - safeHeight,
        width: safeWidth,
        height: safeHeight,
      };
    case 270:
      return {
        left: safeTop,
        top: safePageWidth - safeLeft - safeWidth,
        width: safeHeight,
        height: safeWidth,
      };
    default:
      return {
        left: safeLeft,
        top: safeTop,
        width: safeWidth,
        height: safeHeight,
      };
  }
};
