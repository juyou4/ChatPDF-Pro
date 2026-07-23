import React, { useEffect, useRef } from 'react';

const THUMB_WIDTH = 34;

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const getRangeGeometry = (value, min, max) => {
  const span = max - min;
  const progress = span > 0 ? clamp(((value - min) / span) * 100, 0, 100) : 0;
  const fillOffset = THUMB_WIDTH * (1 - progress / 100);
  return {
    progress,
    width: `calc(${progress}% + ${fillOffset}px)`,
  };
};

export default function SettingsRange({
  ariaLabel,
  className = '',
  disabled = false,
  max = 100,
  min = 0,
  onChange,
  step = 1,
  value,
}) {
  const numericMin = Number(min);
  const numericMax = Number(max);
  const numericValue = Number(value);
  const safeValue = Number.isFinite(numericValue)
    ? clamp(numericValue, numericMin, numericMax)
    : numericMin;
  const initialGeometry = getRangeGeometry(safeValue, numericMin, numericMax);
  const inputRef = useRef(null);
  const fillRef = useRef(null);
  const draggingRef = useRef(false);
  const latestValueRef = useRef(safeValue);
  const committedValueRef = useRef(safeValue);

  useEffect(() => {
    if (draggingRef.current) return;
    latestValueRef.current = safeValue;
    committedValueRef.current = safeValue;
    if (inputRef.current) inputRef.current.value = String(safeValue);
    if (fillRef.current) {
      fillRef.current.style.width = getRangeGeometry(safeValue, numericMin, numericMax).width;
    }
  }, [numericMax, numericMin, safeValue]);

  const updateVisualValue = (nextValue) => {
    if (!Number.isFinite(nextValue)) return;
    const normalizedValue = clamp(nextValue, numericMin, numericMax);
    latestValueRef.current = normalizedValue;
    if (fillRef.current) {
      fillRef.current.style.width = getRangeGeometry(normalizedValue, numericMin, numericMax).width;
    }
  };

  const commitValue = () => {
    const nextValue = latestValueRef.current;
    if (Object.is(nextValue, committedValueRef.current)) return;
    committedValueRef.current = nextValue;
    onChange?.(nextValue);
  };

  const finishPointerInteraction = () => {
    draggingRef.current = false;
    commitValue();
  };

  return (
    <div className={`settings-range ${disabled ? 'settings-range-disabled' : ''} ${className}`.trim()}>
      <span
        ref={fillRef}
        className="settings-range-fill"
        style={{ width: initialGeometry.width }}
        aria-hidden="true"
      />
      <input
        ref={inputRef}
        type="range"
        min={min}
        max={max}
        step={step}
        defaultValue={safeValue}
        disabled={disabled}
        onInput={(event) => {
          updateVisualValue(Number.parseFloat(event.currentTarget.value));
          if (!draggingRef.current) commitValue();
        }}
        onPointerDown={() => {
          draggingRef.current = true;
        }}
        onPointerUp={finishPointerInteraction}
        onPointerCancel={finishPointerInteraction}
        onBlur={finishPointerInteraction}
        onKeyUp={commitValue}
        className="settings-range-input"
        aria-label={ariaLabel}
      />
    </div>
  );
}
