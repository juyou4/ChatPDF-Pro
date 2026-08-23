import { motion, useReducedMotion } from 'framer-motion';
import { useEffect, useMemo, useRef, useState } from 'react';

const buildKeyframes = (from, steps) => {
  const keys = new Set([
    ...Object.keys(from),
    ...steps.flatMap((step) => Object.keys(step)),
  ]);

  const keyframes = {};
  keys.forEach((key) => {
    keyframes[key] = [from[key], ...steps.map((step) => step[key])];
  });
  return keyframes;
};

const splitLetters = (text) => {
  if (typeof Intl !== 'undefined' && typeof Intl.Segmenter === 'function') {
    const segmenter = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
    return Array.from(segmenter.segment(text), ({ segment }) => segment);
  }
  return Array.from(text);
};

const BlurText = ({
  text = '',
  delay = 200,
  className = '',
  animateBy = 'words',
  direction = 'top',
  threshold = 0.1,
  rootMargin = '0px',
  animationFrom,
  animationTo,
  easing = (value) => value,
  onAnimationComplete,
  stepDuration = 0.35,
  maxDelay = Number.POSITIVE_INFINITY,
}) => {
  const elements = useMemo(
    () => (animateBy === 'words' ? text.split(' ') : splitLetters(text)),
    [animateBy, text],
  );
  const [inView, setInView] = useState(false);
  const ref = useRef(null);
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    if (reduceMotion) {
      setInView(true);
      return undefined;
    }
    if (!ref.current) return undefined;
    if (typeof IntersectionObserver !== 'function') {
      setInView(true);
      return undefined;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          observer.unobserve(entry.target);
        }
      },
      { threshold, rootMargin },
    );
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [reduceMotion, rootMargin, threshold]);

  const defaultFrom = useMemo(
    () => (direction === 'top'
      ? { filter: 'blur(10px)', opacity: 0, y: -50 }
      : { filter: 'blur(10px)', opacity: 0, y: 50 }),
    [direction],
  );

  const defaultTo = useMemo(
    () => [
      {
        filter: 'blur(5px)',
        opacity: 0.5,
        y: direction === 'top' ? 5 : -5,
      },
      { filter: 'blur(0px)', opacity: 1, y: 0 },
    ],
    [direction],
  );

  const fromSnapshot = animationFrom ?? defaultFrom;
  const toSnapshots = animationTo ?? defaultTo;
  const animateKeyframes = useMemo(
    () => buildKeyframes(fromSnapshot, toSnapshots),
    [fromSnapshot, toSnapshots],
  );
  const stepCount = toSnapshots.length + 1;
  const totalDuration = stepDuration * (stepCount - 1);
  const times = Array.from(
    { length: stepCount },
    (_, index) => (stepCount === 1 ? 0 : index / (stepCount - 1)),
  );

  return (
    <p
      ref={ref}
      className={className}
      aria-label={text}
      style={{ display: 'flex', flexWrap: 'wrap' }}
    >
      {elements.map((segment, index) => {
        const spanTransition = reduceMotion
          ? { duration: 0 }
          : {
              duration: totalDuration,
              times,
              delay: Math.min((index * delay) / 1000, maxDelay / 1000),
              ease: easing,
            };

        return (
          <motion.span
            aria-hidden="true"
            className="inline-block will-change-[transform,filter,opacity]"
            key={`${index}-${segment}`}
            initial={reduceMotion ? false : fromSnapshot}
            animate={inView || reduceMotion ? animateKeyframes : fromSnapshot}
            transition={spanTransition}
            onAnimationComplete={index === elements.length - 1 ? onAnimationComplete : undefined}
          >
            {segment === ' ' ? '\u00A0' : segment}
            {animateBy === 'words' && index < elements.length - 1 && '\u00A0'}
          </motion.span>
        );
      })}
    </p>
  );
};

export default BlurText;
