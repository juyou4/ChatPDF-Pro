import React from 'react';

const BreatheLoader = ({ className = '' }) => (
  <span className={`ld-breathe ${className}`} aria-hidden="true" />
);

export default BreatheLoader;
