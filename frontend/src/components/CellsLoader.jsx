import React from 'react';

const CellsLoader = ({ active = true, className = '' }) => (
  <span className={`ld-cells ${className}`} data-active={active ? 'true' : 'false'} aria-hidden="true">
    {Array.from({ length: 9 }, (_, index) => <i key={index} />)}
  </span>
);

export default CellsLoader;
