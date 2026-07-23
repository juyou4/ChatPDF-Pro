import React from 'react';

const MinerUScanLoader = ({ size = 24, className = '' }) => {
  const normalizedSize = Math.max(14, Math.round(Number(size) || 24));

  return (
    <span
      aria-hidden="true"
      data-testid="mineru-scan-loader"
      className={`mineru-scan-loader ${className}`}
      style={{ '--mineru-scan-size': `${normalizedSize}px` }}
    />
  );
};

export default MinerUScanLoader;
