import React from 'react';

function PieChart(props) {
  const { percentage, color, borderWidth } = props;

  return (
    <div
      className="pie animate"
      style={{
        '--p': `${percentage}%`,
        '--c': color,
        '--b': `${borderWidth}px`,
      }}
    >
      {percentage}%
    </div>
  );
}

export default PieChart;