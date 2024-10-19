import React from 'react';
import PieChart from './css/component-styles/PieChart.css'; // Assuming PieChart.js is in the same directory

function App() {
  return (
    <div>
      <PieChart percentage={75} color="green" borderWidth={5} />
    </div>
  );
}

export default App;