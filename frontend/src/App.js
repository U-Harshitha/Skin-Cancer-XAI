import React, { useState } from 'react';
import './App.css';
import ImageUpload from './components/ImageUpload';
import PredictionResult from './components/PredictionResult';
import HeatmapVisualization from './components/HeatmapVisualization';

function App() {
  const [result, setResult] = useState(null);

  const handlePredictionResult = (data) => {
    setResult(data);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Skin Cancer Detection System</h1>
        <p>Upload an image for analysis</p>
      </header>
      <main className="App-main">
        <ImageUpload onPredictionResult={handlePredictionResult} />
        {result && (
          <>
            <PredictionResult predictions={result.predictions} />
            <HeatmapVisualization explanations={result.explanations} />
          </>
        )}
      </main>
    </div>
  );
}

export default App;
