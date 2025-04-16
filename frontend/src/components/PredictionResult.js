import React from 'react';
import './PredictionResult.css';

function PredictionResult({ predictions, heatmap, shap }) {
  // Sort a copy of predictions to find top 3
  const sorted = [...predictions].sort((a, b) => b.probability - a.probability);
  const topThree = sorted.slice(0, 3).map(p => p.class);

  const getColorClass = (className) => {
    if (className === topThree[0]) return 'bar red';
    if (className === topThree[1]) return 'bar orange';
    if (className === topThree[2]) return 'bar yellow';
    return 'bar default';
  };

  return (
    <div className="result-container">
      <h2 className="result-title">Prediction Results</h2>
      <ul className="prediction-list">
        {predictions.map((pred, index) => {
          const barWidth = `${(pred.probability * 100).toFixed(2)}%`;
          return (
            <li key={index} className="prediction-item">
              <div className="label-line">
                <span className="prediction-class">{pred.class}</span>
                <span className="prediction-probability">{(pred.probability * 100).toFixed(2)}%</span>
              </div>
              <div className="bar-container">
                <div className={getColorClass(pred.class)} style={{ width: barWidth }} />
              </div>
            </li>
          );
        })}
      </ul>

      {heatmap && (
        <div className="image-container">
          <h3>Grad-CAM Heatmap</h3>
          <img src={`data:image/png;base64,${heatmap}`} alt="Grad-CAM" className="result-image" />
        </div>
      )}

      {shap && (
        <div className="image-container">
          <h3>SHAP Explanation</h3>
          <img src={`data:image/png;base64,${shap}`} alt="SHAP" className="result-image" />
        </div>
      )}
    </div>
  );
}

export default PredictionResult;
