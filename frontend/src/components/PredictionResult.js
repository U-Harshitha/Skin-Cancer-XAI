import React from 'react';

function PredictionResult({ predictions, heatmap, shap }) {
  return (
    <div className="result-container">
      <h2>Prediction Results</h2>
      <ul>
        {predictions.map((pred, index) => (
          <li key={index}>
            <strong>{pred.class}</strong>: {(pred.probability * 100).toFixed(2)}%
          </li>
        ))}
      </ul>

      
      {heatmap && (
        <img 
          src={`data:image/png;base64,${heatmap}`} 
          alt="Grad-CAM" 
          style={{ maxWidth: '100%', borderRadius: '8px', marginTop: '10px' }}
        />
      )}

      
      {shap && (
        
        <img 
          src={`data:image/png;base64,${shap}`} 
          alt="SHAP" 
          style={{ maxWidth: '100%', borderRadius: '8px', marginTop: '10px' }}
        />
      )}
   
    </div>
  );
}

export default PredictionResult;
