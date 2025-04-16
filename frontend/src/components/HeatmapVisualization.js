import React from 'react';
import '../components/HeatmapVisulization.css'; // optional, if you want custom styling

const HeatmapVisualization = ({ explanations }) => {
  return (
    <div className="visualization-container">
      {/* Grad-CAM */}
      {explanations?.visualization && (
        <div className="explanation-section">
          <h3>Grad-CAM Explanation</h3>
          <img
            src={`data:image/png;base64,${explanations.visualization}`}
            alt="Grad-CAM Explanation"
            className="explanation-visualization"
          />
        </div>
      )}

      {/* SHAP
      {explanations?.shap ? (
  <div className="explanation-section">
    <h3>SHAP Explanation</h3>
    <img
      src={`data:image/png;base64,${explanations.shap}`}
      alt="SHAP Explanation"
      className="explanation-visualization"
    />
  </div>
) : (
  <div className="explanation-section">
    <h3>SHAP Explanation not available</h3>
  </div>
)} */}
{explanations?.saliency && (
  <div className="explanation-section">
    <h3>Saliency Map Explanation</h3>
    <img
      src={`data:image/png;base64,${explanations.saliency}`}
      alt="Saliency Map"
      className="explanation-visualization"
    />
  </div>
)}


    </div>
  );
};

export default HeatmapVisualization;
