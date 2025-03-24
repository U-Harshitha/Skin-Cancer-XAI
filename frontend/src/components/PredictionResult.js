import React from 'react';
import HeatmapVisualization from './HeatmapVisualization';

const PredictionResult = ({ predictions, heatmap }) => {
    return (
        <div className="prediction-result">
            <h2>Prediction Results</h2>
            <div className="predictions">
                {predictions.map((pred, index) => (
                    <div key={index} className="prediction-item">
                        <span className="class-name">{pred.class}</span>
                        <div className="probability-bar">
                            <div 
                                className="probability-fill"
                                style={{ width: `${pred.probability * 100}%` }}
                            />
                        </div>
                        <span className="probability-value">
                            {(pred.probability * 100).toFixed(2)}%
                        </span>
                    </div>
                ))}
            </div>
            
            {heatmap && (
                <div className="heatmap">
                    <h3>Model Explanation (Grad-CAM)</h3>
                    <HeatmapVisualization heatmapData={heatmap} />
                </div>
            )}
        </div>
    );
};

export default PredictionResult; 