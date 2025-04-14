import React from 'react';

const PredictionResult = ({ predictions, explanations }) => {
    const getPredictionExplanation = (className, probability) => {
        const percentage = (probability * 100).toFixed(2);
        
        if (percentage > 80) {
            return `High confidence (${percentage}%) prediction of ${className}. The model shows strong indicators in the highlighted regions.`;
        } else if (percentage > 50) {
            return `Moderate confidence (${percentage}%) prediction of ${className}. While there are indicators present, further medical consultation is recommended.`;
        } else {
            return `Low confidence (${percentage}%) prediction of ${className}. The indicators are not strong, and this should be treated as a preliminary assessment only.`;
        }
    };

    const getRiskLevel = (probability) => {
        if (probability > 0.8) return 'high-risk';
        if (probability > 0.5) return 'medium-risk';
        return 'low-risk';
    };

    return (
        <div className="prediction-result">
            <h2>Analysis Results</h2>
            <div className="predictions">
                {predictions.map((pred, index) => (
                    <div key={index} className={`prediction-item ${getRiskLevel(pred.probability)}`}>
                        <div className="prediction-header">
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
                        <div className="prediction-explanation">
                            {getPredictionExplanation(pred.class, pred.probability)}
                        </div>
                    </div>
                ))}
            </div>
            
            {explanations && (
                <div className="visualization-section">
                    <h3>Visual Explanation</h3>
                    <p>The heatmap below shows which regions of the image were most important for the model's prediction. 
                       Warmer colors (red) indicate areas that strongly influenced the prediction, while cooler colors (blue) 
                       had less influence.</p>
                    <HeatmapVisualization explanations={explanations} />
                </div>
            )}
        </div>
    );
};

export default PredictionResult;
