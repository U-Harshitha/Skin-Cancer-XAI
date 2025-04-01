import React from 'react';

const HeatmapVisualization = ({ explanations }) => {
    return (
        <div className="visualization-container">
            <div className="explanation-section">
                <h3>Model Explanation (Grad-CAM)</h3>
                {explanations?.visualization && (
                    <img 
                        src={`data:image/png;base64,${explanations.visualization}`}
                        alt="Grad-CAM explanation"
                        className="explanation-visualization"
                    />
                )}
            </div>
        </div>
    );
};

export default HeatmapVisualization; 