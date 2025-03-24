import React, { useEffect, useRef } from 'react';

const HeatmapVisualization = ({ heatmapData, originalImage }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        if (!canvasRef.current || !heatmapData) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        // Set canvas dimensions
        canvas.width = 224;
        canvas.height = 224;

        // Create heatmap visualization
        const heatmapArray = new Float32Array(heatmapData.flat());
        const colorScale = (value) => {
            const hue = ((1 - value) * 240).toString(10);
            return `hsla(${hue}, 100%, 50%, 0.6)`;
        };

        // Draw heatmap
        for (let i = 0; i < heatmapArray.length; i++) {
            const x = i % 224;
            const y = Math.floor(i / 224);
            const value = heatmapArray[i];
            
            ctx.fillStyle = colorScale(value);
            ctx.fillRect(x, y, 1, 1);
        }
    }, [heatmapData]);

    return (
        <div className="heatmap-container">
            <canvas ref={canvasRef} className="heatmap-canvas" />
        </div>
    );
};

export default HeatmapVisualization; 