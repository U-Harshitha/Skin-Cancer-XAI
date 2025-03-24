import React, { useState } from 'react';
import axios from 'axios';

const ImageUpload = ({ onPredictionResult }) => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleFileSelect = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
        
        // Create preview
        const reader = new FileReader();
        reader.onloadend = () => setPreview(reader.result);
        reader.readAsDataURL(file);
    };

    const handleSubmit = async () => {
        if (!selectedFile) return;

        setLoading(true);
        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await axios.post('http://localhost:5000/api/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                }
            });
            onPredictionResult(response.data);
        } catch (error) {
            console.error('Error:', error);
            alert(error.response?.data?.error || 'Error processing image');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="image-upload">
            <input
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
            />
            {preview && (
                <div className="preview">
                    <img src={preview} alt="Preview" />
                </div>
            )}
            <button 
                onClick={handleSubmit}
                disabled={!selectedFile || loading}
            >
                {loading ? 'Processing...' : 'Analyze Image'}
            </button>
        </div>
    );
};

export default ImageUpload; 