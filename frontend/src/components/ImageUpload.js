import React, { useState } from 'react';
import axios from 'axios';
import './ImageUpload.css'; // Add this line to import the CSS

const ImageUpload = ({ onPredictionResult, formValid, onImageSelect }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setError(null);
    onImageSelect();

    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile || !formValid) return;

    setLoading(true);
    setError(null);
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
      setError(
        error.response?.data?.error || 
        error.response?.data?.details || 
        'Error processing image'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="image-upload">
      <h3>Upload Image</h3>
      <input type="file" accept="image/*" onChange={handleFileSelect} />
      
      {preview && <img className="preview" src={preview} alt="Preview" />}
      
      {error && <div className="error-message">{error}</div>}

      <button 
        className={`analyze-button ${selectedFile && formValid ? 'active' : ''}`}
        onClick={handleSubmit}
        disabled={!formValid || !selectedFile || loading}
      >
        {loading ? 'Processing...' : 'Analyze Image'}
      </button>
    </div>
  );
};

export default ImageUpload;
