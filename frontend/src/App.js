import React, { useState, useEffect } from 'react';
import './App.css';
import ImageUpload from './components/ImageUpload';
import PredictionResult from './components/PredictionResult';
import HeatmapVisualization from './components/HeatmapVisualization';
import ReactMarkdown from 'react-markdown';

function App() {
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState('');
  const [patientInfo, setPatientInfo] = useState({
    age: '',
    sex: '',
    localization: ''
  });
  const [isFormValid, setIsFormValid] = useState(false);
  const [imageSelected, setImageSelected] = useState(false);
  const handlePredictionResult = (data) => {
    setResult(data);
  };
  const handleFormChange = (e) => {
    const { name, value } = e.target;
    setPatientInfo((prev) => ({ ...prev, [name]: value }));
  };
  useEffect(() => {
    const { age, sex, localization } = patientInfo;
    setIsFormValid(
      age >= 2 && age <= 100 &&
      sex !== '' &&
      localization.trim() !== ''
    );
  }, [patientInfo]);
  // // Load puter.ai script dynamically
  // const fetchExplanation = async (predictions, explanations) => {
  //   try {
  //     if (!window.puter || !window.puter.ai) {
  //       setExplanation("Grok API not loaded.");
  //       return;
  //     }
  
  //     const prompt = `
  // The AI model predicted the following skin cancer types: ${JSON.stringify(predictions)}.
  // Based on Grad-CAM or saliency map explanations: ${JSON.stringify(explanations)},
  // provide a short medical-style explanation of the key visual features (e.g., color, border irregularity, asymmetry),
  // and what possible risk factors could be inferred.
  //     `;
  
  //     // 🚫 Removed the 'model' property
  //     const response = await window.puter.ai.chat(prompt);
  
  //     setExplanation(response.message.content);
  //   } catch (error) {
  //     console.error("Grok API error:", error);
  //     setExplanation("An error occurred while generating the explanation.");
  //   }
  // };
  const fetchExplanation = async (predictions, explanations) => {
    try {
      const prompt = `
  The AI model predicted the following skin cancer types: ${JSON.stringify(predictions)}.
  Make the explanation concise and medically informative.
      `;
      // Based on the Grad-CAM or saliency map explanations: ${JSON.stringify(explanations)},
      // explain the visual cues (like color, border, texture) and possible underlying risk factors for this condition.
      const response = await window.puter.ai.chat(prompt, {
        model: "x-ai/grok-3-beta"
      });
  
      setExplanation(response.message.content);
    } catch (error) {
      console.error("Grok API error:", error);
      setExplanation("An error occurred while generating the explanation.");
    }
  };
  
  useEffect(() => {
    if (result?.predictions && result?.explanations) {
      fetchExplanation(result.predictions, result.explanations);
    }
  }, [result]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Skin Cancer Detection System</h1>
        <p>Upload an image and fill the form for analysis</p>
      </header>
      <main className="App-main">
        {/* <ImageUpload onPredictionResult={handlePredictionResult} />

        <form className="patient-info-form">
          <h3>Patient Information</h3>
          <label>
            Age:
            <input type="number" name="age" placeholder="Enter age" />
          </label>
          <label>
            Sex:
            <select name="sex">
              <option value="">Select</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
          </label>
          <label>
            Localization:
            <input type="text" name="localization" placeholder="e.g., back, face" />
          </label>
        </form> */}
        <form className="patient-info-form">
          <h3>Patient Information</h3>
          <label>
            Age:
            <input 
              type="number" 
              name="age" 
              placeholder="Enter age" 
              value={patientInfo.age} 
              min="2" 
              max="100" 
              onChange={handleFormChange} 
            />
          </label>
          <label>
            Sex:
            <select name="sex" value={patientInfo.sex} onChange={handleFormChange}>
              <option value="">Select</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
          </label>
          <label>
            Localization:
            <input 
              type="text" 
              name="localization" 
              placeholder="e.g., back, face" 
              value={patientInfo.localization}
              onChange={handleFormChange}
            />
          </label>
        </form>
        <ImageUpload 
          onPredictionResult={handlePredictionResult} 
          formValid={isFormValid} 
          onImageSelect={() => setImageSelected(true)} 
        />

        {result && (
          <>
            <PredictionResult predictions={result.predictions} />
            <HeatmapVisualization explanations={result.explanations} />
            <div className="explanation-section">
              <h3>AI Explanation</h3>
              <ReactMarkdown>
                {explanation || 'Generating explanation...'}
              </ReactMarkdown>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
