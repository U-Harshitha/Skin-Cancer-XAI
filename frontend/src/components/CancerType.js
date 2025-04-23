import React from 'react';
import { ALL_SKIN_CANCER_TYPES } from '../constants';
import './CancerType.css'; // Optional: for styling

function CancerTypeList({ predictedTypes = [] }) {
  return (
    <div className="cancer-type-list">
      <h3>All Skin Cancer Types</h3>
      <ul>
        {ALL_SKIN_CANCER_TYPES.map((type) => (
          <li 
            key={type} 
            className={predictedTypes.includes(type) ? 'highlighted' : ''}
          >
            {type}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default CancerTypeList;
