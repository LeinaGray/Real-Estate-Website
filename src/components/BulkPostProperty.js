import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../css/BulkPostProperty.css';

const RealEstateListingForm = () => {
  const [csvFileName, setCsvFileName] = useState('');
  const [isFileSubmitted, setIsFileSubmitted] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!csvFileName) {
      console.error('No file submitted');
      return;
    }

    await fakeUploadCSV();
    console.log('CSV file submitted:', csvFileName);
    setIsFileSubmitted(true); 
  };

  const fakeUploadCSV = () => {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve();
      }, 1000);
    });
  };

  const handleFraudCheck = () => {
    navigate('/fraud-results', { state: { csvFileName } });
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'text/csv') {
      setCsvFileName(file.name);
    } else {
      console.error('Please upload a valid CSV file.');
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <fieldset>
        <h2 class="title">Real Estate Listing Form</h2>
        <div class="dropzone-container" id="dropzone">
          <input
            type="file"
            accept=".csv"
            class="absolute inset-0 w-full h-full opacity-0 z-50"
            onChange={handleFileChange}
          />
          <div class="text-center">
          <img class="mx-auto h-12 w-12" src="https://www.svgrepo.com/show/357902/image-upload.svg" alt="Upload" />
            <h3 class="mt-2 text-sm font-medium text-gray-900">
              <label htmlFor="file-upload" class="relative cursor-pointer">
                Drag and drop or browse to upload
              </label>
            </h3>
          </div>
          {csvFileName && (
            <div class="mt-4 text-center">
              <p class="text-sm text-gray-700">
                Uploaded File: <strong>{csvFileName}</strong>
              </p>
            </div>
          )}
        </div>
        <input type="submit" value="Submit" class="submit-button" />
        {/* {isFileSubmitted && (
          <button type="button" onClick={handleFraudCheck} class="fraud-check-button">
            Check Fraud Filter
          </button>
        )} */}
      </fieldset>
    </form>
  );
};

export default RealEstateListingForm;
