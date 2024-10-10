import React, { useState } from 'react';
import '../css/BulkPostProperty.css';

const RealEstateListingForm = () => {
  const [csvFileName, setCsvFileName] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    // Handle form submission here
    console.log('CSV file submitted:', csvFileName);
  };

  // File handling for CSV
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'text/csv') {
      const reader = new FileReader();
      reader.readAsText(file);
      reader.onload = () => {
        console.log('CSV file content:', reader.result);
      };
      setCsvFileName(file.name); // Set the file name to display
    } else {
      console.error('Please upload a valid CSV file.');
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <fieldset>
        {/* Dropzone Section */}
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
                <span class="span-container">Drag and drop</span>
                <span class="span-container text-indigo-600"> or browse</span>
                <span class="span-container"> to upload</span>
              </label>
            </h3>
          </div>

          {/* Display the CSV file name */}
          {csvFileName && (
            <div className="mt-4 text-center">
              <p className="text-sm text-gray-700">Uploaded File: <strong>{csvFileName}</strong></p>
            </div>
          )}
        </div>
        <input type="submit" value="Submit" />
      </fieldset>
    </form>
  );
};

export default RealEstateListingForm;
