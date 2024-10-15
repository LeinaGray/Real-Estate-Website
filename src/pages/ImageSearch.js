//src\pages\ImageSearch.js

import React, { useState } from "react";

const ImageSearch = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [results, setResults] = useState([]);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    setSelectedImage(file);
  };

  const handleSearch = async () => {
    if (!selectedImage) return;

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
        const response = await fetch('http://127.0.0.1:5000/api/search-by-image', { // Changed to 5000
            method: 'POST',
            body: formData,
        });

        const resultData = await response.json();
        setResults(resultData.similarImages);
    } catch (error) {
        console.error("Error searching by image:", error);
    }
  };


  return (
    <div className="image-search-container">
      <h1>Search Properties by Image</h1>
      <input type="file" onChange={handleImageUpload} accept="image/*" />
      <button onClick={handleSearch} className="bg-violet-600 text-white px-4 py-2 rounded-lg">
        Search by Image
      </button>

      {/* Display search results */}
      <div className="results-container">
      {results.map((image, index) => (
        <div key={index}>
          <img src={`http://127.0.0.1:5000/images/${image.url}`} alt={`Similar result ${index}`} />
          <p>Similarity: {image.similarity}</p>
        </div>
      ))}
      </div>
    </div>
  );
};

export default ImageSearch;
