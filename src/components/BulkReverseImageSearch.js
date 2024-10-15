import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../css/BulkReverseImageSearch.css';

const BulkReverseImageSearch = () => {
  const [imageFiles, setImageFiles] = useState([]);
  const [previewUrls, setPreviewUrls] = useState([]);
  const [isFileSubmitted, setIsFileSubmitted] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (imageFiles.length === 0) {
      console.error('No images submitted');
      return;
    }

    await fakeUploadImages();
    console.log('Images submitted:', imageFiles);
    setIsFileSubmitted(true);
  };

  const fakeUploadImages = () => {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve();
      }, 1000);
    });
  };

  const handleReverseSearch = () => {
    navigate('/reverse-image-results', { state: { imageFiles } });
  };

  const handleFileChange = (event) => {
    const files = Array.from(event.target.files).filter(file =>
      file.type === 'image/png' || file.type === 'image/jpeg'
    );
    if (files.length > 0) {
      setImageFiles(files);
      const urls = files.map(file => URL.createObjectURL(file));
      setPreviewUrls(urls);
    } else {
      console.error('Please upload valid image files (PNG or JPG).');
    }
  };

  const removeImage = (index) => {
    const newImageFiles = [...imageFiles];
    const newPreviewUrls = [...previewUrls];
    newImageFiles.splice(index, 1);
    newPreviewUrls.splice(index, 1);
    setImageFiles(newImageFiles);
    setPreviewUrls(newPreviewUrls);
  };

  return (
    <form onSubmit={handleSubmit}>
      <fieldset>
        <h2 className="title">Bulk Reverse Image Search</h2>
        <div className="dropzone-container" id="dropzone">
          <input
            type="file"
            accept=".png, .jpg, .jpeg"
            multiple
            className="absolute inset-0 w-full h-full opacity-0 z-50"
            onChange={handleFileChange}
          />
          <div className="text-center">
            <img
              className="mx-auto h-12 w-12"
              src="https://www.svgrepo.com/show/357902/image-upload.svg"
              alt="Upload"
            />
            <h3 className="mt-2 text-sm font-medium text-gray-900">
              <label htmlFor="file-upload" className="relative cursor-pointer">
                Drag and drop or browse to upload images
              </label>
            </h3>
          </div>
        </div>
        {previewUrls.length > 0 && (
          <div className="preview-container mt-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {previewUrls.map((url, index) => (
              <div key={index} className="preview-item relative">
                <img src={url} alt={`Preview ${index}`} className="preview-image w-full h-32 object-cover rounded-md" />
                <button
                  type="button"
                  onClick={() => removeImage(index)}
                  className="remove-button absolute top-1 right-1 bg-red-500 text-white rounded-full p-1 text-xs"
                >
                  &times;
                </button>
              </div>
            ))}
          </div>
        )}
        <input type="submit" value="Search" className="submit-button mt-4" />
      </fieldset>
    </form>
  );
};

export default BulkReverseImageSearch;
