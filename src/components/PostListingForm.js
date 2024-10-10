import React, { useState } from 'react';
import '../css/PostListingForm.css';

const RealEstateListingForm = () => {
  const [title, setTitle] = useState('');
  const [propertyType, setPropertyType] = useState('');
  const [price, setPrice] = useState('');
  const [address, setAddress] = useState('');
  const [numBedrooms, setNumBedrooms] = useState('');
  const [numBathrooms, setNumBathrooms] = useState('');
  const [floorArea, setFloorArea] = useState('');
  const [description, setDescription] = useState('');
  const [amenities, setAmenities] = useState('');
  const [images, setImages] = useState([]);

  const handleSubmit = (e) => {
    e.preventDefault();
    // Handle form submission here
    console.log('Form submitted:', {
      title,
      propertyType,
      price,
      address,
      numBedrooms,
      numBathrooms,
      floorArea,
      description,
      amenities,
      images,
    });
  };

  // ----------handle file
  const handleFileUpload = (file) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      setImages((prevImages) => [...prevImages, reader.result]);
    };
  };

  // Drag-and-drop event handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    e.target.classList.add('border-indigo-600');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.target.classList.remove('border-indigo-600');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.target.classList.remove('border-indigo-600');
    const file = e.dataTransfer.files[0];
    handleFileUpload(file);
  };

  const handleFileChange = (event) => {
    const files = Array.from(event.target.files);
    const newImages = files.map(file => URL.createObjectURL(file));
  
    setImages(prevImages => [...prevImages, ...newImages]);
  };
  
  //--------------------------

  return (
    <form onSubmit={handleSubmit}>
      <fieldset>
        <h2 class="title">Real Estate Listing Form</h2>
        <label htmlFor="title">Title:</label>
        <input type="text" id="title" name="title" value={title} onChange={(e) => setTitle(e.target.value)} /><br /><br />
        <label htmlFor="propertyType">Property Type:</label>
        <select id="propertyType" name="propertyType" value={propertyType} onChange={(e) => setPropertyType(e.target.value)}>
            <option value="">Select Property Type</option>
            <option value="House">House</option>
            <option value="House">House and Lot</option>
            <option value="House">Lot</option>
            <option value="Condo">Condo</option>
            <option value="Apartment">Apartment</option>
            <option value="Other">Other</option>
        </select><br /><br />
        <label htmlFor="price">Price:</label>
        <input type="number" id="price" name="price" min="1" value={price} onChange={(e) => setPrice(e.target.value)} /><br /><br />
        <label htmlFor="address">Address:</label>
        <input type="text" id="address" name="address" value={address} onChange={(e) => setAddress(e.target.value)} /><br /><br />
        <label htmlFor="numBedrooms">Number of Bedrooms:</label>
        <input type="number" id="numBedrooms" name="numBedrooms" min="0" value={numBedrooms} onChange={(e) => setNumBedrooms(e.target.value)} /><br /><br />
        <label htmlFor="numBathrooms">Number of Bathrooms:</label>
        <input type="number" id="numBathrooms" name="numBathrooms" min="0" alue={numBathrooms} onChange={(e) => setNumBathrooms(e.target.value)} /><br /><br />
        <label htmlFor="floorArea">Floor Area:</label>
        <input type="number" id="floorArea" name="floorArea" value={floorArea} onChange={(e) => setFloorArea(e.target.value)} /><br /><br />
        <label htmlFor="description">Description:</label>
        <textarea id="description" name="description" value={description} onChange={(e) => setDescription(e.target.value)}></textarea><br /><br />
        <label htmlFor="amenities">Amenities:</label>
        <textarea id="amenities" name="amenities" value={amenities} onChange={(e) => setAmenities(e.target.value)}></textarea><br /><br />
        
        {/* Dropzone Section */}
        <div class="dropzone-container" 
          id="dropzone" 
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}>
          <input type="file" class="absolute inset-0 w-full h-full opacity-0 z-50" onChange={handleFileChange} multiple/>
          <div class="text-center">
            <img class="mx-auto h-12 w-12" src="https://www.svgrepo.com/show/357902/image-upload.svg" alt="Upload" />
            <h3 class="mt-2 text-sm font-medium text-gray-900">
              <label htmlFor="file-upload" class="relative cursor-pointer">
                <span class="span-container">Drag and drop</span>
                <span class="span-container text-indigo-600"> or browse</span>
                <span class="span-container"> to upload</span>
              </label>
            </h3>
            <p class="mt-1 text-xs text-gray-500">PNG, JPG up to 10MB</p>
          </div>

          {/* Image preview */}
          {images.length > 0 && (
            <div className="mt-4 flex flex-wrap justify-center items-center">
              {images.map((image, index) => (
                <img key={index} src={image} alt={`Preview ${index}`} className="max-h-40 m-1" />
              ))}
            </div>
          )}

        </div>
        <input type="submit" value="Submit"/>
      </fieldset>
    </form>
  );
};

export default RealEstateListingForm;