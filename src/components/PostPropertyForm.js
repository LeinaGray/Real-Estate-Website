import React, { useState } from 'react';

const PostPropertyForm = () => {
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

  return (
    <form onSubmit={handleSubmit}>
        <fieldset>
            <legend><h2>Real Estate Listing Form</h2></legend>
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
            <input type="number" id="price" name="price" value={price} onChange={(e) => setPrice(e.target.value)} /><br /><br />
            <label htmlFor="address">Address:</label>
            <input type="text" id="address" name="address" value={address} onChange={(e) => setAddress(e.target.value)} /><br /><br />
            <label htmlFor="numBedrooms">Number of Bedrooms:</label>
            <input type="number" id="numBedrooms" name="numBedrooms" value={numBedrooms} onChange={(e) => setNumBedrooms(e.target.value)} /><br /><br />
            <label htmlFor="numBathrooms">Number of Bathrooms:</label>
            <input type="number" id="numBathrooms" name="numBathrooms" value={numBathrooms} onChange={(e) => setNumBathrooms(e.target.value)} /><br /><br />
            <label htmlFor="floorArea">Floor Area:</label>
            <input type="number" id="floorArea" name="floorArea" value={floorArea} onChange={(e) => setFloorArea(e.target.value)} /><br /><br />
            <label htmlFor="description">Description:</label>
            <textarea id="description" name="description" value={description} onChange={(e) => setDescription(e.target.value)}></textarea><br /><br />
            <label htmlFor="amenities">Amenities:</label>
            <textarea id="amenities" name="amenities" value={amenities} onChange={(e) => setAmenities(e.target.value)}></textarea><br /><br />
            <label htmlFor="images">Images:</label>
            <input type="file" id="images" name="images" multiple onChange={(e) => setImages(e.target.files)} /><br /><br />
            <input type="submit" value="Submit"/>
        </fieldset>
    </form>
  );
};

export default PostPropertyForm;