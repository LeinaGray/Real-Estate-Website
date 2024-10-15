import React, { useState } from "react";
import { useNavigate } from 'react-router-dom'; // Import useNavigate for routing
import { FaCamera } from "react-icons/fa"; // Import camera icon
import Search from "./Search"; // Assuming Search is another component you want to include

const Banner = () => {
    const [searchTerm, setSearchTerm] = useState("");
    const navigate = useNavigate(); // Hook for navigation

    // Handle the input change
    const handleInputChange = (e) => {
        setSearchTerm(e.target.value);
    };

    // Redirect to image search page when camera button is clicked
    const handleCameraClick = () => {
        navigate('/image-search'); // Navigate to image search page
    };

    return (
        <section className="h-full max-h-[640px] mb-8 xl:mb-24">
            <div className="flex items-center justify-center mb-4">
                <h1 className="text-3xl font-bold text-gray-800 mr-56">
                    Search for Properties
                </h1>
                <div className="flex items-center border-2 rounded-lg overflow-hidden w-full max-w-[400px] ml-56">
                    <input
                        type="text"
                        value={searchTerm}
                        onChange={handleInputChange}
                        placeholder="Type to search..."
                        className="flex-grow px-4 py-2 outline-none border-none" // No border for input
                    />
                    <button 
                        onClick={handleCameraClick} // Handle the camera button click
                        className="bg-violet-500 p-2 rounded-lg mr-1" // Added margin-right to camera button
                    >
                        <FaCamera className="text-white" />
                    </button>
                </div>
            </div>

            <Search /> {/* Include the Search component */}
        </section>
    );
};

export default Banner;
