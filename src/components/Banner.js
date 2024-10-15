import React, { useState } from "react";
import { FaCamera } from "react-icons/fa"; // Import camera icon
import Search from "./Search";

const Banner = () => {
    const [searchTerm, setSearchTerm] = useState("");

    // Handle the input change
    const handleInputChange = (e) => {
        setSearchTerm(e.target.value);
    };

    // Handle the camera button click to upload an image
    const handleCameraClick = () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                console.log("Uploaded file:", file);
                // Add logic to handle the uploaded file
            }
        };
        input.click();
    };

    return (
        <section className="h-full max-h-[640px] mb-8 xl:mb-24">
            <div className="flex items-center justify-center mb-4">
                <h1 className="text-3xl font-bold text-gray-800 mr-56">
                    Search for Properties
                </h1>
                <div className="flex items-center border-2 rounded-lg overflow-hidden w-full max-w-[400px] ml-56"> {/* Container border */}
                    <input
                        type="text"
                        value={searchTerm}
                        onChange={handleInputChange}
                        placeholder="Type to search..."
                        className="flex-grow px-4 py-2 outline-none border-none" // No border for input
                    />
                    <button 
                        onClick={handleCameraClick} 
                        className="bg-violet-500 p-2 rounded-lg mr-1" // Added margin-right to camera button
                    >
                        <FaCamera className="text-white" />
                    </button>
                </div>
            </div>

            <Search />
        </section>
    );
};

export default Banner;
