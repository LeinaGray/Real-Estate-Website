//src\components\HouseList.js

import React, { useContext, useState } from 'react';
import { HouseContext } from "./HouseContext";
import House from "./House";
import { Link } from "react-router-dom";
import { ImSpinner2 } from "react-icons/im";

const HouseList = () => {
    const { houses, loading } = useContext(HouseContext);
    
    // Pagination state
    const [currentPage, setCurrentPage] = useState(1);
    const housesPerPage = 9; // Number of houses to show per page

    // Calculate total pages
    const totalPages = Math.ceil(houses.length / housesPerPage);

    // Get current houses to display
    const indexOfLastHouse = currentPage * housesPerPage;
    const indexOfFirstHouse = indexOfLastHouse - housesPerPage;
    const currentHouses = houses.slice(indexOfFirstHouse, indexOfLastHouse);

    if (loading) {
        return (
            <ImSpinner2 className='mx-auto animate-spin text-4xl text-violet-700 font-bold mt-[200px]' />
        );
    }

    if (currentHouses.length < 1) {
        return <div>Sorry, no match found!</div>;
    }

    return (
        <section className='mb-20'>
            <div className="container mx-auto max-w-[1100px]">
                <div className='grid md:grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-20'>
                    {currentHouses.map((house, index) => {
                        return (
                            <Link to={`/property/${house.post_url}`} key={index}>
                                <House house={house} />
                            </Link>
                        )
                    })}
                </div>

                {/* Pagination Controls */}
                <div className="pagination flex justify-center mt-4">
                    <button
                        onClick={() => setCurrentPage(currentPage > 1 ? currentPage - 1 : 1)}
                        disabled={currentPage === 1}
                        aria-label="Previous page"
                        className="mx-2 bg-violet-600 text-white px-4 py-2 rounded-lg"
                    >
                        Previous
                    </button>
                    <span>{` Page ${currentPage} of ${totalPages} `}</span>
                    <button
                        onClick={() => setCurrentPage(currentPage < totalPages ? currentPage + 1 : totalPages)}
                        disabled={currentPage === totalPages}
                        aria-label="Next page"
                        className="mx-2 bg-violet-600 text-white px-4 py-2 rounded-lg"
                    >
                        Next
                    </button>
                </div>
            </div>
        </section>
    );
}

export default HouseList;
