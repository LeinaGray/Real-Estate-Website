//src\components\HouseContext.js

import React, { useState, useEffect, createContext } from 'react';
import Papa from 'papaparse'; // Import PapaParse

export const HouseContext = createContext();

const HouseContextProvider = ({ children }) => {
    const [houses, setHouses] = useState([]);
    const [country, setCountry] = useState("Location (any)");
    const [price, setPrice] = useState('Price range (any)');
    const [loading, setLoading] = useState(true);
    const [countries, setCountries] = useState([]); // Initialize countries
    const [properties, setProperties] = useState([]); // Initialize properties

    useEffect(() => {
        const loadData = async () => {
            const response = await fetch('/metadata.csv'); 
            const reader = response.body.getReader();
            const result = await reader.read();
            const decoder = new TextDecoder("utf-8");
            const csvData = decoder.decode(result.value);

            Papa.parse(csvData, {
                header: true,
                complete: (results) => {
                    setHouses(results.data);
                    setLoading(false);
                    
                    // Extract unique countries and properties after loading the houses
                    const uniqueCountries = [...new Set(results.data.map(house => house.location))];
                    const uniqueProperties = [...new Set(results.data.map(house => house.title))];
                    
                    setCountries(uniqueCountries);
                    setProperties(uniqueProperties);
                },
                error: (error) => {
                    console.error("Error reading CSV:", error);
                    setLoading(false);
                },
            });
        };

        loadData();
    }, []);

    const handleClick = () => {
        setLoading(true);

        const minPrice = parseInt(price.split(' ')[0]);
        const maxPrice = parseInt(price.split(' ')[2]);

        const newHouses = houses.filter((house) => {
            const housePrice = parseInt(house.price.replace(/[^\d]/g, '')); // Convert price to integer
            return housePrice >= minPrice && housePrice <= maxPrice;
        });

        setTimeout(() => {
            setHouses(newHouses.length < 1 ? [] : newHouses);
            setLoading(false);
        }, 1000);
    };

    return (
        <HouseContext.Provider value={{
            houses,
            loading,
            handleClick,
            country,
            setCountry,
            price,
            setPrice,
            countries, // Pass countries to the context
            properties, // Pass properties to the context
        }}>
            {children}
        </HouseContext.Provider>
    );
};

export default HouseContextProvider;