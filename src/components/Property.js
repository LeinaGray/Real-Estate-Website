//src\components\Property.js

import React from 'react';
import { RiHeart3Line } from "react-icons/ri";

const House = ({ house }) => {
    const { image_paths, title, location, price } = house;

    return (
        <div className='bg-white shadow-1 pb-5 rounded-lg w-full max-w-[300px] mx-auto cursor-pointer hover:shadow-2xl transition text-gray-600'>
            <img className='w-full h-[150px] rounded-t-lg' src={image_paths} alt={title} />
            <div className='p-3'>
                <div className='flex justify-between items-center px-3'>
                    <div className='text-lg text-violet-600 mb-4 font-bold pl-2'>
                        ${price} <span className='text-gray-500 font-light text-sm'>/month</span>
                    </div>
                    <RiHeart3Line className='text-3xl hover:text-red-500' />
                </div>
                <div className='flex gap-x-2'>
                    <div className='text-gray-900 font-bold'>{location}</div>
                </div>
            </div>
        </div>
    );
}

export default House;