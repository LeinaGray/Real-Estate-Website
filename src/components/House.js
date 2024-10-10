import React from 'react';
import "../css/Property.css"

import {BiBed, BiBath, BiArea} from 'react-icons/bi';
import {RiHeart3Line } from "react-icons/ri";

const House = ({house}) => {
    const {image, name, type, country, address, bedrooms, bathrooms, surface, price} = house;

    return (
        <div class=" w-[352px] h-[424px] relative">
            <div class="Rectangle392 w-[352px] h-[424px] left-0 top-0 absolute bg-white rounded-lg border border-[#f0effb]"></div>
            <div class="Favorited">
                <div class="Ellipse"></div>
                <div class="Frame w-6 h-6 px-[3px] pt-[5px] pb-[3.64px] left-[12px] top-[12px] absolute justify-center items-center inline-flex">
                <RiHeart3Line className='text-3xl hover:text-red-500'/>
                </div>
            </div>
            <h3 class="Title">{name}</h3>
            <h3 class="Address">{address}</h3>
            <h3 class="Price">${price}</h3>
            <div class="Attributes">
                <div class="Attribute-item">
                <div class="Icon"><BiBed/> </div>
                <p class="Label">{bedrooms}</p>
                </div>
                <div class="Attribute-item">
                <div class="Icon"><BiBath/> </div>
                <p class="Label">{bathrooms}</p>
                </div>
                <div class="Attribute-item">
                <div class="Icon"><BiArea/> </div>
                <p class="Label">{surface}</p>
                </div>
            </div>
        
            <img class="Image" src={image} />

        </div>
    )
}

export default House;