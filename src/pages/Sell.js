import React from "react";

import Banner from "../components/Banner"; 
import PostListingForm from "../components/PostListingForm";

const Home = () => {
    return(
        <div className="min-h-[1800px] ">
            <Banner/> 
            <PostListing />
        </div>
    )
}

export default Home;