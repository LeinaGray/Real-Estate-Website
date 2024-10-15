import React from "react";

import Banner from "../components/Banner"; 
import BulkPostProperty from "../components/BulkPostProperty";
import PostListingForm from "../components/PostListingForm";
import BulkReverseImageSearch from "../components/BulkReverseImageSearch";

const Sell = () => {
    return(
        <div className="min-h-[1800px] ">
            <PostListingForm/>
            {/* <BulkPostProperty /> */}
            <BulkReverseImageSearch />
        </div>
    )
}

export default Sell;