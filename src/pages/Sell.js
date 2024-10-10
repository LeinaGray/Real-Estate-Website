import React from "react";

import Banner from "../components/Banner"; 
import BulkPostProperty from "../components/BulkPostProperty";
import PostListingForm from "../components/PostListingForm";

const Sell = () => {
    return(
        <div className="min-h-[1800px] ">
            {/* <PostListingForm/> */}
            <BulkPostProperty />
        </div>
    )
}

export default Sell;