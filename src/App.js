//src\App.js

import React from "react";
import { Routes, Route } from "react-router-dom";
import Header from "./components/Header";
import Footer from "./components/Footer";
import Home from "./pages/Home";
import Sell from "./pages/Sell";
import ImageSearch from "./pages/ImageSearch"; // Import ImageSearch component

function App() {
  return (
    <div className="max-w-[1440px] mx-auto bg-white ">
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/sell" element={<Sell />} />
        <Route path="/image-search" element={<ImageSearch />} /> {/* New route for image search */}
      </Routes>
      <Footer />
    </div>
  );
}

export default App;
