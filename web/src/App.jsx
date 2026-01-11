import React, { useState } from 'react';
import Home from './pages/Home';
import Analysis from './pages/Analysis';
import Navbar from './components/Navbar';

function App() {
  const [currentPage, setCurrentPage] = useState('home');

  return (
    <div className="min-h-screen bg-[#020617] text-white">
      <Navbar onNavigate={setCurrentPage} currentPage={currentPage} />
      <main>
        {currentPage === 'home' ? <Home onAnalyze={() => setCurrentPage('analysis')} /> : <Analysis />}
      </main>
    </div>
  );
}

export default App;
