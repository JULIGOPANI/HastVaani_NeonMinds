import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import TechnologyPage from './pages/Detection';
import AboutPage from './pages/AboutPage';
import ContactPage from './pages/ContactPage';
import BackgroundParticles from './components/BackgroundParticles';
import LearningModule from './pages/LearningModule'; // Import LearningModule
import Quiz from './pages/Quiz'; // Regular import for Quiz
import { Game } from './pages/Game';
import AnimateLearn from './pages/AnimateLearn'; // Import AnimateLearn
import Chatbot from './components/Chatbot/Chatbot'; // Import Chatbot

import LoginPage from './pages/LoginPage';
import Logout from './components/Logout'; // Import Logout

function App() {
  return (
    
      <div className="min-h-screen flex flex-col bg-primary-dark relative overflow-hidden">
        <BackgroundParticles />
        <Navbar />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/technology" element={<TechnologyPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="/contact" element={<ContactPage />} />
            <Route path="/learning-module" element={<LearningModule />} /> {/* Add LearningModule route */}
            <Route path="/games" element={<Game />} />
            <Route path="/quiz" element={<Quiz />} /> {/* Add Quiz route */}
            <Route path="/animatelearn" element={<AnimateLearn />} />
            <Route path="/login" element={<LoginPage />} /> {/* Add Login route */}
            <Route path="/signup" element={<LoginPage />} /> {/* Add Signup route */}
            <Route path="/logout" element={<Logout />} /> {/* Add Logout route */}
          </Routes>
        </main>
        <Chatbot /> {/* Add Chatbot component */}
        <Footer />
      </div>
    
  );
}

export default App;