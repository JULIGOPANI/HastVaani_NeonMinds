import React from 'react';
import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';

const HeroSection: React.FC = () => {
  return (
    <section className="hero-section py-10 flex items-center justify-center min-h-screen"> {/* Centered vertically */}
      {/* Background gradient */}
      
      
      {/* Floating elements */}
      <motion.div 
        className="absolute top-1/4 left-1/4 w-64 h-64 rounded-full bg-accent/5 blur-3xl z-0"
        animate={{ 
          x: [0, 30, 0],
          y: [0, -30, 0],
        }}
        transition={{ 
          duration: 8,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      
      <motion.div 
        className="absolute bottom-1/3 right-1/4 w-80 h-80 rounded-full bg-accent-purple/5 blur-3xl z-0"
        animate={{ 
          x: [0, -40, 0],
          y: [0, 40, 0],
        }}
        transition={{ 
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />

      <div className="container mx-auto px-4 relative z-10">
        <div className="flex flex-col items-center text-center">
          <motion.h1 
            className="text-4xl md:text-6xl lg:text-7xl font-display font-bold mb-6 max-w-4xl mt-20" // Added margin-top
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <span className="text-white">The language of hands,</span><br />
            <span className="text-accent neon-text">the voice of hearts</span>
          </motion.h1>
          
          <motion.h2 
            className="text-4xl md:text-6xl lg:text-7xl font-display font-bold mb-6 max-w-4xl text-accent" // Added new text
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            हस्तवाणी
          </motion.h2>
          
          <motion.p 
            className="text-xl text-gray-300 max-w-2xl mb-10"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            हस्तवाणी bridges communication gaps through advanced AI-powered Indian Sign Language recognition, making the world more accessible for everyone.
          </motion.p>
          
          <motion.div 
            className="flex flex-col sm:flex-row gap-4 mb-16"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <Link to="/technology" className="btn-primary flex items-center gap-2">
              <span>Try हस्तवाणी </span>
              <ArrowRight className="w-5 h-5" />
            </Link>
            <Link to="/about" className="btn-secondary">
              Learn More
            </Link>
          </motion.div>
          
          <motion.div 
            className="w-full max-w-4xl"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.6 }}
          >

          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;