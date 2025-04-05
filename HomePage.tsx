import React from 'react';
import HeroSection from '../components/HeroSection';
import FeaturesSection from '../components/FeaturesSection';
import StatisticsSection from '../components/StatisticsSection';
import TechnologySection from '../components/TechnologySection';
// import Footer from '../components/Footer'; // Removed as it is unused
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';

const HomePage: React.FC = () => {
  return (
    <div>
      <HeroSection />
      <TechnologySection />
      <FeaturesSection />
      <StatisticsSection />
      
      <motion.section 
        className="py-20 relative"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8 }}
      >
        
      </motion.section>
      
    </div>
  );
};

export default HomePage;