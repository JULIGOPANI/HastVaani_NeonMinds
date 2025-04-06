import React from 'react';
import { motion } from 'framer-motion';
import { Code, Cpu, Camera, Cloud, Link } from 'lucide-react';

const TechnologySection: React.FC = () => {
  return (
    <motion.section
      className="py-20 relative"
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.8 }}
    >
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-display font-bold mb-4">
            Our <span className="text-accent neon-text">Technology</span> Stack
          </h2>
          <p className="text-gray-300 max-w-2xl mx-auto">
            Powered by cutting-edge AI and machine learning technologies to provide accurate and responsive sign language recognition.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          <motion.div 
            className="p-6 rounded-xl border border-dark-lightest hover:border-accent/30 hero-gradient"
            whileHover={{ y: -5 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <div className="bg-accent/10 p-3 rounded-lg w-14 h-14 flex items-center justify-center mb-4">
              <Camera size={28} className="text-accent" />
            </div>
            <h3 className="text-xl font-bold mb-3">Computer Vision</h3>
            <p className="text-gray-300">
              Advanced image processing algorithms to accurately detect and track hand movements and gestures in real-time.
            </p>
          </motion.div>
          
          <motion.div 
            className="p-6 rounded-xl border border-dark-lightest hover:border-accent/30 hero-gradient"
            whileHover={{ y: -5 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <div className="bg-accent/10 p-3 rounded-lg w-14 h-14 flex items-center justify-center mb-4">
              <Cpu size={28} className="text-accent" />
            </div>
            <h3 className="text-xl font-bold mb-3">Neural Networks</h3>
            <p className="text-gray-300">
              Custom-trained convolutional neural networks that understand and interpret the nuances of Indian Sign Language.
            </p>
          </motion.div>
          
          <motion.div 
            className="p-6 rounded-xl border border-dark-lightest hover:border-accent/30 hero-gradient"
            whileHover={{ y: -5 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <div className="bg-accent/10 p-3 rounded-lg w-14 h-14 flex items-center justify-center mb-4">
              <Code size={28} className="text-accent" />
            </div>
            <h3 className="text-xl font-bold mb-3">React Frontend</h3>
            <p className="text-gray-300">
              Responsive and intuitive user interface built with React and Tailwind CSS for a seamless user experience.
            </p>
          </motion.div>
          
          <motion.div 
            className="p-6 rounded-xl border border-dark-lightest hover:border-accent/30 hero-gradient"
            whileHover={{ y: -5 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <div className="bg-accent/10 p-3 rounded-lg w-14 h-14 flex items-center justify-center mb-4">
              <Cloud size={28} className="text-accent" />
            </div>
            <h3 className="text-xl font-bold mb-3">FastAPI Backend</h3>
            <p className="text-gray-300">
              High-performance Python backend that processes sign language recognition requests with minimal latency.
            </p>
          </motion.div>
        </div>
        
        {/*  */}
      </div>
    </motion.section>
  );
};

export default TechnologySection;