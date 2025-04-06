import React from 'react';
import { motion } from 'framer-motion';
import { BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer, XAxis, YAxis, Tooltip, Legend } from 'recharts';

const StatisticsSection: React.FC = () => {
  // Data for pie chart - Global hearing impairment distribution
  const pieData = [
    { name: 'South Asia', value: 90.5 },
    { name: 'Sub-Saharan Africa', value: 48.3 },
    { name: 'East Asia', value: 86.7 },
    { name: 'Europe & Central Asia', value: 57.2 },
    { name: 'Other Regions', value: 83.3 }
  ];

  // Data for bar chart - Hearing impairment in India by age group (millions)
  const barData = [
    { age: '0-14', value: 3.2 },
    { age: '15-29', value: 5.7 },
    { age: '30-44', value: 8.3 },
    { age: '45-59', value: 12.5 },
    { age: '60+', value: 18.9 }
  ];

  // Colors for pie chart
  const COLORS = ['#00f2ff', '#00d0ff', '#00a8ff', '#0080ff', '#bf00ff'];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.6 }
    }
  };

  return (
    <section className="py-20 relative overflow-hidden">
      <div className="container mx-auto px-4">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-3xl md:text-4xl font-display font-bold mb-4">
            <span className="text-accent neon-text">Statistics</span> & Impact
          </h2>
          <p className="text-gray-300 max-w-2xl mx-auto">
            Understanding the scale of hearing impairment globally and in India helps us recognize the importance of accessible communication tools like हस्तवाणी.
          </p>
        </motion.div>

        <motion.div 
          className="grid grid-cols-1 md:grid-cols-2 gap-8"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          <motion.div 
            className="card h-full"
            variants={itemVariants}
          >
            <h3 className="text-xl font-semibold mb-6">Global Hearing Impairment Distribution (Millions)</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [`${value} million`, 'Population']} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </motion.div>

          <motion.div 
            className="card h-full"
            variants={itemVariants}
          >
            <h3 className="text-xl font-semibold mb-6">Hearing Impairment in India by Age (Millions)</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={barData}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <XAxis dataKey="age" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${value} million`, 'Population']} />
                  <Legend />
                  <Bar dataKey="value" name="Population" fill="#00f2ff" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        </motion.div>

        <motion.div 
          className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          <motion.div 
            className="card text-center"
            variants={itemVariants}
          >
            <div className="text-4xl font-bold text-accent mb-2 neon-text">18+</div>
            <h3 className="text-xl font-semibold mb-2">Million</h3>
            <p className="text-gray-400">People in India with significant hearing impairment</p>
          </motion.div>

          <motion.div 
            className="card text-center"
            variants={itemVariants}
          >
            <div className="text-4xl font-bold text-accent mb-2 neon-text">5%</div>
            <h3 className="text-xl font-semibold mb-2">Of Population</h3>
            <p className="text-gray-400">Percentage of global population with disabling hearing loss</p>
          </motion.div>

          <motion.div 
            className="card text-center"
            variants={itemVariants}
          >
            <div className="text-4xl font-bold text-accent mb-2 neon-text">90%</div>
            <h3 className="text-xl font-semibold mb-2">Communication Gap</h3>
            <p className="text-gray-400">Of hearing people cannot communicate in sign language</p>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

export default StatisticsSection;