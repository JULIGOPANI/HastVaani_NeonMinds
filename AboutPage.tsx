import React from 'react';
import { motion } from 'framer-motion';
import { Users, Heart, Award, BookOpen } from 'lucide-react';

const AboutPage: React.FC = () => {
  const teamMembers = [
    {
      name: "Juli Gopani",
      // role: "Founder & AI Developer",
      image: "/src/assets/Juli img.jpg", // Updated image path
      // bio: "Tech enthusiast with expertise in AI and sign language recognition. Passionate about creating accessible technology for all."
    },
    {
      name: "Hinal Dobaria",
      // role: "Co-founder & Product Lead",
      image: "/src/assets/hinal img.jpg",
      // bio: "UX specialist focused on creating intuitive and accessible interfaces. Dedicated to bridging education gaps through technology."
    },
    {
      name: "Shreya Vekariya",
      // role: "Co-founder & Product Lead",
      image: "/src/assets/shreya img.jpg",
    //   bio: "UX specialist focused on creating intuitive and accessible interfaces. Dedicated to bridging education gaps through technology."
    // 
    },
    {
      name: "Pratvi Hirpara",
      // role: "Co-founder & Product Lead",
      image: "/src/assets/pratvi img.jpg",
      // bio: "UX specialist focused on creating intuitive and accessible interfaces. Dedicated to bridging education gaps through technology."
    }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.5 }
    }
  };

  return (
    <div className="pt-24 pb-16">
      <div className="container mx-auto px-4">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-4xl md:text-5xl font-display font-bold mb-6">
            About <span className="text-accent neon-text">हस्तवाणी</span>
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
          Our Mission is to make education accessible and inclusive for the deaf and mute community by leveraging technology to teach Indian Sign Language.</p>
        </motion.div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center mb-16">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <h2 className="text-2xl md:text-3xl font-display font-bold mb-6">
              Our <span className="text-accent neon-text">Story</span>
            </h2>
            
            <div className="space-y-6 text-gray-300">
              <p>
              हस्तवाणी (Hastavaani) was created with the vision of providing an educational platform dedicated entirely to Indian Sign Language. Unlike traditional resources, our platform ensures that learning ISL is interactive, engaging, and structured for students, educators, and anyone eager to learn sign language.
              </p>
              <p>
              Founded by Juli Gopani, Hinal Dobaria, Shreya Vekariya & Pratvi Hirpara, our platform integrates AI-based gesture recognition, animated sign demonstrations, and interactive learning modules to make sign language education more accessible than ever before.
              </p>
              <p>
                Our name, हस्तवाणी (Hastavaani), combines "हस्त" (hand) and "वाणी" (voice), symbolizing our commitment to transforming hand gestures into an educational voice for all.
              </p>
            </div>
          </motion.div>
          
          <motion.div
            className="order-first lg:order-last mb-8 lg:mb-0"
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <div className="relative rounded-xl overflow-hidden border border-accent/20 shadow-lg shadow-accent/10 w-2/3 mx-auto"> {/* Reduced width */}
              <img 
                src="/src/assets/about2.jpg" 
                alt="हस्तवाणी Team" 
                className="w-full h-auto object-contain" // Ensures the entire image is visible
              />
            </div>
          </motion.div>
        </div>

        <motion.div 
          className="mb-16"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-2xl md:text-3xl font-display font-bold mb-8 text-center">
            Our <span className="text-accent neon-text">Values</span>
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-dark-lighter p-6 rounded-xl border border-dark-lightest text-center">
              <div className="w-16 h-16 rounded-full bg-accent/20 flex items-center justify-center text-accent mx-auto mb-4">
                <Users className="w-8 h-8" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Inclusivity</h3>
              <p className="text-gray-400">
              Education should be for everyone, irrespective of physical abilities.
              </p>
            </div>

            <div className="bg-dark-lighter p-6 rounded-xl border border-dark-lightest text-center">
              <div className="w-16 h-16 rounded-full bg-accent/20 flex items-center justify-center text-accent mx-auto mb-4">
                <Heart className="w-8 h-8" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Empathy</h3>
              <p className="text-gray-400">
              Understanding the challenges of the deaf and mute community to build meaningful solutions.
              </p>
            </div>

            <div className="bg-dark-lighter p-6 rounded-xl border border-dark-lightest text-center">
              <div className="w-16 h-16 rounded-full bg-accent/20 flex items-center justify-center text-accent mx-auto mb-4">
                <Award className="w-8 h-8" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Innoation</h3>
              <p className="text-gray-400">
              Utilizing AI and advanced technology to enhance sign language learning.
              </p>
            </div>

            <div className="bg-dark-lighter p-6 rounded-xl border border-dark-lightest text-center">
              <div className="w-16 h-16 rounded-full bg-accent/20 flex items-center justify-center text-accent mx-auto mb-4">
                <BookOpen className="w-8 h-8" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Education for All</h3>
              <p className="text-gray-400">
              Spreading awareness about ISL and fostering an inclusive learning environment.
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-2xl md:text-3xl font-display font-bold mb-8 text-center">
            Meet Our <span className="text-accent neon-text">Team</span>
          </h2>

          <motion.div
            className="grid grid-cols-1 md:grid-cols-4 gap-6 max-w-6xl mx-auto"
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            {teamMembers.map((member, index) => (
              <motion.div
                key={index}
                className="bg-dark-lighter p-4 rounded-xl border border-dark-lightest overflow-hidden"
                variants={itemVariants}
              >
                <div className="flex flex-col items-center">
                  <div className="w-full h-60 rounded-xl overflow-hidden mb-4"> {/* Adjusted height */}
                    <img
                      src={member.image}
                      alt={member.name}
                      className="w-full h-full object-cover object-center"
                    />
                  </div>
                  <div className="text-center">
                    <h3 className="text-xl font-semibold">{member.name}</h3>
                    {/* <p className="text-accent mb-2">{member.role}</p>
                    <p className="text-gray-300">{member.bio}</p> */}
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default AboutPage;