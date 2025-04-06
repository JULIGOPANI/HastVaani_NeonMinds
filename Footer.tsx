import React from 'react';
import { Link } from 'react-router-dom';
import { Github, Twitter, Linkedin, Mail } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-dark-darker py-12 border-t border-dark-lightest">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          <div>
            <h3 className="text-xl font-bold mb-4">हस्तवाणी</h3>
            <p className="text-gray-400 mb-4">
              Breaking barriers through Indian Sign Language recognition technology.
            </p>
            <div className="flex space-x-4">
              <a href="#" className="text-gray-400 hover:text-accent transition">
                <Github size={20} />
              </a>
              <a href="#" className="text-gray-400 hover:text-accent transition">
                <Twitter size={20} />
              </a>
              <a href="#" className="text-gray-400 hover:text-accent transition">
                <Linkedin size={20} />
              </a>
              <a href="#" className="text-gray-400 hover:text-accent transition">
                <Mail size={20} />
              </a>
            </div>
          </div>
          
          <div>
            <h4 className="text-lg font-semibold mb-4">Navigation</h4>
            <ul className="space-y-2">
              <li>
                <Link to="/" className="text-gray-400 hover:text-accent transition">
                  Home
                </Link>
              </li>
              <li>
                <Link to="/detection" className="text-gray-400 hover:text-accent transition">
                  Detection
                </Link>
              </li>
              <li>
                <Link to="/about" className="text-gray-400 hover:text-accent transition">
                  About
                </Link>
              </li>
              <li>
                <Link to="/contact" className="text-gray-400 hover:text-accent transition">
                  Contact
                </Link>
              </li>
            </ul>
          </div>
          
          {/* <div>
            <h4 className="text-lg font-semibold mb-4">Resources</h4>
            <ul className="space-y-2">
              <li>
                <a href="#" className="text-gray-400 hover:text-accent transition">
                  Documentation
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-400 hover:text-accent transition">
                  API Reference
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-400 hover:text-accent transition">
                  Blog
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-400 hover:text-accent transition">
                  Research Papers
                </a>
              </li>
            </ul>
          </div>
           */}
          <div>
            <h4 className="text-lg font-semibold mb-4">Contact Us</h4>
            <address className="not-italic text-gray-400">
              <p className="mb-2">Email: juligopani2004@gmail.com</p>
              <p className="mb-2">Phone: +91 9726197170</p>
              {/* <p>Address: Tech Hub, Bangalore, India</p> */}
            </address>
          </div>
        </div>
        
        <div className="border-t border-dark-lightest pt-8 mt-8 text-center text-gray-500 text-sm">
          <p>&copy; {new Date().getFullYear()} हस्तवाणी. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;