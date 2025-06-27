import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  AcademicCapIcon,
  Bars3Icon,
  XMarkIcon,
  UserIcon,
  ChartBarIcon,
  Cog6ToothIcon
} from '@heroicons/react/24/outline';

const Navbar = ({ user, onLogout }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-brand">
          <AcademicCapIcon className="w-8 h-8" />
          <span>AI Learning</span>
        </Link>

        {/* Desktop Navigation */}
        <div className="navbar-nav desktop-nav">
          {user ? (
            <>
              <Link 
                to="/dashboard" 
                className={`nav-link ${isActive('/dashboard') ? 'active' : ''}`}
              >
                <ChartBarIcon className="w-4 h-4 mr-1" />
                Dashboard
              </Link>
              <Link 
                to="/learning-path" 
                className={`nav-link ${isActive('/learning-path') ? 'active' : ''}`}
              >
                Learning Path
              </Link>
              <div className="user-menu">
                <button className="user-menu-trigger">
                  <UserIcon className="w-5 h-5" />
                  <span>{user.name || user.username}</span>
                </button>
                <div className="user-menu-dropdown">
                  <Link to="/profile" className="dropdown-item">
                    <UserIcon className="w-4 h-4" />
                    Profile
                  </Link>
                  <Link to="/settings" className="dropdown-item">
                    <Cog6ToothIcon className="w-4 h-4" />
                    Settings
                  </Link>
                  <button onClick={onLogout} className="dropdown-item logout">
                    Logout
                  </button>
                </div>
              </div>
            </>
          ) : (
            <>
              <Link 
                to="/login" 
                className={`nav-link ${isActive('/login') ? 'active' : ''}`}
              >
                Sign In
              </Link>
              <Link 
                to="/register" 
                className="nav-link btn-primary"
              >
                Get Started
              </Link>
            </>
          )}
        </div>

        {/* Mobile Menu Button */}
        <button className="mobile-menu-btn" onClick={toggleMenu}>
          {isMenuOpen ? (
            <XMarkIcon className="w-6 h-6" />
          ) : (
            <Bars3Icon className="w-6 h-6" />
          )}
        </button>
      </div>

      {/* Mobile Navigation */}
      {isMenuOpen && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="mobile-nav"
        >
          {user ? (
            <>
              <Link to="/dashboard" className="mobile-nav-link" onClick={toggleMenu}>
                <ChartBarIcon className="w-5 h-5" />
                Dashboard
              </Link>
              <Link to="/learning-path" className="mobile-nav-link" onClick={toggleMenu}>
                Learning Path
              </Link>
              <Link to="/profile" className="mobile-nav-link" onClick={toggleMenu}>
                <UserIcon className="w-5 h-5" />
                Profile
              </Link>
              <button 
                onClick={() => { onLogout(); toggleMenu(); }} 
                className="mobile-nav-link logout"
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="mobile-nav-link" onClick={toggleMenu}>
                Sign In
              </Link>
              <Link to="/register" className="mobile-nav-link primary" onClick={toggleMenu}>
                Get Started
              </Link>
            </>
          )}
        </motion.div>
      )}
    </nav>
  );
};

export default Navbar;
