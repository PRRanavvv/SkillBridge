import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import * as tf from '@tensorflow/tfjs';

// Components
import Navbar from './components/Common/Navbar';
import HomePage from './components/Home/HomePage';
import Dashboard from './components/Dashboard/Dashboard';
import ProfileSetup from './components/Profile/ProfileSetup';
import Assessment from './components/Assessment/Assessment';
import LearningPath from './components/LearningPath/LearningPath';
import Login from './components/Auth/Login';
import Register from './components/Auth/Register';

// Services
import { MLModelService } from './services/MLModelService';
import { AuthService } from './services/AuthService';

// Styles
import './styles/global.css';

// Protected Route Component
const ProtectedRoute = ({ children, user }) => {
  return user ? children : <Navigate to="/login" replace />;
};

// Public Route Component (redirects authenticated users)
const PublicRoute = ({ children, user, redirectTo = "/dashboard" }) => {
  return user ? <Navigate to={redirectTo} replace /> : children;
};

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [mlModel, setMlModel] = useState(null);

  useEffect(() => {
    initializeApp();
  }, []);

  const initializeApp = async () => {
    try {
      setLoading(true);
      
      // Initialize TensorFlow.js
      await tf.ready();
      console.log('TensorFlow.js initialized');

      // Initialize ML Model Service
      const modelService = new MLModelService();
      await modelService.initialize();
      setMlModel(modelService);

      // Check for existing user session
      const currentUser = AuthService.getCurrentUser();
      if (currentUser) {
        setUser(currentUser);
      }
    } catch (error) {
      console.error('Failed to initialize app:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = (userData) => {
    setUser(userData);
  };

  const handleLogout = () => {
    AuthService.logout();
    setUser(null);
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Initializing AI Learning System...</p>
      </div>
    );
  }

  return (
    <Router>
      <div className="App">
        <Navbar user={user} onLogout={handleLogout} />
        <main className="main-content">
          <Routes>
            {/* Public Routes */}
            <Route path="/" element={<HomePage />} />
            
            {/* Authentication Routes */}
            <Route 
              path="/login" 
              element={
                <PublicRoute user={user}>
                  <Login onLogin={handleLogin} />
                </PublicRoute>
              } 
            />
            <Route 
              path="/register" 
              element={
                <PublicRoute user={user}>
                  <Register onLogin={handleLogin} />
                </PublicRoute>
              } 
            />
            
            {/* Protected Routes */}
            <Route 
              path="/profile-setup" 
              element={
                <ProtectedRoute user={user}>
                  <ProfileSetup user={user} mlModel={mlModel} />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/dashboard" 
              element={
                <ProtectedRoute user={user}>
                  <Dashboard user={user} mlModel={mlModel} />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/assessment/:skill" 
              element={
                <ProtectedRoute user={user}>
                  <Assessment user={user} mlModel={mlModel} />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/learning-path" 
              element={
                <ProtectedRoute user={user}>
                  <LearningPath user={user} mlModel={mlModel} />
                </ProtectedRoute>
              } 
            />
            
            {/* Catch-all route for 404 */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <Toaster position="top-right" />
      </div>
    </Router>
  );
}

export default App;
