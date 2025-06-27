import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon, 
  AcademicCapIcon, 
  TrophyIcon,
  ClockIcon,
  ArrowTrendingUpIcon
} from '@heroicons/react/24/outline';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

// Components
import SkillRadarChart from './SkillRadarChart';
import LearningProgressChart from './LearningProgressChart';
import RecommendationCard from './RecommendationCard';
import SkillProgressChart from './SkillProgressChart';
import CategoryDistributionChart from './CategoryDistributionChart';

// Services
import { APIService } from '../../services/APIService';

const Dashboard = ({ user, mlModel }) => {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [skills, setSkills] = useState([]);
  const [learningPath, setLearningPath] = useState(null);
  const [progressData, setProgressData] = useState([]);

  useEffect(() => {
    loadDashboardData();
  }, [user]);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      // Get user profile data
      const profileData = await APIService.getUserProfile(user.id);
      
      if (profileData && mlModel) {
        // Generate learning path using ML model
        const generatedPath = await mlModel.generateLearningPath(
          profileData.skills || [],
          profileData.targetSkills || []
        );
        
        setSkills(profileData.skills || []);
        setLearningPath(generatedPath);
        setDashboardData(profileData);
        setProgressData(profileData.progressData || []);
      }
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSkillUpdate = async (skillName, newLevel) => {
    try {
      await APIService.updateSkillProgress(user.id, skillName, newLevel);
      await loadDashboardData(); // Refresh data
    } catch (error) {
      console.error('Error updating skill:', error);
    }
  };

  if (loading) {
    return (
      <div className="dashboard-loading">
        <div className="loading-spinner"></div>
        <p>Loading your personalized dashboard...</p>
      </div>
    );
  }

  const overallProgress = skills.length > 0 
    ? skills.reduce((sum, skill) => sum + skill.level, 0) / skills.length * 100
    : 0;

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="welcome-section"
        >
          <h1>Welcome back, {user.name || user.username}!</h1>
          <p>Here's your personalized learning overview</p>
        </motion.div>
      </div>

      <div className="dashboard-grid">
        {/* Overview Cards */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="stats-grid"
        >
          <div className="stat-card">
            <div className="stat-icon">
              <AcademicCapIcon className="w-8 h-8" />
            </div>
            <div className="stat-content">
              <h3>Total Skills</h3>
              <p className="stat-number">{skills.length}</p>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon">
              <TrophyIcon className="w-8 h-8" />
            </div>
            <div className="stat-content">
              <h3>Skill Level</h3>
              <p className="stat-number">{overallProgress.toFixed(0)}%</p>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon">
              <ClockIcon className="w-8 h-8" />
            </div>
            <div className="stat-content">
              <h3>Learning Hours</h3>
              <p className="stat-number">
                {learningPath ? learningPath.estimatedDuration : 0}h
              </p>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon">
              <ArrowTrendingUpIcon className="w-8 h-8" />
            </div>
            <div className="stat-content">
              <h3>Progress</h3>
              <div className="progress-circle">
                <CircularProgressbar
                  value={overallProgress}
                  text={`${overallProgress.toFixed(0)}%`}
                  styles={buildStyles({
                    textSize: '16px',
                    pathColor: '#3b82f6',
                    textColor: '#1f2937',
                    trailColor: '#e5e7eb'
                  })}
                />
              </div>
            </div>
          </div>
        </motion.div>

        {/* Skills Visualization */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="chart-section"
        >
          <div className="chart-card">
            <h3>Skill Distribution</h3>
            <SkillRadarChart skills={skills} />
          </div>
        </motion.div>

        {/* Learning Progress */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="progress-section"
        >
          <div className="progress-card">
            <h3>Learning Progress</h3>
            <LearningProgressChart learningPath={learningPath} />
          </div>
        </motion.div>

        {/* Skill Progress Over Time */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="chart-section"
        >
          <div className="chart-card">
            <h3>Skill Progress Over Time</h3>
            <SkillProgressChart progressData={progressData} />
          </div>
        </motion.div>

        {/* Skills by Category */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="chart-section"
        >
          <div className="chart-card">
            <h3>Skills by Category</h3>
            <CategoryDistributionChart skills={skills} />
          </div>
        </motion.div>

        {/* Current Skills */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="skills-section"
        >
          <div className="skills-card">
            <h3>Your Skills</h3>
            <div className="skills-list">
              {skills.map((skill, index) => (
                <div key={index} className="skill-item">
                  <div className="skill-info">
                    <span className="skill-name">{skill.name}</span>
                    <span className="skill-category">{skill.category}</span>
                  </div>
                  <div className="skill-level">
                    <div className="level-bar">
                      <div 
                        className="level-fill"
                        style={{ width: `${skill.level * 100}%` }}
                      ></div>
                    </div>
                    <span className="level-text">
                      {(skill.level * 100).toFixed(0)}%
                    </span>
                  </div>
                  <button
                    onClick={() => handleSkillUpdate(skill.name, skill.level + 0.1)}
                    className="update-btn"
                  >
                    Update
                  </button>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Recommendations */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="recommendations-section"
        >
          <div className="recommendations-card">
            <h3>Recommended for You</h3>
            <div className="recommendations-list">
              {learningPath?.recommendedResources?.slice(0, 3).map((resource, index) => (
                <RecommendationCard
                  key={index}
                  resource={resource}
                  onStart={() => console.log('Starting resource:', resource.title)}
                />
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;
// In Dashboard component
useEffect(() => {
  const fetchChartData = async () => {
    try {
      const response = await fetch(`/api/chart-data/${user.id}`);
      const chartData = await response.json();
      setChartData(chartData);
    } catch (error) {
      console.error('Error fetching chart data:', error);
    }
  };

  if (user) {
    fetchChartData();
  }
}, [user]);
