import React from 'react';
import { motion } from 'framer-motion';
import {
  ClockIcon,
  StarIcon,
  PlayIcon,
  BookmarkIcon
} from '@heroicons/react/24/outline';

const RecommendationCard = ({ resource, onStart }) => {
  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800';
      case 'intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const formatDuration = (minutes) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
  };

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="recommendation-card"
    >
      <div className="card-header">
        <h4 className="resource-title">{resource.title}</h4>
        <div className="resource-meta">
          <span className={`difficulty-badge ${getDifficultyColor(resource.difficulty)}`}>
            {resource.difficulty}
          </span>
          <div className="rating">
            <StarIcon className="w-4 h-4 text-yellow-400 fill-current" />
            <span>{resource.rating}</span>
          </div>
        </div>
      </div>

      <p className="resource-description">{resource.description}</p>

      <div className="resource-details">
        <div className="detail-item">
          <ClockIcon className="w-4 h-4" />
          <span>{formatDuration(resource.duration)}</span>
        </div>
        <div className="skills-tags">
          {resource.skills.slice(0, 3).map((skill, index) => (
            <span key={index} className="skill-tag">
              {skill}
            </span>
          ))}
          {resource.skills.length > 3 && (
            <span className="skill-tag more">
              +{resource.skills.length - 3} more
            </span>
          )}
        </div>
      </div>

      <div className="card-actions">
        <button
          onClick={() => onStart(resource)}
          className="start-btn"
        >
          <PlayIcon className="w-4 h-4" />
          Start Learning
        </button>
        <button className="bookmark-btn">
          <BookmarkIcon className="w-4 h-4" />
        </button>
      </div>
    </motion.div>
  );
};

export default RecommendationCard;
