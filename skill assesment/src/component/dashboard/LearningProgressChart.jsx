import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';

const LearningProgressChart = ({ learningPath }) => {
  if (!learningPath || !learningPath.recommendedResources) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">No learning path data available</p>
      </div>
    );
  }

  const data = learningPath.recommendedResources.slice(0, 6).map((resource, index) => ({
    name: resource.title.length > 20 ? resource.title.substring(0, 20) + '...' : resource.title,
    duration: resource.duration,
    difficulty: resource.difficulty,
    completed: Math.random() > 0.5 // Simulate completion status
  }));

  const getBarColor = (difficulty, completed) => {
    if (completed) return '#10b981';
    
    switch (difficulty) {
      case 'beginner': return '#3b82f6';
      case 'intermediate': return '#f59e0b';
      case 'advanced': return '#ef4444';
      default: return '#6b7280';
    }
  };

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="name" 
          angle={-45}
          textAnchor="end"
          height={80}
          fontSize={12}
        />
        <YAxis label={{ value: 'Duration (min)', angle: -90, position: 'insideLeft' }} />
        <Tooltip 
          formatter={(value, name, props) => [
            `${value} minutes`,
            `Duration (${props.payload.difficulty})`
          ]}
        />
        <Bar dataKey="duration" radius={[4, 4, 0, 0]}>
          {data.map((entry, index) => (
            <Cell 
              key={`cell-${index}`} 
              fill={getBarColor(entry.difficulty, entry.completed)} 
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

export default LearningProgressChart;
