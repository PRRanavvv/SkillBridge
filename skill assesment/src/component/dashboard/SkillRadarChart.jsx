import React from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer
} from 'recharts';

const SkillRadarChart = ({ skills }) => {
  // Group skills by category and calculate averages
  const categoryData = {};
  
  skills.forEach(skill => {
    if (!categoryData[skill.category]) {
      categoryData[skill.category] = { total: 0, count: 0 };
    }
    categoryData[skill.category].total += skill.level;
    categoryData[skill.category].count += 1;
  });

  const data = Object.entries(categoryData).map(([category, stats]) => ({
    category: category.replace('_', ' ').toUpperCase(),
    level: (stats.total / stats.count) * 100
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <RadarChart data={data}>
        <PolarGrid />
        <PolarAngleAxis dataKey="category" />
        <PolarRadiusAxis 
          angle={90} 
          domain={[0, 100]} 
          tick={{ fontSize: 12 }}
        />
        <Radar
          name="Skill Level"
          dataKey="level"
          stroke="#3b82f6"
          fill="#3b82f6"
          fillOpacity={0.3}
          strokeWidth={2}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
};

export default SkillRadarChart;
