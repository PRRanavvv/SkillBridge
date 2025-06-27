import React from 'react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend
} from 'recharts';

const CategoryDistributionChart = ({ skills }) => {
  // Group skills by category
  const categoryData = skills.reduce((acc, skill) => {
    const category = skill.category || 'Other';
    acc[category] = (acc[category] || 0) + 1;
    return acc;
  }, {});

  const data = Object.entries(categoryData).map(([category, count]) => ({
    name: category.replace('_', ' ').toUpperCase(),
    value: count,
    percentage: ((count / skills.length) * 100).toFixed(1)
  }));

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'];

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          outerRadius={100}
          fill="#8884d8"
          dataKey="value"
          label={({ name, percentage }) => `${name}: ${percentage}%`}
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip formatter={(value, name) => [value, 'Skills']} />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
};

export default CategoryDistributionChart;
