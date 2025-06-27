import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';

const SkillProgressChart = ({ progressData }) => {
  if (!progressData || progressData.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">No progress data available</p>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={progressData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis domain={[0, 100]} />
        <Tooltip formatter={(value) => [`${value}%`, 'Progress']} />
        <Legend />
        {Object.keys(progressData[0])
          .filter(key => key !== 'date')
          .map((skill, index) => (
            <Line
              key={skill}
              type="monotone"
              dataKey={skill}
              stroke={`hsl(${index * 60}, 70%, 50%)`}
              strokeWidth={2}
              dot={{ r: 4 }}
            />
          ))}
      </LineChart>
    </ResponsiveContainer>
  );
};

export default SkillProgressChart;
