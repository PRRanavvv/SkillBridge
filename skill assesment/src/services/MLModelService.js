import * as tf from '@tensorflow/tfjs';
import { toast } from 'react-hot-toast';

export class MLModelService {
  constructor() {
    this.models = {};
    this.isInitialized = false;
  }

  async initialize() {
    try {
      console.log('Initializing ML Model Service...');
      
      // Load pre-trained models
      await this.loadSkillAssessmentModel();
      await this.loadRecommendationModel();
      
      this.isInitialized = true;
      console.log('ML Model Service initialized successfully');
    } catch (error) {
      console.error('Failed to initialize ML Model Service:', error);
      throw error;
    }
  }

  async loadSkillAssessmentModel() {
    try {
      // Create a simple neural network for skill assessment
      const model = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [10], units: 64, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.3 }),
          tf.layers.dense({ units: 32, activation: 'relu' }),
          tf.layers.dense({ units: 4, activation: 'softmax' }) // 4 difficulty levels
        ]
      });

      model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });

      this.models.skillAssessment = model;
      console.log('Skill Assessment Model loaded');
    } catch (error) {
      console.error('Failed to load skill assessment model:', error);
    }
  }

  async loadRecommendationModel() {
    try {
      // Create recommendation model
      const model = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [50], units: 128, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.4 }),
          tf.layers.dense({ units: 64, activation: 'relu' }),
          tf.layers.dense({ units: 32, activation: 'relu' }),
          tf.layers.dense({ units: 1, activation: 'sigmoid' })
        ]
      });

      model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
      });

      this.models.recommendation = model;
      console.log('Recommendation Model loaded');
    } catch (error) {
      console.error('Failed to load recommendation model:', error);
    }
  }

  async extractSkillsFromText(text) {
    try {
      // Simulate skill extraction using TensorFlow.js
      const skills = [];
      const skillKeywords = {
        'programming': ['python', 'javascript', 'java', 'react', 'angular', 'vue'],
        'data_science': ['machine learning', 'tensorflow', 'pytorch', 'pandas', 'numpy'],
        'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
        'database': ['sql', 'mongodb', 'postgresql', 'mysql']
      };

      const textLower = text.toLowerCase();
      
      for (const [category, keywords] of Object.entries(skillKeywords)) {
        for (const keyword of keywords) {
          if (textLower.includes(keyword)) {
            const confidence = Math.random() * 0.4 + 0.6; // 0.6-1.0
            const level = Math.random() * 0.5 + 0.5; // 0.5-1.0
            
            skills.push({
              name: keyword,
              category,
              level,
              confidence,
              source: 'resume'
            });
          }
        }
      }

      return skills;
    } catch (error) {
      console.error('Error extracting skills:', error);
      return [];
    }
  }

  async assessSkill(skillName, userResponses) {
    try {
      if (!this.models.skillAssessment) {
        throw new Error('Skill assessment model not loaded');
      }

      // Convert responses to tensor
      const inputData = this.preprocessAssessmentData(userResponses);
      const inputTensor = tf.tensor2d([inputData]);

      // Make prediction
      const prediction = this.models.skillAssessment.predict(inputTensor);
      const predictionData = await prediction.data();

      // Clean up tensors
      inputTensor.dispose();
      prediction.dispose();

      // Convert to skill level (0-1)
      const skillLevel = this.convertPredictionToLevel(predictionData);
      
      return {
        skillName,
        estimatedLevel: skillLevel,
        confidence: Math.max(...predictionData),
        recommendations: this.generateSkillRecommendations(skillName, skillLevel)
      };
    } catch (error) {
      console.error('Error in skill assessment:', error);
      throw error;
    }
  }

  preprocessAssessmentData(responses) {
    // Convert user responses to numerical features
    const features = new Array(10).fill(0);
    
    responses.forEach((response, index) => {
      if (index < 10) {
        features[index] = response.correct ? 1 : 0;
      }
    });

    return features;
  }

  convertPredictionToLevel(predictionData) {
    // Convert softmax output to skill level
    const levels = [0.25, 0.5, 0.75, 1.0]; // beginner, intermediate, advanced, expert
    let weightedSum = 0;
    
    for (let i = 0; i < predictionData.length; i++) {
      weightedSum += predictionData[i] * levels[i];
    }
    
    return Math.min(1.0, Math.max(0.0, weightedSum));
  }

  generateSkillRecommendations(skillName, level) {
    const recommendations = [];
    
    if (level < 0.4) {
      recommendations.push(`Start with ${skillName} fundamentals`);
      recommendations.push('Practice basic concepts daily');
    } else if (level < 0.7) {
      recommendations.push(`Build intermediate ${skillName} projects`);
      recommendations.push('Join online communities and forums');
    } else {
      recommendations.push(`Contribute to open source ${skillName} projects`);
      recommendations.push('Mentor others and teach concepts');
    }

    return recommendations;
  }

  async generateLearningPath(currentSkills, targetSkills, preferences = {}) {
    try {
      // Simulate learning path generation
      const skillGaps = this.identifySkillGaps(currentSkills, targetSkills);
      const resources = await this.recommendResources(skillGaps, preferences);
      
      return {
        skillGaps,
        recommendedResources: resources,
        estimatedDuration: this.calculateDuration(resources),
        priorityOrder: this.prioritizeSkills(skillGaps, currentSkills)
      };
    } catch (error) {
      console.error('Error generating learning path:', error);
      throw error;
    }
  }

  identifySkillGaps(currentSkills, targetSkills) {
    const currentSkillNames = currentSkills.map(s => s.name.toLowerCase());
    const gaps = targetSkills.filter(target => 
      !currentSkillNames.includes(target.toLowerCase())
    );
    
    return gaps;
  }

  async recommendResources(skillGaps, preferences) {
    const resources = [];
    const resourceDatabase = {
      'python': [
        {
          title: 'Python Fundamentals Course',
          description: 'Learn Python basics with hands-on projects',
          difficulty: 'beginner',
          duration: 180,
          rating: 4.5,
          url: 'https://example.com/python-course'
        }
      ],
      'machine learning': [
        {
          title: 'Machine Learning with TensorFlow',
          description: 'Build ML models from scratch',
          difficulty: 'intermediate',
          duration: 300,
          rating: 4.7,
          url: 'https://example.com/ml-course'
        }
      ]
    };

    for (const skill of skillGaps) {
      const skillResources = resourceDatabase[skill.toLowerCase()] || [];
      resources.push(...skillResources);
    }

    return resources;
  }

  calculateDuration(resources) {
    const totalMinutes = resources.reduce((sum, resource) => sum + resource.duration, 0);
    return Math.ceil(totalMinutes / 60); // Convert to hours
  }

  prioritizeSkills(skillGaps, currentSkills) {
    // Simple prioritization based on prerequisites
    const prerequisites = {
      'machine learning': ['python', 'statistics'],
      'deep learning': ['machine learning', 'tensorflow'],
      'react': ['javascript', 'html', 'css']
    };

    const prioritized = [];
    const remaining = [...skillGaps];

    while (remaining.length > 0) {
      for (let i = 0; i < remaining.length; i++) {
        const skill = remaining[i];
        const prereqs = prerequisites[skill.toLowerCase()] || [];
        
        const hasPrereqs = prereqs.every(prereq => 
          currentSkills.some(cs => cs.name.toLowerCase() === prereq) ||
          prioritized.includes(prereq)
        );

        if (hasPrereqs) {
          prioritized.push(skill);
          remaining.splice(i, 1);
          break;
        }
      }

      // Prevent infinite loop
      if (remaining.length > 0 && prioritized.length === 0) {
        prioritized.push(remaining.shift());
      }
    }

    return prioritized;
  }

  async processResumeFile(file) {
    try {
      const text = await this.extractTextFromFile(file);
      const skills = await this.extractSkillsFromText(text);
      
      return {
        extractedText: text,
        skills,
        processedAt: new Date().toISOString()
      };
    } catch (error) {
      console.error('Error processing resume:', error);
      throw error;
    }
  }

  async extractTextFromFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (event) => {
        try {
          const text = event.target.result;
          resolve(text);
        } catch (error) {
          reject(error);
        }
      };
      
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  }
}
