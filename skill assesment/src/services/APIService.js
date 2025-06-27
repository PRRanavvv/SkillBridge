import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

class APIService {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor for authentication
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('authToken');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response.data,
      (error) => {
        console.error('API Error:', error);
        return Promise.reject(error);
      }
    );
  }

  // User Management
  async register(userData) {
    return this.client.post('/register', userData);
  }

  async login(credentials) {
    const response = await this.client.post('/login', credentials);
    if (response.token) {
      localStorage.setItem('authToken', response.token);
    }
    return response;
  }

  async logout() {
    localStorage.removeItem('authToken');
    return this.client.post('/logout');
  }

  // Profile Management
  async createUserProfile(profileData) {
    return this.client.post('/create_profile', profileData);
  }

  async getUserProfile(userId) {
    return this.client.get(`/profile/${userId}`);
  }

  async updateUserProfile(userId, profileData) {
    return this.client.put(`/profile/${userId}`, profileData);
  }

  // Skills and Assessment
  async uploadResume(formData) {
    return this.client.post('/upload_resume', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  }

  async conductAssessment(assessmentData) {
    return this.client.post('/conduct_assessment', assessmentData);
  }

  async updateSkillProgress(userId, skillName, progress) {
    return this.client.post('/update_progress', {
      user_id: userId,
      skill_name: skillName,
      new_level: progress,
    });
  }

  // Learning Path
  async generateLearningPath(userId, targetSkills) {
    return this.client.post('/generate_learning_path', {
      user_id: userId,
      target_skills: targetSkills,
    });
  }

  async getLearningResources(skillGaps) {
    return this.client.post('/get_resources', {
      skill_gaps: skillGaps,
    });
  }

  // Visualizations
  async generateVisualization(vizData) {
    return this.client.post('/generate_visualization', vizData);
  }

  // Dashboard
  async getDashboardData(userId) {
    return this.client.get(`/dashboard/${userId}`);
  }
}

export const APIService = new APIService();
export default APIService;
