class AuthService {
  constructor() {
    this.currentUser = null;
    this.isAuthenticated = false;
    this.loadUserFromStorage();
  }

  loadUserFromStorage() {
    try {
      const userData = localStorage.getItem('currentUser');
      const token = localStorage.getItem('authToken');
      
      if (userData && token) {
        this.currentUser = JSON.parse(userData);
        this.isAuthenticated = true;
      }
    } catch (error) {
      console.error('Error loading user from storage:', error);
      this.logout();
    }
  }

  login(userData, token) {
    this.currentUser = userData;
    this.isAuthenticated = true;
    
    localStorage.setItem('currentUser', JSON.stringify(userData));
    localStorage.setItem('authToken', token);
    
    return userData;
  }

  logout() {
    this.currentUser = null;
    this.isAuthenticated = false;
    
    localStorage.removeItem('currentUser');
    localStorage.removeItem('authToken');
  }

  getCurrentUser() {
    return this.currentUser;
  }

  isUserAuthenticated() {
    return this.isAuthenticated;
  }

  getAuthToken() {
    return localStorage.getItem('authToken');
  }

  updateUser(userData) {
    if (this.isAuthenticated) {
      this.currentUser = { ...this.currentUser, ...userData };
      localStorage.setItem('currentUser', JSON.stringify(this.currentUser));
    }
  }
}

export const AuthService = new AuthService();
export default AuthService;
