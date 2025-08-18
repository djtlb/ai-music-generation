import { useState, useEffect, createContext, useContext } from 'react';

// Create an authentication context
const AuthContext = createContext(null);

/**
 * Authentication provider component
 * In a real app, this would interact with your authentication API
 */
export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  
  // Check for stored user on mount
  useEffect(() => {
    const storedUser = localStorage.getItem('beatAddicts_user');
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser));
      } catch (e) {
        console.error('Failed to parse stored user:', e);
        localStorage.removeItem('beatAddicts_user');
      }
    }
    setIsLoading(false);
  }, []);
  
  // Save user to storage when it changes
  useEffect(() => {
    if (user) {
      localStorage.setItem('beatAddicts_user', JSON.stringify(user));
    } else {
      localStorage.removeItem('beatAddicts_user');
    }
  }, [user]);
  
  /**
   * Sign in a user
   * In a real app, this would call your authentication API
   */
  const signIn = async (credentials) => {
    setIsLoading(true);
    
    try {
      // Mock authentication (replace with actual API call)
      // In production, this would be an API request to your auth server
      const mockedUser = {
        id: 'user_' + Math.random().toString(36).substr(2, 9),
        name: credentials.email.split('@')[0],
        email: credentials.email,
        role: 'user',
        subscriptionTier: 'free',
        createdAt: new Date().toISOString(),
      };
      
      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 500));
      
      setUser(mockedUser);
      return { success: true, user: mockedUser };
    } catch (error) {
      console.error('Sign in failed:', error);
      return { success: false, error: error.message || 'Authentication failed' };
    } finally {
      setIsLoading(false);
    }
  };
  
  /**
   * Sign up a new user
   * In a real app, this would call your registration API
   */
  const signUp = async (userData) => {
    setIsLoading(true);
    
    try {
      // Mock registration (replace with actual API call)
      const mockedUser = {
        id: 'user_' + Math.random().toString(36).substr(2, 9),
        name: userData.name || userData.email.split('@')[0],
        email: userData.email,
        role: 'user',
        subscriptionTier: 'free',
        createdAt: new Date().toISOString(),
      };
      
      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 800));
      
      setUser(mockedUser);
      return { success: true, user: mockedUser };
    } catch (error) {
      console.error('Sign up failed:', error);
      return { success: false, error: error.message || 'Registration failed' };
    } finally {
      setIsLoading(false);
    }
  };
  
  /**
   * Sign out the current user
   */
  const signOut = async () => {
    setIsLoading(true);
    
    try {
      // Perform any cleanup needed
      // In a real app, this might call an API to invalidate tokens
      
      // Clear user state
      setUser(null);
      return { success: true };
    } catch (error) {
      console.error('Sign out failed:', error);
      return { success: false, error: error.message };
    } finally {
      setIsLoading(false);
    }
  };
  
  /**
   * Update user profile information
   */
  const updateProfile = async (updates) => {
    setIsLoading(true);
    
    try {
      // In a real app, this would call your API
      
      // Update local user state
      const updatedUser = { ...user, ...updates };
      setUser(updatedUser);
      return { success: true, user: updatedUser };
    } catch (error) {
      console.error('Profile update failed:', error);
      return { success: false, error: error.message };
    } finally {
      setIsLoading(false);
    }
  };
  
  /**
   * Check if the user has a specific permission
   */
  const hasPermission = (permission) => {
    if (!user) return false;
    
    // In a real app, this would check against user roles and permissions
    const rolePermissions = {
      admin: ['*'], // Admin has all permissions
      user: ['view_project', 'edit_project', 'generate_music'],
      guest: ['view_project'],
    };
    
    // Get permissions for the user's role
    const userPermissions = rolePermissions[user.role] || [];
    
    // Check if the user has the specific permission or all permissions
    return userPermissions.includes('*') || userPermissions.includes(permission);
  };
  
  // Create a temporary user for demo purposes if none exists
  useEffect(() => {
    if (!isLoading && !user) {
      // Create a demo user automatically for this application
      const demoUser = {
        id: 'user_demo',
        name: 'Demo User',
        email: 'demo@beataddicts.com',
        role: 'user',
        subscriptionTier: 'pro', // Give pro features for demo
        createdAt: new Date().toISOString(),
      };
      
      setUser(demoUser);
    }
  }, [isLoading, user]);
  
  // Auth context value
  const value = {
    user,
    isLoading,
    signIn,
    signUp,
    signOut,
    updateProfile,
    hasPermission,
    isAuthenticated: !!user,
  };
  
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/**
 * Hook to use the auth context
 */
export function useUser() {
  const context = useContext(AuthContext);
  if (context === null) {
    throw new Error('useUser must be used within an AuthProvider');
  }
  return context;
}

export default useUser;
