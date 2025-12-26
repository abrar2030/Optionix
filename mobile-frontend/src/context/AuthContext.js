import React, { createContext, useState, useContext, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { authService } from '../services/api';

const AuthContext = createContext({});

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Check for existing auth token on app load
    useEffect(() => {
        checkAuth();
    }, []);

    const checkAuth = async () => {
        try {
            const token = await AsyncStorage.getItem('authToken');
            if (token) {
                // Verify token and get user profile
                const userProfile = await authService.getUserProfile();
                setUser(userProfile);
            }
        } catch (err) {
            console.error('Auth check failed:', err);
            // Clear invalid token
            await AsyncStorage.removeItem('authToken');
        } finally {
            setLoading(false);
        }
    };

    const login = async (email, password) => {
        try {
            setLoading(true);
            setError(null);
            const response = await authService.login(email, password);

            // Store token
            if (response.token || response.access_token) {
                await AsyncStorage.setItem('authToken', response.token || response.access_token);
            }

            // Get user profile
            const userProfile = await authService.getUserProfile();
            setUser(userProfile);

            return { success: true };
        } catch (err) {
            const errorMessage =
                err.response?.data?.message ||
                err.response?.data?.detail ||
                'Login failed. Please check your credentials.';
            setError(errorMessage);
            throw new Error(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    const register = async (userData) => {
        try {
            setLoading(true);
            setError(null);
            const response = await authService.register(userData);

            // Some APIs return token on registration, others require login
            if (response.token || response.access_token) {
                await AsyncStorage.setItem('authToken', response.token || response.access_token);
                const userProfile = await authService.getUserProfile();
                setUser(userProfile);
            }

            return { success: true, requiresLogin: !(response.token || response.access_token) };
        } catch (err) {
            const errorMessage =
                err.response?.data?.message ||
                err.response?.data?.detail ||
                'Registration failed. Please try again.';
            setError(errorMessage);
            throw new Error(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    const logout = async () => {
        try {
            setLoading(true);
            await authService.logout();
        } catch (err) {
            console.error('Logout API call failed:', err);
            // Continue with local logout even if API call fails
        } finally {
            await AsyncStorage.removeItem('authToken');
            setUser(null);
            setError(null);
            setLoading(false);
        }
    };

    const clearError = () => {
        setError(null);
    };

    return (
        <AuthContext.Provider
            value={{
                user,
                loading,
                error,
                login,
                register,
                logout,
                clearError,
                isAuthenticated: !!user,
            }}
        >
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};

export default AuthContext;
