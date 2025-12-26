import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import api from './api';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Check if user is already logged in on mount
    useEffect(() => {
        const checkAuth = async () => {
            const token = localStorage.getItem('auth_token');
            if (token) {
                try {
                    // Verify token is still valid by making a test request
                    const response = await api.get('/auth/me');
                    setUser(response.data);
                } catch (err) {
                    // Token is invalid, clear it
                    localStorage.removeItem('auth_token');
                    setUser(null);
                }
            }
            setLoading(false);
        };

        checkAuth();
    }, []);

    const login = useCallback(async (email, password) => {
        try {
            setLoading(true);
            setError(null);

            // Create FormData for login
            const formData = new FormData();
            formData.append('username', email);
            formData.append('password', password);

            const response = await api.post('/auth/login', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            const { access_token, user: userData } = response.data;

            // Store token
            localStorage.setItem('auth_token', access_token);

            // Set user data
            setUser(userData);

            return { success: true };
        } catch (err) {
            const errorMessage =
                err.response?.data?.detail ||
                err.message ||
                'Login failed. Please check your credentials.';
            setError(errorMessage);
            return { success: false, error: errorMessage };
        } finally {
            setLoading(false);
        }
    }, []);

    const register = useCallback(async (email, password, fullName) => {
        try {
            setLoading(true);
            setError(null);

            const response = await api.post('/auth/register', {
                email,
                password,
                full_name: fullName,
            });

            const { access_token, user: userData } = response.data;

            // Store token
            localStorage.setItem('auth_token', access_token);

            // Set user data
            setUser(userData);

            return { success: true };
        } catch (err) {
            const errorMessage =
                err.response?.data?.detail ||
                err.message ||
                'Registration failed. Please try again.';
            setError(errorMessage);
            return { success: false, error: errorMessage };
        } finally {
            setLoading(false);
        }
    }, []);

    const logout = useCallback(() => {
        localStorage.removeItem('auth_token');
        setUser(null);
        setError(null);
    }, []);

    const value = {
        user,
        loading,
        error,
        login,
        register,
        logout,
        isAuthenticated: !!user,
    };

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};

export default AuthContext;
