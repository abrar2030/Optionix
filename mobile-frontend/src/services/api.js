import axios from 'axios';
import Constants from 'expo-constants';

// Define the base URL for the backend API
// Priority: Environment variable > Constants > Default
const API_BASE_URL =
    Constants.expoConfig?.extra?.apiBaseUrl || process.env.API_BASE_URL || 'http://localhost:8000';

const API_TIMEOUT = Constants.expoConfig?.extra?.apiTimeout || process.env.API_TIMEOUT || 30000;

const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: parseInt(API_TIMEOUT),
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor for adding auth token
api.interceptors.request.use(
    async (config) => {
        try {
            // Try to get token from AsyncStorage if available
            const AsyncStorage = require('@react-native-async-storage/async-storage').default;
            const token = await AsyncStorage.getItem('authToken');
            if (token) {
                config.headers.Authorization = `Bearer ${token}`;
            }
        } catch (error) {
            // If AsyncStorage is not available or fails, continue without token
            console.warn('Could not retrieve auth token:', error);
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    },
);

// Response interceptor for error handling
api.interceptors.response.use(
    (response) => response,
    async (error) => {
        if (error.response?.status === 401) {
            // Handle unauthorized access
            try {
                const AsyncStorage = require('@react-native-async-storage/async-storage').default;
                await AsyncStorage.removeItem('authToken');
            } catch (e) {
                console.warn('Could not remove auth token:', e);
            }
        }
        return Promise.reject(error);
    },
);

// Authentication services
export const authService = {
    login: async (email, password) => {
        try {
            const response = await api.post('/auth/login', { email, password });
            return response.data;
        } catch (error) {
            console.error('Error during login:', error);
            throw error;
        }
    },
    register: async (userData) => {
        try {
            const response = await api.post('/auth/register', userData);
            return response.data;
        } catch (error) {
            console.error('Error during registration:', error);
            throw error;
        }
    },
    logout: async () => {
        try {
            const response = await api.post('/auth/logout');
            return response.data;
        } catch (error) {
            console.error('Error during logout:', error);
            throw error;
        }
    },
    getUserProfile: async () => {
        try {
            const response = await api.get('/auth/profile');
            return response.data;
        } catch (error) {
            console.error('Error fetching user profile:', error);
            throw error;
        }
    },
};

// Market data services
export const marketService = {
    getMarketOverview: async () => {
        try {
            const response = await api.get('/market/overview');
            return response.data;
        } catch (error) {
            console.error('Error fetching market overview:', error);
            throw error;
        }
    },
    getPriceHistory: async (symbol, timeframe = '1d') => {
        try {
            const response = await api.get(`/market/price-history/${symbol}`, {
                params: { timeframe },
            });
            return response.data;
        } catch (error) {
            console.error('Error fetching price history:', error);
            throw error;
        }
    },
    getOptionChain: async (symbol, expiry) => {
        try {
            const response = await api.get(`/market/option-chain/${symbol}`, {
                params: { expiry },
            });
            return response.data;
        } catch (error) {
            console.error('Error fetching option chain:', error);
            throw error;
        }
    },
    getOrderBook: async (symbol) => {
        try {
            const response = await api.get(`/market/order-book/${symbol}`);
            return response.data;
        } catch (error) {
            console.error('Error fetching order book:', error);
            throw error;
        }
    },
    predictVolatility: async (data) => {
        try {
            const response = await api.post('/market/volatility', data);
            return response.data;
        } catch (error) {
            console.error('Error predicting volatility:', error);
            throw error;
        }
    },
};

// Portfolio services
export const portfolioService = {
    getPortfolioSummary: async () => {
        try {
            const response = await api.get('/portfolio/summary');
            return response.data;
        } catch (error) {
            console.error('Error fetching portfolio summary:', error);
            throw error;
        }
    },
    getPositions: async () => {
        try {
            const response = await api.get('/portfolio/positions');
            return response.data;
        } catch (error) {
            console.error('Error fetching positions:', error);
            throw error;
        }
    },
    getPositionHealth: async (address) => {
        try {
            const response = await api.get(`/position_health/${address}`);
            return response.data;
        } catch (error) {
            console.error('Error fetching position health:', error);
            throw error;
        }
    },
    getTransactionHistory: async () => {
        try {
            const response = await api.get('/portfolio/transactions');
            return response.data;
        } catch (error) {
            console.error('Error fetching transaction history:', error);
            throw error;
        }
    },
    getPerformanceHistory: async (timeframe = '1M') => {
        try {
            const response = await api.get('/portfolio/performance', {
                params: { timeframe },
            });
            return response.data;
        } catch (error) {
            console.error('Error fetching performance history:', error);
            throw error;
        }
    },
};

// Trading services
export const tradingService = {
    executeTrade: async (tradeData) => {
        try {
            const response = await api.post('/trading/execute', tradeData);
            return response.data;
        } catch (error) {
            console.error('Error executing trade:', error);
            throw error;
        }
    },
    calculateOptionPrice: async (optionData) => {
        try {
            const response = await api.post('/trading/calculate-price', optionData);
            return response.data;
        } catch (error) {
            console.error('Error calculating option price:', error);
            throw error;
        }
    },
    calculateGreeks: async (optionData) => {
        try {
            const response = await api.post('/trading/calculate-greeks', optionData);
            return response.data;
        } catch (error) {
            console.error('Error calculating greeks:', error);
            throw error;
        }
    },
};

// Analytics services
export const analyticsService = {
    getRiskAssessment: async () => {
        try {
            const response = await api.get('/analytics/risk-assessment');
            return response.data;
        } catch (error) {
            console.error('Error fetching risk assessment:', error);
            throw error;
        }
    },
    getVolatilityAnalysis: async (symbol, timeframe = '1M') => {
        try {
            const response = await api.get('/analytics/volatility', {
                params: { symbol, timeframe },
            });
            return response.data;
        } catch (error) {
            console.error('Error fetching volatility analysis:', error);
            throw error;
        }
    },
    getMarketSentiment: async () => {
        try {
            const response = await api.get('/analytics/market-sentiment');
            return response.data;
        } catch (error) {
            console.error('Error fetching market sentiment:', error);
            throw error;
        }
    },
};

// Watchlist services
export const watchlistService = {
    getWatchlist: async () => {
        try {
            const response = await api.get('/watchlist');
            return response.data;
        } catch (error) {
            console.error('Error fetching watchlist:', error);
            throw error;
        }
    },
    addToWatchlist: async (symbol) => {
        try {
            const response = await api.post('/watchlist', { symbol });
            return response.data;
        } catch (error) {
            console.error('Error adding to watchlist:', error);
            throw error;
        }
    },
    removeFromWatchlist: async (symbol) => {
        try {
            const response = await api.delete(`/watchlist/${symbol}`);
            return response.data;
        } catch (error) {
            console.error('Error removing from watchlist:', error);
            throw error;
        }
    },
    getSymbolQuote: async (symbol) => {
        try {
            const response = await api.get(`/market/quote/${symbol}`);
            return response.data;
        } catch (error) {
            console.error('Error fetching symbol quote:', error);
            throw error;
        }
    },
};

export default api;
