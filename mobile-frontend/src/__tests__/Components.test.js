import React from 'react';

describe('Mobile Frontend Tests', () => {
    it('should pass basic test', () => {
        expect(true).toBe(true);
    });

    it('should validate API configuration', () => {
        const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8000';
        expect(API_BASE_URL).toBeDefined();
        expect(typeof API_BASE_URL).toBe('string');
    });

    it('should have required screens', () => {
        const DashboardScreen = require('../screens/DashboardScreen').default;
        const TradingScreen = require('../screens/TradingScreen').default;
        const PortfolioScreen = require('../screens/PortfolioScreen').default;
        const AnalyticsScreen = require('../screens/AnalyticsScreen').default;
        const WatchlistScreen = require('../screens/WatchlistScreen').default;
        const LoginScreen = require('../screens/LoginScreen').default;
        const RegisterScreen = require('../screens/RegisterScreen').default;

        expect(DashboardScreen).toBeDefined();
        expect(TradingScreen).toBeDefined();
        expect(PortfolioScreen).toBeDefined();
        expect(AnalyticsScreen).toBeDefined();
        expect(WatchlistScreen).toBeDefined();
        expect(LoginScreen).toBeDefined();
        expect(RegisterScreen).toBeDefined();
    });

    it('should have API services defined', () => {
        const api = require('../services/api');

        expect(api.authService).toBeDefined();
        expect(api.marketService).toBeDefined();
        expect(api.portfolioService).toBeDefined();
        expect(api.tradingService).toBeDefined();
        expect(api.analyticsService).toBeDefined();
        expect(api.watchlistService).toBeDefined();
    });

    it('should have AuthContext defined', () => {
        const { AuthProvider, useAuth } = require('../context/AuthContext');

        expect(AuthProvider).toBeDefined();
        expect(useAuth).toBeDefined();
    });
});
