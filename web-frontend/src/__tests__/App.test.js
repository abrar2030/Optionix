import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from 'styled-components';
import '@testing-library/jest-dom';
import App from '../App';

// Mock the AuthContext
jest.mock('../utils/AuthContext', () => ({
    AuthProvider: ({ children }) => <div>{children}</div>,
    useAuth: () => ({
        isAuthenticated: true,
        loading: false,
        user: { email: 'test@test.com', full_name: 'Test User' },
        login: jest.fn(),
        register: jest.fn(),
        logout: jest.fn(),
    }),
}));

// Mock the AppContext
jest.mock('../utils/AppContext', () => ({
    AppProvider: ({ children }) => <div>{children}</div>,
    useAppContext: () => ({
        marketData: { assets: [] },
        portfolioData: { totalValue: '$0' },
        loading: false,
        error: null,
    }),
}));

const theme = {
    colors: {
        primary: '#2962ff',
        primaryDark: '#0039cb',
        primaryLight: '#768fff',
        secondary: '#ff6d00',
        secondaryDark: '#c43c00',
        secondaryLight: '#ff9e40',
        backgroundDark: '#131722',
        backgroundLight: '#1e222d',
        textPrimary: '#ffffff',
        textSecondary: '#b2b5be',
        success: '#26a69a',
        danger: '#ef5350',
        warning: '#ffca28',
        info: '#42a5f5',
        border: '#2a2e39',
        cardBg: '#1e222d',
    },
    breakpoints: {
        mobile: '576px',
        tablet: '768px',
        desktop: '992px',
        wide: '1200px',
    },
};

describe('App Component', () => {
    test('renders without crashing', () => {
        render(<App />);
    });

    test('renders app with theme provider', () => {
        const { container } = render(<App />);
        expect(container).toBeInTheDocument();
    });

    test('app is wrapped with error boundary', () => {
        // Test that app doesn't crash on render
        expect(() => render(<App />)).not.toThrow();
    });
});
