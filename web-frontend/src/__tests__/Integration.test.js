import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from 'styled-components';
import '@testing-library/jest-dom';
import Login from '../pages/Login';
import { AuthProvider } from '../utils/AuthContext';

const theme = {
    colors: {
        primary: '#2962ff',
        primaryDark: '#0039cb',
        backgroundDark: '#131722',
        backgroundLight: '#1e222d',
        textPrimary: '#ffffff',
        textSecondary: '#b2b5be',
        danger: '#ef5350',
        border: '#2a2e39',
        cardBg: '#1e222d',
    },
    breakpoints: {
        mobile: '576px',
        tablet: '768px',
    },
};

const AllProviders = ({ children }) => (
    <ThemeProvider theme={theme}>
        <BrowserRouter>
            <AuthProvider>{children}</AuthProvider>
        </BrowserRouter>
    </ThemeProvider>
);

describe('Login Integration Tests', () => {
    test('renders login form', () => {
        render(<Login />, { wrapper: AllProviders });

        expect(screen.getByText(/Optionix/i)).toBeInTheDocument();
        expect(screen.getByLabelText(/Email/i)).toBeInTheDocument();
        expect(screen.getByLabelText(/Password/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Sign In/i })).toBeInTheDocument();
    });

    test('switches between login and register modes', () => {
        render(<Login />, { wrapper: AllProviders });

        const signUpLink = screen.getByText(/Sign Up/i);
        fireEvent.click(signUpLink);

        expect(screen.getByRole('button', { name: /Create Account/i })).toBeInTheDocument();
        expect(screen.getByLabelText(/Full Name/i)).toBeInTheDocument();
    });

    test('validates required fields', async () => {
        render(<Login />, { wrapper: AllProviders });

        const submitButton = screen.getByRole('button', { name: /Sign In/i });
        const emailInput = screen.getByLabelText(/Email/i);
        const passwordInput = screen.getByLabelText(/Password/i);

        expect(emailInput).toHaveAttribute('required');
        expect(passwordInput).toHaveAttribute('required');
    });

    test('allows input in form fields', () => {
        render(<Login />, { wrapper: AllProviders });

        const emailInput = screen.getByLabelText(/Email/i);
        const passwordInput = screen.getByLabelText(/Password/i);

        fireEvent.change(emailInput, { target: { value: 'test@example.com' } });
        fireEvent.change(passwordInput, { target: { value: 'password123' } });

        expect(emailInput).toHaveValue('test@example.com');
        expect(passwordInput).toHaveValue('password123');
    });
});
