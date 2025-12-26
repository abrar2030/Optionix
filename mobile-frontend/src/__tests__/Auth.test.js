import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { NavigationContainer } from '@react-navigation/native';
import LoginScreen from '../screens/LoginScreen';
import RegisterScreen from '../screens/RegisterScreen';
import { AuthProvider } from '../context/AuthContext';

// Mock the API
jest.mock('../services/api');

const renderWithProviders = (component) => {
    return render(
        <NavigationContainer>
            <AuthProvider>{component}</AuthProvider>
        </NavigationContainer>,
    );
};

describe('LoginScreen', () => {
    it('renders login form correctly', () => {
        const { getByText, getByPlaceholderText } = renderWithProviders(
            <LoginScreen navigation={{ navigate: jest.fn() }} />,
        );

        expect(getByText('Welcome to Optionix')).toBeTruthy();
        expect(getByText('Sign In')).toBeTruthy();
    });

    it('validates email format', async () => {
        const { getByLabelText, getByText } = renderWithProviders(
            <LoginScreen navigation={{ navigate: jest.fn() }} />,
        );

        const emailInput = getByLabelText('Email');
        const loginButton = getByText('Sign In');

        fireEvent.changeText(emailInput, 'invalid-email');
        fireEvent.press(loginButton);

        await waitFor(() => {
            expect(getByText('Email is invalid')).toBeTruthy();
        });
    });

    it('validates required fields', async () => {
        const { getByText } = renderWithProviders(
            <LoginScreen navigation={{ navigate: jest.fn() }} />,
        );

        const loginButton = getByText('Sign In');
        fireEvent.press(loginButton);

        await waitFor(() => {
            expect(getByText('Email is required')).toBeTruthy();
            expect(getByText('Password is required')).toBeTruthy();
        });
    });
});

describe('RegisterScreen', () => {
    it('renders registration form correctly', () => {
        const { getByText } = renderWithProviders(
            <RegisterScreen navigation={{ navigate: jest.fn() }} />,
        );

        expect(getByText('Create Account')).toBeTruthy();
        expect(getByText('Sign Up')).toBeTruthy();
    });

    it('validates password strength', async () => {
        const { getByLabelText, getByText } = renderWithProviders(
            <RegisterScreen navigation={{ navigate: jest.fn() }} />,
        );

        const passwordInput = getByLabelText('Password');
        const registerButton = getByText('Sign Up');

        fireEvent.changeText(passwordInput, 'weak');
        fireEvent.press(registerButton);

        await waitFor(() => {
            expect(getByText(/Password must/)).toBeTruthy();
        });
    });

    it('validates password confirmation', async () => {
        const { getByLabelText, getByText } = renderWithProviders(
            <RegisterScreen navigation={{ navigate: jest.fn() }} />,
        );

        const passwordInput = getByLabelText('Password');
        const confirmInput = getByLabelText('Confirm Password');
        const registerButton = getByText('Sign Up');

        fireEvent.changeText(passwordInput, 'ValidPass123');
        fireEvent.changeText(confirmInput, 'DifferentPass123');
        fireEvent.press(registerButton);

        await waitFor(() => {
            expect(getByText('Passwords do not match')).toBeTruthy();
        });
    });
});
