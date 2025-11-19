import { render, fireEvent, waitFor } from '@testing-library/react-native';
import App from '../App';
import { NavigationContainer } from '@react-navigation/native';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Mock the API calls
const mockApi = {
  login: jest.fn(),
  register: jest.fn(),
  getUserProfile: jest.fn(),
};

jest.mock('../services/api', () => mockApi);

const renderWithNavigation = (component) => {
  return render(
    <NavigationContainer>
      {component}
    </NavigationContainer>
  );
};

describe('Mobile App Component', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();
    // Clear AsyncStorage
    AsyncStorage.clear();
  });

  afterEach(() => {
    // Clean up after each test
    jest.resetAllMocks();
  });

  test('renders login screen by default', () => {
    const { getByText } = renderWithNavigation(<App />);
    expect(getByText('Login')).toBeTruthy();
  });

  test('handles successful login', async () => {
    mockApi.login.mockResolvedValueOnce({ token: 'fake-token' });
    mockApi.getUserProfile.mockResolvedValueOnce({ username: 'testuser' });

    const { getByPlaceholderText, getByText } = renderWithNavigation(<App />);

    const usernameInput = getByPlaceholderText('Username');
    const passwordInput = getByPlaceholderText('Password');
    const loginButton = getByText('Login');

    fireEvent.changeText(usernameInput, 'testuser');
    fireEvent.changeText(passwordInput, 'password123');

    // Check loading state
    fireEvent.press(loginButton);
    expect(loginButton.props.disabled).toBeTruthy();

    await waitFor(() => {
      expect(mockApi.login).toHaveBeenCalledWith('testuser', 'password123');
    });

    // Check navigation after successful login
    await waitFor(() => {
      expect(getByText('Dashboard')).toBeTruthy();
    });
  });

  test('handles login error', async () => {
    mockApi.login.mockRejectedValueOnce(new Error('Invalid credentials'));
    const { getByPlaceholderText, getByText } = renderWithNavigation(<App />);

    const usernameInput = getByPlaceholderText('Username');
    const passwordInput = getByPlaceholderText('Password');
    const loginButton = getByText('Login');

    fireEvent.changeText(usernameInput, 'wronguser');
    fireEvent.changeText(passwordInput, 'wrongpass');
    fireEvent.press(loginButton);

    await waitFor(() => {
      expect(getByText('Invalid credentials')).toBeTruthy();
    });
    expect(loginButton.props.disabled).toBeFalsy();
  });

  test('validates login form', async () => {
    const { getByText } = renderWithNavigation(<App />);

    const loginButton = getByText('Login');
    fireEvent.press(loginButton);

    expect(getByText('Username is required')).toBeTruthy();
    expect(getByText('Password is required')).toBeTruthy();
    expect(mockApi.login).not.toHaveBeenCalled();
  });

  test('navigates to registration screen', () => {
    const { getByText } = renderWithNavigation(<App />);
    fireEvent.press(getByText('Register'));
    expect(getByText('Create Account')).toBeTruthy();
  });

  test('handles successful registration', async () => {
    mockApi.register.mockResolvedValueOnce({ success: true });
    const { getByText, getByPlaceholderText } = renderWithNavigation(<App />);

    // Navigate to registration
    fireEvent.press(getByText('Register'));

    // Fill registration form
    const usernameInput = getByPlaceholderText('Username');
    const emailInput = getByPlaceholderText('Email');
    const passwordInput = getByPlaceholderText('Password');
    const registerButton = getByText('Register');

    fireEvent.changeText(usernameInput, 'newuser');
    fireEvent.changeText(emailInput, 'newuser@example.com');
    fireEvent.changeText(passwordInput, 'password123');

    // Check loading state
    fireEvent.press(registerButton);
    expect(registerButton.props.disabled).toBeTruthy();

    await waitFor(() => {
      expect(mockApi.register).toHaveBeenCalledWith({
        username: 'newuser',
        email: 'newuser@example.com',
        password: 'password123',
      });
    });

    // Check navigation after successful registration
    await waitFor(() => {
      expect(getByText('Login')).toBeTruthy();
    });
  });

  test('validates registration form', async () => {
    const { getByText } = renderWithNavigation(<App />);

    // Navigate to registration
    fireEvent.press(getByText('Register'));

    const registerButton = getByText('Register');
    fireEvent.press(registerButton);

    expect(getByText('Username is required')).toBeTruthy();
    expect(getByText('Email is required')).toBeTruthy();
    expect(getByText('Password is required')).toBeTruthy();
    expect(mockApi.register).not.toHaveBeenCalled();
  });

  test('validates email format in registration', async () => {
    const { getByText, getByPlaceholderText } = renderWithNavigation(<App />);

    // Navigate to registration
    fireEvent.press(getByText('Register'));

    const emailInput = getByPlaceholderText('Email');
    fireEvent.changeText(emailInput, 'invalid-email');

    const registerButton = getByText('Register');
    fireEvent.press(registerButton);

    expect(getByText('Invalid email format')).toBeTruthy();
    expect(mockApi.register).not.toHaveBeenCalled();
  });

  test('persists authentication state', async () => {
    // Mock AsyncStorage to return a stored token
    AsyncStorage.getItem.mockResolvedValueOnce('stored-token');
    mockApi.getUserProfile.mockResolvedValueOnce({ username: 'testuser' });

    const { getByText } = renderWithNavigation(<App />);

    // Should navigate to dashboard if token exists
    await waitFor(() => {
      expect(getByText('Dashboard')).toBeTruthy();
    });
  });
});
