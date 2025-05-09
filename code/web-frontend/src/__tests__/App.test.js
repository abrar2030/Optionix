import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from '../App';
import { BrowserRouter } from 'react-router-dom';
import { act } from 'react-dom/test-utils';

// Mock the API calls
const mockApi = {
  login: jest.fn(),
  register: jest.fn(),
  getUserProfile: jest.fn(),
};

jest.mock('../services/api', () => mockApi);

const renderWithRouter = (component) => {
  return render(
    <BrowserRouter>
      {component}
    </BrowserRouter>
  );
};

describe('App Component', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();
    // Reset localStorage
    localStorage.clear();
  });

  afterEach(() => {
    // Clean up after each test
    jest.resetAllMocks();
  });

  test('renders login page by default', () => {
    renderWithRouter(<App />);
    expect(screen.getByText(/login/i)).toBeInTheDocument();
  });

  test('handles successful login', async () => {
    mockApi.login.mockResolvedValueOnce({ token: 'fake-token' });
    mockApi.getUserProfile.mockResolvedValueOnce({ username: 'testuser' });
    
    renderWithRouter(<App />);
    
    fireEvent.change(screen.getByLabelText(/username/i), {
      target: { value: 'testuser' },
    });
    fireEvent.change(screen.getByLabelText(/password/i), {
      target: { value: 'password123' },
    });
    
    // Check loading state
    const loginButton = screen.getByRole('button', { name: /login/i });
    fireEvent.click(loginButton);
    expect(loginButton).toBeDisabled();
    
    await waitFor(() => {
      expect(mockApi.login).toHaveBeenCalledWith('testuser', 'password123');
    });

    // Check navigation after successful login
    await waitFor(() => {
      expect(screen.getByText(/dashboard/i)).toBeInTheDocument();
    });
  });

  test('handles login error', async () => {
    mockApi.login.mockRejectedValueOnce(new Error('Invalid credentials'));
    renderWithRouter(<App />);
    
    fireEvent.change(screen.getByLabelText(/username/i), {
      target: { value: 'wronguser' },
    });
    fireEvent.change(screen.getByLabelText(/password/i), {
      target: { value: 'wrongpass' },
    });
    
    const loginButton = screen.getByRole('button', { name: /login/i });
    fireEvent.click(loginButton);
    
    await waitFor(() => {
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
    });
    expect(loginButton).not.toBeDisabled();
  });

  test('validates login form', async () => {
    renderWithRouter(<App />);
    
    const loginButton = screen.getByRole('button', { name: /login/i });
    fireEvent.click(loginButton);
    
    expect(screen.getByText(/username is required/i)).toBeInTheDocument();
    expect(screen.getByText(/password is required/i)).toBeInTheDocument();
    expect(mockApi.login).not.toHaveBeenCalled();
  });

  test('navigates to registration page', () => {
    renderWithRouter(<App />);
    fireEvent.click(screen.getByText(/register/i));
    expect(screen.getByText(/create account/i)).toBeInTheDocument();
  });

  test('handles successful registration', async () => {
    mockApi.register.mockResolvedValueOnce({ success: true });
    renderWithRouter(<App />);
    
    // Navigate to registration
    fireEvent.click(screen.getByText(/register/i));
    
    // Fill registration form
    fireEvent.change(screen.getByLabelText(/username/i), {
      target: { value: 'newuser' },
    });
    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'newuser@example.com' },
    });
    fireEvent.change(screen.getByLabelText(/password/i), {
      target: { value: 'password123' },
    });
    
    const registerButton = screen.getByRole('button', { name: /register/i });
    fireEvent.click(registerButton);
    
    // Check loading state
    expect(registerButton).toBeDisabled();
    
    await waitFor(() => {
      expect(mockApi.register).toHaveBeenCalledWith({
        username: 'newuser',
        email: 'newuser@example.com',
        password: 'password123',
      });
    });

    // Check navigation after successful registration
    await waitFor(() => {
      expect(screen.getByText(/login/i)).toBeInTheDocument();
    });
  });

  test('validates registration form', async () => {
    renderWithRouter(<App />);
    
    // Navigate to registration
    fireEvent.click(screen.getByText(/register/i));
    
    const registerButton = screen.getByRole('button', { name: /register/i });
    fireEvent.click(registerButton);
    
    expect(screen.getByText(/username is required/i)).toBeInTheDocument();
    expect(screen.getByText(/email is required/i)).toBeInTheDocument();
    expect(screen.getByText(/password is required/i)).toBeInTheDocument();
    expect(mockApi.register).not.toHaveBeenCalled();
  });

  test('validates email format in registration', async () => {
    renderWithRouter(<App />);
    
    // Navigate to registration
    fireEvent.click(screen.getByText(/register/i));
    
    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'invalid-email' },
    });
    
    const registerButton = screen.getByRole('button', { name: /register/i });
    fireEvent.click(registerButton);
    
    expect(screen.getByText(/invalid email format/i)).toBeInTheDocument();
    expect(mockApi.register).not.toHaveBeenCalled();
  });
}); 