import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import App from "../App";
import { act } from "react"; // Corrected import

// Mock for HTMLCanvasElement.getContext
if (typeof HTMLCanvasElement !== "undefined") {
  HTMLCanvasElement.prototype.getContext = () => {
    return {
      fillRect: jest.fn(),
      clearRect: jest.fn(),
      getImageData: jest.fn((x, y, w, h) => {
        return {
          data: new Uint8ClampedArray(w * h * 4),
        };
      }),
      putImageData: jest.fn(),
      createImageData: jest.fn(() => {
        return {
          data: new Uint8ClampedArray(0),
        };
      }),
      setTransform: jest.fn(),
      drawImage: jest.fn(),
      save: jest.fn(),
      fillText: jest.fn(),
      restore: jest.fn(),
      beginPath: jest.fn(),
      moveTo: jest.fn(),
      lineTo: jest.fn(),
      closePath: jest.fn(),
      stroke: jest.fn(),
      translate: jest.fn(),
      scale: jest.fn(),
      rotate: jest.fn(),
      arc: jest.fn(),
      fill: jest.fn(),
      measureText: jest.fn(() => {
        return { width: 0 };
      }),
      transform: jest.fn(),
      rect: jest.fn(),
      clip: jest.fn(),
    };
  };
}

// Mock react-chartjs-2 and chart.js
jest.mock("react-chartjs-2", () => ({
  Line: () => <div data-testid="mocked-line-chart">Mocked Line Chart</div>,
  Bar: () => <div data-testid="mocked-bar-chart">Mocked Bar Chart</div>,
  Doughnut: () => (
    <div data-testid="mocked-doughnut-chart">Mocked Doughnut Chart</div>
  ),
  Pie: () => <div data-testid="mocked-pie-chart">Mocked Pie Chart</div>,
}));

jest.mock("chart.js", () => {
  const mockChartInstance = {
    destroy: jest.fn(),
    update: jest.fn(),
  };
  const MockChart = jest.fn().mockImplementation(() => mockChartInstance);
  MockChart.register = jest.fn();
  return {
    Chart: MockChart,
    registerables: [],
  };
});

// Mock the API calls
const mockApi = {
  login: jest.fn(),
  register: jest.fn(),
  getUserProfile: jest.fn(),
};
jest.mock("../utils/api", () => {
  console.log("[DEBUG] jest.mock factory for ../utils/api CALLED");
  return { __esModule: true, default: mockApi };
});

// Mock AppContext - specifically the useAppContext hook
let mockUserValue = null;
let mockLoadingValue = false;
const mockSetUser = jest.fn((newUser) => {
  mockUserValue = newUser;
});
const mockSetLoading = jest.fn((newLoading) => {
  mockLoadingValue = newLoading;
});

jest.mock("../utils/AppContext", () => {
  const originalModule = jest.requireActual("../utils/AppContext");
  return {
    __esModule: true,
    ...originalModule, // Preserve original exports like AppProvider
    useAppContext: () => ({
      // Override only useAppContext
      user: mockUserValue,
      setUser: mockSetUser,
      loading: mockLoadingValue,
      setLoading: mockSetLoading,
    }),
  };
});

const renderApp = () => {
  return render(<App />);
};

describe("App Component", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();

    mockUserValue = null;
    mockLoadingValue = false;
    mockSetUser.mockImplementation((newUser) => {
      mockUserValue = newUser;
    });
    mockSetLoading.mockImplementation((newLoading) => {
      mockLoadingValue = newLoading;
    });

    mockApi.getUserProfile.mockResolvedValue(null);
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  test("renders login page by default", async () => {
    renderApp();
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: /login/i, level: 2 }),
      ).toBeInTheDocument(),
    );
  });

  test("handles successful login", async () => {
    mockApi.login.mockResolvedValueOnce({ token: "fake-token" });
    mockApi.getUserProfile.mockResolvedValueOnce({ username: "testuser" });

    const { container } = renderApp(); // Get container for form selection

    const usernameInput = await screen.findByLabelText(/username/i);
    const passwordInput = await screen.findByLabelText(/password/i);
    const loginButton = await screen.findByRole("button", { name: /login/i });
    const loginForm = container.querySelector("form"); // Select the form element

    await act(async () => {
      fireEvent.change(usernameInput, { target: { value: "testuser" } });
      fireEvent.change(passwordInput, { target: { value: "password123" } });
    });

    expect(loginButton).not.toBeDisabled();

    await act(async () => {
      fireEvent.submit(loginForm); // Use form element for submit
    });

    await waitFor(() => {
      expect(mockApi.login).toHaveBeenCalledWith("testuser", "password123");
    });

    await waitFor(() => {
      expect(mockSetUser).toHaveBeenCalledWith(
        expect.objectContaining({ username: "testuser" }),
      );
    });

    await waitFor(() => {
      expect(
        screen.queryByRole("heading", { name: /login/i, level: 2 }),
      ).not.toBeInTheDocument();
      expect(screen.getByText(/dashboard/i)).toBeInTheDocument();
    });
  });

  test("handles login error", async () => {
    mockApi.login.mockRejectedValueOnce(new Error("Invalid credentials"));
    const { container } = renderApp();

    const usernameInput = await screen.findByLabelText(/username/i);
    const passwordInput = await screen.findByLabelText(/password/i);
    const loginButton = await screen.findByRole("button", { name: /login/i });
    const loginForm = container.querySelector("form");

    await act(async () => {
      fireEvent.change(usernameInput, { target: { value: "wronguser" } });
      fireEvent.change(passwordInput, { target: { value: "wrongpass" } });
    });

    expect(loginButton).not.toBeDisabled();
    await act(async () => {
      fireEvent.submit(loginForm);
    });

    await waitFor(() => {
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
    });
    await waitFor(() => expect(loginButton).not.toBeDisabled());
  });

  test("validates login form", async () => {
    const { container } = renderApp();
    const loginButton = await screen.findByRole("button", { name: /login/i });
    const loginForm = container.querySelector("form");

    await act(async () => {
      fireEvent.submit(loginForm);
    });

    await waitFor(() =>
      expect(
        screen.getByText(/Username and Password are required/i),
      ).toBeInTheDocument(),
    );
    expect(mockApi.login).not.toHaveBeenCalled();
  });

  test("navigates to registration page", async () => {
    renderApp();
    const registerLink = await screen.findByText(/register/i, {
      selector: "span",
    });
    await act(async () => {
      fireEvent.click(registerLink);
    });
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: /create account/i, level: 2 }),
      ).toBeInTheDocument(),
    );
  });

  test("handles successful registration", async () => {
    mockApi.register.mockResolvedValueOnce({ success: true });
    const { container } = renderApp();

    const registerLink = await screen.findByText(/register/i, {
      selector: "span",
    });
    await act(async () => {
      fireEvent.click(registerLink);
    });

    const usernameInput = await screen.findByLabelText(/username/i);
    const emailInput = await screen.findByLabelText(/email/i);
    const passwordInput = await screen.findByLabelText(/password/i);
    const registerButton = await screen.findByRole("button", {
      name: /register/i,
    });
    // Assuming the registration form is the only form after clicking register link
    const registrationForm = container.querySelector("form");

    await act(async () => {
      fireEvent.change(usernameInput, { target: { value: "newuser" } });
      fireEvent.change(emailInput, {
        target: { value: "newuser@example.com" },
      });
      fireEvent.change(passwordInput, { target: { value: "password123" } });
    });

    expect(registerButton).not.toBeDisabled();
    await act(async () => {
      fireEvent.submit(registrationForm);
    });

    await waitFor(() => {
      expect(mockApi.register).toHaveBeenCalledWith({
        username: "newuser",
        email: "newuser@example.com",
        password: "password123",
      });
    });

    await waitFor(() => {
      expect(
        screen.getByText(/Registration successful! Please login./i),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("heading", { name: /login/i, level: 2 }),
      ).toBeInTheDocument();
    });
  });

  test("validates registration form for empty fields", async () => {
    const { container } = renderApp();
    const registerLink = await screen.findByText(/register/i, {
      selector: "span",
    });
    await act(async () => {
      fireEvent.click(registerLink);
    });

    const registerButton = await screen.findByRole("button", {
      name: /register/i,
    });
    const registrationForm = container.querySelector("form");

    await act(async () => {
      fireEvent.submit(registrationForm);
    });

    await waitFor(() =>
      expect(
        screen.getByText(/Username, Email, and Password are required/i),
      ).toBeInTheDocument(),
    );
    expect(mockApi.register).not.toHaveBeenCalled();
  });

  test("validates email format in registration", async () => {
    const { container } = renderApp();
    const registerLink = await screen.findByText(/register/i, {
      selector: "span",
    });
    await act(async () => {
      fireEvent.click(registerLink);
    });

    const usernameInput = await screen.findByLabelText(/username/i);
    const emailInput = await screen.findByLabelText(/email/i);
    const passwordInput = await screen.findByLabelText(/password/i);
    const registerButton = await screen.findByRole("button", {
      name: /register/i,
    });
    const registrationForm = container.querySelector("form");

    await act(async () => {
      fireEvent.change(usernameInput, { target: { value: "testuser" } });
      fireEvent.change(emailInput, { target: { value: "invalid-email" } });
      fireEvent.change(passwordInput, { target: { value: "password123" } });
    });

    expect(registerButton).not.toBeDisabled();
    await act(async () => {
      fireEvent.submit(registrationForm);
    });

    await waitFor(() =>
      expect(screen.getByText(/invalid email format/i)).toBeInTheDocument(),
    );
    expect(mockApi.register).not.toHaveBeenCalled();
  });
});
