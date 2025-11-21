import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// Mock the components directory structure
jest.mock("../components/common/Header", () => () => (
  <div data-testid="mock-header">Header</div>
));
jest.mock("../components/common/Footer", () => () => (
  <div data-testid="mock-footer">Footer</div>
));
jest.mock("../components/dashboard/PortfolioSummary", () => () => (
  <div data-testid="mock-portfolio">Portfolio Summary</div>
));
jest.mock("../components/dashboard/MarketOverview", () => () => (
  <div data-testid="mock-market">Market Overview</div>
));

// Mock the API service
jest.mock("../services/api", () => ({
  fetchDashboardData: jest.fn(),
  fetchUserProfile: jest.fn(),
}));

// Mock the blockchain service
jest.mock("../services/blockchain", () => ({
  getWalletStatus: jest.fn(),
}));

import api from "../services/api";
import blockchain from "../services/blockchain";
import Dashboard from "../pages/Dashboard";

describe("Dashboard Page", () => {
  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();

    // Setup default mock responses
    api.fetchDashboardData.mockResolvedValue({
      portfolioValue: 25000,
      pnl: 1250,
      pnlPercentage: 5.25,
      openPositions: 3,
      marginUsed: 12500,
    });

    api.fetchUserProfile.mockResolvedValue({
      username: "testuser",
      email: "test@example.com",
      joinDate: "2023-01-15",
    });

    blockchain.getWalletStatus.mockResolvedValue({
      connected: true,
      address: "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
      balance: "3.25",
    });
  });

  test("renders dashboard with user data", async () => {
    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>,
    );

    // Check that API calls were made
    expect(api.fetchDashboardData).toHaveBeenCalled();
    expect(api.fetchUserProfile).toHaveBeenCalled();
    expect(blockchain.getWalletStatus).toHaveBeenCalled();

    // Wait for user data to be displayed
    await waitFor(() => {
      expect(screen.getByText(/testuser/i)).toBeInTheDocument();
      expect(screen.getByText(/\$25,000/)).toBeInTheDocument();
      expect(screen.getByText(/\+\$1,250/)).toBeInTheDocument();
      expect(screen.getByText(/5\.25%/)).toBeInTheDocument();
    });

    // Check that all components are rendered
    expect(screen.getByTestId("mock-header")).toBeInTheDocument();
    expect(screen.getByTestId("mock-portfolio")).toBeInTheDocument();
    expect(screen.getByTestId("mock-market")).toBeInTheDocument();
    expect(screen.getByTestId("mock-footer")).toBeInTheDocument();
  });

  test("displays loading state while fetching data", () => {
    // Don't resolve the promises yet
    api.fetchDashboardData.mockReturnValue(new Promise(() => {}));
    api.fetchUserProfile.mockReturnValue(new Promise(() => {}));

    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>,
    );

    // Check for loading indicator
    expect(screen.getByText(/loading dashboard/i)).toBeInTheDocument();
  });

  test("handles error state when API calls fail", async () => {
    // Mock APIs to return errors
    api.fetchDashboardData.mockRejectedValue(
      new Error("Failed to fetch dashboard data"),
    );

    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>,
    );

    // Check for error message
    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
      expect(
        screen.getByText(/failed to fetch dashboard data/i),
      ).toBeInTheDocument();
    });
  });

  test("refreshes data when refresh button is clicked", async () => {
    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>,
    );

    // Wait for initial data to load
    await waitFor(() => {
      expect(screen.getByText(/\$25,000/)).toBeInTheDocument();
    });

    // Clear mock calls
    api.fetchDashboardData.mockClear();

    // Find and click the refresh button
    const refreshButton = screen.getByRole("button", { name: /refresh/i });
    fireEvent.click(refreshButton);

    // Check that API was called again
    expect(api.fetchDashboardData).toHaveBeenCalled();
  });

  test("navigates to trading page when trade button is clicked", async () => {
    const mockNavigate = jest.fn();
    jest.mock("react-router-dom", () => ({
      ...jest.requireActual("react-router-dom"),
      useNavigate: () => mockNavigate,
    }));

    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>,
    );

    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText(/\$25,000/)).toBeInTheDocument();
    });

    // Find and click the trade button
    const tradeButton = screen.getByRole("button", { name: /trade now/i });
    fireEvent.click(tradeButton);

    // Check that navigation was triggered
    expect(mockNavigate).toHaveBeenCalledWith("/trading");
  });
});
