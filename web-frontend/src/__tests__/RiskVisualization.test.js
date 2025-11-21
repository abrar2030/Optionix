import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import RiskVisualization from "../components/RiskVisualization";

// Mock the chart.js library
jest.mock("react-chartjs-2", () => ({
  Line: () => <div data-testid="mock-line-chart">Mock Chart</div>,
}));

// Mock the API service
jest.mock("../services/api", () => ({
  fetchRiskMetrics: jest.fn(),
}));

import api from "../services/api";

describe("RiskVisualization Component", () => {
  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();

    // Setup default mock responses
    api.fetchRiskMetrics.mockResolvedValue({
      volatility: 0.25,
      delta: 0.65,
      gamma: 0.12,
      theta: -0.08,
      vega: 0.35,
      rho: 0.03,
      timePoints: [1, 2, 3, 4, 5],
      pricePoints: [100, 105, 110, 115, 120],
      deltaValues: [0.6, 0.62, 0.65, 0.68, 0.7],
    });
  });

  test("renders risk visualization component", async () => {
    render(
      <MemoryRouter>
        <RiskVisualization positionId="123" />
      </MemoryRouter>,
    );

    // Check that risk metrics are fetched
    expect(api.fetchRiskMetrics).toHaveBeenCalledWith("123");

    // Wait for risk metrics to be displayed
    await waitFor(() => {
      expect(screen.getByText(/volatility/i)).toBeInTheDocument();
      expect(screen.getByText(/25%/)).toBeInTheDocument();
      expect(screen.getByText(/delta/i)).toBeInTheDocument();
      expect(screen.getByText(/0\.65/)).toBeInTheDocument();
    });

    // Check that chart is rendered
    expect(screen.getByTestId("mock-line-chart")).toBeInTheDocument();
  });

  test("displays loading state while fetching data", () => {
    // Don't resolve the promise yet
    api.fetchRiskMetrics.mockReturnValue(new Promise(() => {}));

    render(
      <MemoryRouter>
        <RiskVisualization positionId="123" />
      </MemoryRouter>,
    );

    // Check for loading indicator
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  test("handles error state when API call fails", async () => {
    // Mock API to return an error
    api.fetchRiskMetrics.mockRejectedValue(
      new Error("Failed to fetch risk metrics"),
    );

    render(
      <MemoryRouter>
        <RiskVisualization positionId="123" />
      </MemoryRouter>,
    );

    // Check for error message
    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
      expect(
        screen.getByText(/failed to fetch risk metrics/i),
      ).toBeInTheDocument();
    });
  });

  test("updates visualization when timeframe is changed", async () => {
    render(
      <MemoryRouter>
        <RiskVisualization positionId="123" />
      </MemoryRouter>,
    );

    // Wait for initial data to load
    await waitFor(() => {
      expect(screen.getByText(/volatility/i)).toBeInTheDocument();
    });

    // Find and click the timeframe selector
    const timeframeSelect = screen.getByLabelText(/timeframe/i);
    fireEvent.change(timeframeSelect, { target: { value: "monthly" } });

    // Check that API was called with new timeframe
    expect(api.fetchRiskMetrics).toHaveBeenCalledWith("123", "monthly");
  });

  test("toggles between different risk metrics", async () => {
    render(
      <MemoryRouter>
        <RiskVisualization positionId="123" />
      </MemoryRouter>,
    );

    // Wait for initial data to load
    await waitFor(() => {
      expect(screen.getByText(/volatility/i)).toBeInTheDocument();
    });

    // Find and click the delta tab
    const deltaTab = screen.getByRole("tab", { name: /delta/i });
    fireEvent.click(deltaTab);

    // Check that delta chart is now active
    expect(screen.getByText(/delta exposure/i)).toBeInTheDocument();

    // Find and click the theta tab
    const thetaTab = screen.getByRole("tab", { name: /theta/i });
    fireEvent.click(thetaTab);

    // Check that theta chart is now active
    expect(screen.getByText(/time decay/i)).toBeInTheDocument();
  });
});
