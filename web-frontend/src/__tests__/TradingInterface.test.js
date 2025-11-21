import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import TradingInterface from "../components/TradingInterface";

// Mock the API service
jest.mock("../services/api", () => ({
  fetchMarketData: jest.fn(),
  submitTrade: jest.fn(),
}));

// Mock the blockchain service
jest.mock("../services/blockchain", () => ({
  connectWallet: jest.fn(),
  getAccountBalance: jest.fn(),
  executeTransaction: jest.fn(),
}));

import api from "../services/api";
import blockchain from "../services/blockchain";

describe("TradingInterface Component", () => {
  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();

    // Setup default mock responses
    api.fetchMarketData.mockResolvedValue({
      price: 1500.25,
      change: 2.5,
      volume: 1250000,
      options: [
        { strike: 1600, premium: 45.2, type: "call" },
        { strike: 1400, premium: 38.75, type: "put" },
      ],
    });

    api.submitTrade.mockResolvedValue({ success: true, txId: "0x123456789" });

    blockchain.connectWallet.mockResolvedValue({
      address: "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
    });
    blockchain.getAccountBalance.mockResolvedValue("2.5");
    blockchain.executeTransaction.mockResolvedValue({ hash: "0x987654321" });
  });

  test("renders trading interface with market data", async () => {
    render(
      <MemoryRouter>
        <TradingInterface />
      </MemoryRouter>,
    );

    // Check that market data is fetched and displayed
    expect(api.fetchMarketData).toHaveBeenCalled();

    // Wait for market data to be displayed
    await waitFor(() => {
      expect(screen.getByText(/1500\.25/)).toBeInTheDocument();
      expect(screen.getByText(/2\.5%/)).toBeInTheDocument();
    });

    // Check that options are displayed
    await waitFor(() => {
      expect(screen.getByText(/1600/)).toBeInTheDocument();
      expect(screen.getByText(/1400/)).toBeInTheDocument();
    });
  });

  test("connects wallet when connect button is clicked", async () => {
    render(
      <MemoryRouter>
        <TradingInterface />
      </MemoryRouter>,
    );

    // Find and click the connect wallet button
    const connectButton = screen.getByRole("button", {
      name: /connect wallet/i,
    });
    fireEvent.click(connectButton);

    // Check that the wallet connection function was called
    expect(blockchain.connectWallet).toHaveBeenCalled();

    // Wait for wallet address to be displayed
    await waitFor(() => {
      expect(screen.getByText(/0x742d.*44e/i)).toBeInTheDocument();
      expect(screen.getByText(/2\.5 ETH/i)).toBeInTheDocument();
    });
  });

  test("submits a trade when form is filled and submitted", async () => {
    render(
      <MemoryRouter>
        <TradingInterface />
      </MemoryRouter>,
    );

    // Connect wallet first
    const connectButton = screen.getByRole("button", {
      name: /connect wallet/i,
    });
    fireEvent.click(connectButton);
    await waitFor(() =>
      expect(screen.getByText(/0x742d.*44e/i)).toBeInTheDocument(),
    );

    // Fill out the trade form
    const amountInput = screen.getByLabelText(/amount/i);
    const strikeInput = screen.getByLabelText(/strike price/i);
    const typeSelect = screen.getByLabelText(/option type/i);

    fireEvent.change(amountInput, { target: { value: "1" } });
    fireEvent.change(strikeInput, { target: { value: "1600" } });
    fireEvent.change(typeSelect, { target: { value: "call" } });

    // Submit the form
    const submitButton = screen.getByRole("button", { name: /submit trade/i });
    fireEvent.click(submitButton);

    // Check that the API and blockchain functions were called
    await waitFor(() => {
      expect(api.submitTrade).toHaveBeenCalledWith({
        amount: "1",
        strike: "1600",
        type: "call",
        address: "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
      });
      expect(blockchain.executeTransaction).toHaveBeenCalled();
    });

    // Check for success message
    await waitFor(() => {
      expect(screen.getByText(/trade successful/i)).toBeInTheDocument();
      expect(screen.getByText(/0x987654321/i)).toBeInTheDocument();
    });
  });

  test("displays error when trade submission fails", async () => {
    // Mock API to return an error
    api.submitTrade.mockRejectedValue(new Error("Insufficient funds"));

    render(
      <MemoryRouter>
        <TradingInterface />
      </MemoryRouter>,
    );

    // Connect wallet first
    const connectButton = screen.getByRole("button", {
      name: /connect wallet/i,
    });
    fireEvent.click(connectButton);
    await waitFor(() =>
      expect(screen.getByText(/0x742d.*44e/i)).toBeInTheDocument(),
    );

    // Fill out the trade form
    const amountInput = screen.getByLabelText(/amount/i);
    const strikeInput = screen.getByLabelText(/strike price/i);
    const typeSelect = screen.getByLabelText(/option type/i);

    fireEvent.change(amountInput, { target: { value: "1" } });
    fireEvent.change(strikeInput, { target: { value: "1600" } });
    fireEvent.change(typeSelect, { target: { value: "call" } });

    // Submit the form
    const submitButton = screen.getByRole("button", { name: /submit trade/i });
    fireEvent.click(submitButton);

    // Check for error message
    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
      expect(screen.getByText(/insufficient funds/i)).toBeInTheDocument();
    });
  });
});
