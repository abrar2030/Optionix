import React, { createContext, useContext, useState, useEffect } from "react";
import { marketService, portfolioService } from "../services/apiServices";

// Create context
const AppContext = createContext();

// Provider component
export const AppProvider = ({ children }) => {
  // Global state
  const [marketData, setMarketData] = useState(null);
  const [portfolioData, setPortfolioData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState("dark");

  // Initialize data
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        setLoading(true);

        // In a real app, we would fetch this data from the API
        // For now, we'll use mock data since the backend isn't fully connected

        // Mock market data
        const marketOverview = {
          assets: [
            {
              id: 1,
              name: "BTC",
              fullName: "Bitcoin",
              price: "$42,567.89",
              change: "+2.34%",
              isPositive: true,
              volume: "$28.5B",
              ivRank: "45%",
              color: "#f7931a",
            },
            {
              id: 2,
              name: "ETH",
              fullName: "Ethereum",
              price: "$3,245.67",
              change: "+1.56%",
              isPositive: true,
              volume: "$15.2B",
              ivRank: "38%",
              color: "#627eea",
            },
            {
              id: 3,
              name: "SOL",
              fullName: "Solana",
              price: "$118.45",
              change: "-0.78%",
              isPositive: false,
              volume: "$4.8B",
              ivRank: "52%",
              color: "#00ffbd",
            },
            {
              id: 4,
              name: "AAPL",
              fullName: "Apple Inc.",
              price: "$178.32",
              change: "+0.45%",
              isPositive: true,
              volume: "$3.2B",
              ivRank: "32%",
              color: "#a2aaad",
            },
            {
              id: 5,
              name: "TSLA",
              fullName: "Tesla Inc.",
              price: "$215.67",
              change: "-1.23%",
              isPositive: false,
              volume: "$5.1B",
              ivRank: "65%",
              color: "#cc0000",
            },
          ],
        };

        // Mock portfolio data
        const portfolioSummary = {
          totalValue: "$24,875.65",
          totalChange: "+5.27%",
          isPositive: true,
          positions: 12,
          positionsChange: "+2",
          profitLoss: "$1,243.89",
          profitLossChange: "+12.3%",
          riskLevel: "Medium",
          riskChange: "+2.1%",
          allocation: [
            { label: "BTC Options", value: "35%", color: "#2962ff" },
            { label: "ETH Options", value: "25%", color: "#26a69a" },
            { label: "SOL Options", value: "15%", color: "#ff6d00" },
            { label: "AAPL Options", value: "15%", color: "#42a5f5" },
            { label: "TSLA Options", value: "10%", color: "#ec407a" },
          ],
          transactions: [
            {
              id: 1,
              type: "buy",
              title: "BTC-USD Call Option",
              details: "Strike: $45,000 | Exp: Apr 30, 2025",
              amount: "+2 Contracts",
              timestamp: "2 hours ago",
            },
            {
              id: 2,
              type: "sell",
              title: "ETH-USD Put Option",
              details: "Strike: $3,200 | Exp: May 15, 2025",
              amount: "-5 Contracts",
              timestamp: "5 hours ago",
            },
            {
              id: 3,
              type: "buy",
              title: "AAPL Call Option",
              details: "Strike: $180 | Exp: Jun 20, 2025",
              amount: "+10 Contracts",
              timestamp: "1 day ago",
            },
            {
              id: 4,
              type: "sell",
              title: "TSLA Put Option",
              details: "Strike: $210 | Exp: May 5, 2025",
              amount: "-3 Contracts",
              timestamp: "2 days ago",
            },
            {
              id: 5,
              type: "buy",
              title: "SOL-USD Call Option",
              details: "Strike: $120 | Exp: Apr 25, 2025",
              amount: "+8 Contracts",
              timestamp: "3 days ago",
            },
          ],
        };

        setMarketData(marketOverview);
        setPortfolioData(portfolioSummary);
        setLoading(false);
      } catch (err) {
        console.error("Error fetching initial data:", err);
        setError("Failed to load application data. Please try again later.");
        setLoading(false);
      }
    };

    fetchInitialData();
  }, []);

  // Toggle theme function
  const toggleTheme = () => {
    setTheme((prevTheme) => (prevTheme === "dark" ? "light" : "dark"));
  };

  // Context value
  const value = {
    marketData,
    portfolioData,
    loading,
    error,
    theme,
    toggleTheme,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

// Custom hook to use the context
export const useAppContext = () => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error("useAppContext must be used within an AppProvider");
  }
  return context;
};
