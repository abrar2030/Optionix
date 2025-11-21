import axios from "axios";

// Define the base URL for the backend API
// TODO: Replace with the actual backend URL when deployed or if different locally
const API_BASE_URL = "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Market data services
export const marketService = {
  getMarketOverview: async () => {
    try {
      const response = await api.get("/market/overview");
      return response.data;
    } catch (error) {
      console.error("Error fetching market overview:", error);
      throw error;
    }
  },
  getPriceHistory: async (symbol, timeframe) => {
    try {
      const response = await api.get(`/market/price-history/${symbol}`, {
        params: { timeframe },
      });
      return response.data;
    } catch (error) {
      console.error("Error fetching price history:", error);
      throw error;
    }
  },
  getOptionChain: async (symbol, expiry) => {
    try {
      const response = await api.get(`/market/option-chain/${symbol}`, {
        params: { expiry },
      });
      return response.data;
    } catch (error) {
      console.error("Error fetching option chain:", error);
      throw error;
    }
  },
  getOrderBook: async (symbol) => {
    try {
      const response = await api.get(`/market/order-book/${symbol}`);
      return response.data;
    } catch (error) {
      console.error("Error fetching order book:", error);
      throw error;
    }
  },
  predictVolatility: async (data) => {
    try {
      const response = await api.post("/predict_volatility", data);
      return response.data;
    } catch (error) {
      console.error("Error predicting volatility:", error);
      throw error;
    }
  },
};

// Portfolio services
export const portfolioService = {
  getPortfolioSummary: async () => {
    try {
      const response = await api.get("/portfolio/summary");
      return response.data;
    } catch (error) {
      console.error("Error fetching portfolio summary:", error);
      throw error;
    }
  },
  getPositions: async () => {
    try {
      const response = await api.get("/portfolio/positions");
      return response.data;
    } catch (error) {
      console.error("Error fetching positions:", error);
      throw error;
    }
  },
  getPositionHealth: async (address) => {
    try {
      const response = await api.get(`/position_health/${address}`);
      return response.data;
    } catch (error) {
      console.error("Error fetching position health:", error);
      throw error;
    }
  },
  getTransactionHistory: async () => {
    try {
      const response = await api.get("/portfolio/transactions");
      return response.data;
    } catch (error) {
      console.error("Error fetching transaction history:", error);
      throw error;
    }
  },
  getPerformanceHistory: async (timeframe) => {
    try {
      const response = await api.get("/portfolio/performance", {
        params: { timeframe },
      });
      return response.data;
    } catch (error) {
      console.error("Error fetching performance history:", error);
      throw error;
    }
  },
};

// Trading services
export const tradingService = {
  executeTrade: async (tradeData) => {
    try {
      const response = await api.post("/trading/execute", tradeData);
      return response.data;
    } catch (error) {
      console.error("Error executing trade:", error);
      throw error;
    }
  },
  calculateOptionPrice: async (optionData) => {
    try {
      const response = await api.post("/trading/calculate-price", optionData);
      return response.data;
    } catch (error) {
      console.error("Error calculating option price:", error);
      throw error;
    }
  },
  calculateGreeks: async (optionData) => {
    try {
      const response = await api.post("/trading/calculate-greeks", optionData);
      return response.data;
    } catch (error) {
      console.error("Error calculating greeks:", error);
      throw error;
    }
  },
};

// Analytics services
export const analyticsService = {
  getRiskAssessment: async () => {
    try {
      const response = await api.get("/analytics/risk-assessment");
      return response.data;
    } catch (error) {
      console.error("Error fetching risk assessment:", error);
      throw error;
    }
  },
  getVolatilityAnalysis: async (symbol, timeframe) => {
    try {
      const response = await api.get("/analytics/volatility", {
        params: { symbol, timeframe },
      });
      return response.data;
    } catch (error) {
      console.error("Error fetching volatility analysis:", error);
      throw error;
    }
  },
  getMarketSentiment: async () => {
    try {
      const response = await api.get("/analytics/market-sentiment");
      return response.data;
    } catch (error) {
      console.error("Error fetching market sentiment:", error);
      throw error;
    }
  },
};

export default api;
