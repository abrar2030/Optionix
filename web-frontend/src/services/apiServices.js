import api from "../utils/api";

// Market data services
export const marketService = {
  // Get market overview data
  getMarketOverview: async () => {
    try {
      const response = await api.get("/market/overview");
      return response.data;
    } catch (error) {
      console.error("Error fetching market overview:", error);
      throw error;
    }
  },

  // Get price history for a specific asset
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

  // Get option chain data
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

  // Get order book data
  getOrderBook: async (symbol) => {
    try {
      const response = await api.get(`/market/order-book/${symbol}`);
      return response.data;
    } catch (error) {
      console.error("Error fetching order book:", error);
      throw error;
    }
  },

  // Predict volatility using AI model
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
  // Get portfolio summary
  getPortfolioSummary: async () => {
    try {
      const response = await api.get("/portfolio/summary");
      return response.data;
    } catch (error) {
      console.error("Error fetching portfolio summary:", error);
      throw error;
    }
  },

  // Get open positions
  getPositions: async () => {
    try {
      const response = await api.get("/portfolio/positions");
      return response.data;
    } catch (error) {
      console.error("Error fetching positions:", error);
      throw error;
    }
  },

  // Get position health
  getPositionHealth: async (address) => {
    try {
      const response = await api.get(`/position_health/${address}`);
      return response.data;
    } catch (error) {
      console.error("Error fetching position health:", error);
      throw error;
    }
  },

  // Get transaction history
  getTransactionHistory: async () => {
    try {
      const response = await api.get("/portfolio/transactions");
      return response.data;
    } catch (error) {
      console.error("Error fetching transaction history:", error);
      throw error;
    }
  },

  // Get performance history
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
  // Execute a trade
  executeTrade: async (tradeData) => {
    try {
      const response = await api.post("/trading/execute", tradeData);
      return response.data;
    } catch (error) {
      console.error("Error executing trade:", error);
      throw error;
    }
  },

  // Calculate option price
  calculateOptionPrice: async (optionData) => {
    try {
      const response = await api.post("/trading/calculate-price", optionData);
      return response.data;
    } catch (error) {
      console.error("Error calculating option price:", error);
      throw error;
    }
  },

  // Calculate option greeks
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
  // Get risk assessment
  getRiskAssessment: async () => {
    try {
      const response = await api.get("/analytics/risk-assessment");
      return response.data;
    } catch (error) {
      console.error("Error fetching risk assessment:", error);
      throw error;
    }
  },

  // Get volatility analysis
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

  // Get market sentiment
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
