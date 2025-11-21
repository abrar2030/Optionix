import React, { useState, useEffect } from "react";
import styled from "styled-components";
import {
  FiTrendingUp,
  FiDollarSign,
  FiActivity,
  FiPieChart,
} from "react-icons/fi";
import { useAppContext } from "../utils/AppContext"; // Import useAppContext
import api from "../utils/api"; // Import api

// Components
import PriceChart from "../components/dashboard/PriceChart";
import PortfolioSummary from "../components/dashboard/PortfolioSummary";
import RecentTransactions from "../components/dashboard/RecentTransactions";
import MarketOverview from "../components/dashboard/MarketOverview";

// Styled components for Login/Register form (can be moved to a separate file)
const AuthContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 80vh; // Adjust as needed
  padding: 20px;
`;

const AuthForm = styled.form`
  background-color: ${(props) => props.theme.colors.cardBg};
  padding: 30px;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  width: 100%;
  max-width: 400px;
  border: 1px solid ${(props) => props.theme.colors.border};
`;

const AuthTitle = styled.h2`
  color: ${(props) => props.theme.colors.textPrimary};
  text-align: center;
  margin-bottom: 20px;
`;

const InputGroup = styled.div`
  margin-bottom: 15px;
`;

const Label = styled.label`
  display: block;
  color: ${(props) => props.theme.colors.textSecondary};
  margin-bottom: 5px;
  font-size: 14px;
`;

const Input = styled.input`
  width: 100%;
  padding: 10px;
  border-radius: 4px;
  border: 1px solid ${(props) => props.theme.colors.border};
  background-color: ${(props) => props.theme.colors.backgroundLight};
  color: ${(props) => props.theme.colors.textPrimary};
  font-size: 16px;
  box-sizing: border-box;
`;

const Button = styled.button`
  width: 100%;
  padding: 12px;
  background-color: ${(props) => props.theme.colors.primary};
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 16px;
  cursor: pointer;
  transition: background-color 0.3s ease;
  margin-top: 10px;

  &:hover {
    background-color: ${(props) => props.theme.colors.primaryDark};
  }

  &:disabled {
    background-color: #555;
    cursor: not-allowed;
  }
`;

const ToggleAuthLink = styled.p`
  color: ${(props) => props.theme.colors.textSecondary};
  text-align: center;
  margin-top: 15px;
  font-size: 14px;

  span {
    color: ${(props) => props.theme.colors.primaryLight};
    cursor: pointer;
    text-decoration: underline;
  }
`;

const ErrorMessage = styled.p`
  color: ${(props) => props.theme.colors.danger};
  font-size: 14px;
  text-align: center;
  margin-bottom: 10px;
`;

// Dashboard specific styled components (from original file)
const DashboardContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 20px;
`;

const StatCard = styled.div`
  grid-column: span 3;
  background-color: ${(props) => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${(props) => props.theme.colors.border};
  display: flex;
  flex-direction: column;

  @media (max-width: ${(props) => props.theme.breakpoints.desktop}) {
    grid-column: span 6;
  }

  @media (max-width: ${(props) => props.theme.breakpoints.mobile}) {
    grid-column: span 12;
  }
`;

const StatHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
`;

const StatTitle = styled.h3`
  font-size: 14px;
  font-weight: 500;
  color: ${(props) => props.theme.colors.textSecondary};
  margin: 0;
`;

const StatIcon = styled.div`
  width: 36px;
  height: 36px;
  border-radius: 8px;
  background-color: ${(props) => props.color || props.theme.colors.primary};
  opacity: 0.1;
  display: flex;
  align-items: center;
  justify-content: center;

  svg {
    color: ${(props) => props.color || props.theme.colors.primary};
    font-size: 20px;
    opacity: 10;
  }
`;

const StatValue = styled.div`
  font-size: 24px;
  font-weight: 700;
  color: ${(props) => props.theme.colors.textPrimary};
  margin-bottom: 4px;
`;

const StatChange = styled.div`
  font-size: 12px;
  color: ${(props) =>
    props.isPositive ? props.theme.colors.success : props.theme.colors.danger};
  display: flex;
  align-items: center;

  svg {
    margin-right: 4px;
  }
`;

const ChartCard = styled.div`
  grid-column: span 8;
  background-color: ${(props) => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${(props) => props.theme.colors.border};

  @media (max-width: ${(props) => props.theme.breakpoints.tablet}) {
    grid-column: span 12;
  }
`;

const PortfolioCard = styled.div`
  grid-column: span 4;
  background-color: ${(props) => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${(props) => props.theme.colors.border};

  @media (max-width: ${(props) => props.theme.breakpoints.tablet}) {
    grid-column: span 12;
  }
`;

const TransactionsCard = styled.div`
  grid-column: span 6;
  background-color: ${(props) => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${(props) => props.theme.colors.border};

  @media (max-width: ${(props) => props.theme.breakpoints.tablet}) {
    grid-column: span 12;
  }
`;

const MarketCard = styled.div`
  grid-column: span 6;
  background-color: ${(props) => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${(props) => props.theme.colors.border};

  @media (max-width: ${(props) => props.theme.breakpoints.tablet}) {
    grid-column: span 12;
  }
`;

const CardTitle = styled.h2`
  font-size: 18px;
  font-weight: 600;
  color: ${(props) => props.theme.colors.textPrimary};
  margin: 0 0 20px 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const Dashboard = () => {
  const { user, setUser, loading, setLoading } = useAppContext();
  const [isLoginView, setIsLoginView] = useState(true);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState(""); // For registration
  const [error, setError] = useState("");

  const handleLogin = async (e) => {
    console.log("[DEBUG] handleLogin triggered"); // DEBUG LOG
    e.preventDefault();
    setError("");
    if (!username || !password) {
      console.log("[DEBUG] handleLogin: Username or Password missing"); // DEBUG LOG
      setError("Username and Password are required");
      return;
    }
    setLoading(true);
    try {
      console.log(
        `[DEBUG] handleLogin: Attempting API login with username: ${username}`,
      ); // DEBUG LOG
      const response = await api.login(username, password);
      console.log("[DEBUG] handleLogin: API login successful, token received"); // DEBUG LOG
      localStorage.setItem("token", response.token);
      const profile = await api.getUserProfile();
      console.log("[DEBUG] handleLogin: User profile fetched"); // DEBUG LOG
      setUser(profile);
    } catch (err) {
      console.error("[DEBUG] handleLogin: API login error:", err); // DEBUG LOG
      setError(err.message || "Login failed");
    }
    setLoading(false);
  };

  const handleRegister = async (e) => {
    console.log("[DEBUG] handleRegister triggered"); // DEBUG LOG
    e.preventDefault();
    setError("");
    if (!username || !password || !email) {
      console.log(
        "[DEBUG] handleRegister: Username, Email or Password missing",
      ); // DEBUG LOG
      setError("Username, Email, and Password are required");
      return;
    }
    // Basic email validation
    if (!/\S+@\S+\.\S+/.test(email)) {
      console.log("[DEBUG] handleRegister: Invalid email format"); // DEBUG LOG
      setError("Invalid email format");
      return;
    }
    setLoading(true);
    try {
      console.log(
        `[DEBUG] handleRegister: Attempting API register with username: ${username}, email: ${email}`,
      ); // DEBUG LOG
      await api.register({ username, email, password });
      console.log("[DEBUG] handleRegister: API registration successful"); // DEBUG LOG
      // Automatically switch to login view after successful registration
      setIsLoginView(true);
      setError("Registration successful! Please login.");
    } catch (err) {
      console.error("[DEBUG] handleRegister: API registration error:", err); // DEBUG LOG
      setError(err.message || "Registration failed");
    }
    setLoading(false);
  };

  if (!user) {
    return (
      <AuthContainer>
        <AuthForm onSubmit={isLoginView ? handleLogin : handleRegister}>
          <AuthTitle>{isLoginView ? "Login" : "Create Account"}</AuthTitle>
          {error && <ErrorMessage>{error}</ErrorMessage>}
          <InputGroup>
            <Label htmlFor="username">Username</Label>
            <Input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </InputGroup>
          {!isLoginView && (
            <InputGroup>
              <Label htmlFor="email">Email</Label>
              <Input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </InputGroup>
          )}
          <InputGroup>
            <Label htmlFor="password">Password</Label>
            <Input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </InputGroup>
          <Button type="submit" disabled={loading}>
            {loading ? "Processing..." : isLoginView ? "Login" : "Register"}
          </Button>
          <ToggleAuthLink
            onClick={() => {
              setIsLoginView(!isLoginView);
              setError("");
            }}
          >
            {isLoginView
              ? "Don't have an account? "
              : "Already have an account? "}
            <span>{isLoginView ? "Register" : "Login"}</span>
          </ToggleAuthLink>
        </AuthForm>
      </AuthContainer>
    );
  }

  // Original Dashboard content for authenticated users
  return (
    <DashboardContainer>
      <StatCard>
        <StatHeader>
          <StatTitle>Total Portfolio Value</StatTitle>
          <StatIcon color="#2962ff">
            <FiDollarSign />
          </StatIcon>
        </StatHeader>
        <StatValue>$24,875.65</StatValue>
        <StatChange isPositive={true}>+5.27% today</StatChange>
      </StatCard>

      <StatCard>
        <StatHeader>
          <StatTitle>Open Positions</StatTitle>
          <StatIcon color="#26a69a">
            <FiActivity />
          </StatIcon>
        </StatHeader>
        <StatValue>12</StatValue>
        <StatChange isPositive={true}>+2 new today</StatChange>
      </StatCard>

      <StatCard>
        <StatHeader>
          <StatTitle>Profit/Loss</StatTitle>
          <StatIcon color="#ff6d00">
            <FiTrendingUp />
          </StatIcon>
        </StatHeader>
        <StatValue>$1,243.89</StatValue>
        <StatChange isPositive={true}>+12.3% this week</StatChange>
      </StatCard>

      <StatCard>
        <StatHeader>
          <StatTitle>Portfolio Risk</StatTitle>
          <StatIcon color="#ef5350">
            <FiPieChart />
          </StatIcon>
        </StatHeader>
        <StatValue>Medium</StatValue>
        <StatChange isPositive={false}>+2.1% since yesterday</StatChange>
      </StatCard>

      <ChartCard>
        <CardTitle>Price Chart</CardTitle>
        <PriceChart />
      </ChartCard>

      <PortfolioCard>
        <CardTitle>Portfolio Allocation</CardTitle>
        <PortfolioSummary />
      </PortfolioCard>

      <TransactionsCard>
        <CardTitle>Recent Transactions</CardTitle>
        <RecentTransactions />
      </TransactionsCard>

      <MarketCard>
        <CardTitle>Market Overview</CardTitle>
        <MarketOverview />
      </MarketCard>
    </DashboardContainer>
  );
};

export default Dashboard;
