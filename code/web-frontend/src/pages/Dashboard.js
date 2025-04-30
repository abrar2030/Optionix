import React from 'react';
import styled from 'styled-components';
import { FiTrendingUp, FiDollarSign, FiActivity, FiPieChart } from 'react-icons/fi';

// Components
import PriceChart from '../components/dashboard/PriceChart';
import PortfolioSummary from '../components/dashboard/PortfolioSummary';
import RecentTransactions from '../components/dashboard/RecentTransactions';
import MarketOverview from '../components/dashboard/MarketOverview';

const DashboardContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 20px;
`;

const StatCard = styled.div`
  grid-column: span 3;
  background-color: ${props => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${props => props.theme.colors.border};
  display: flex;
  flex-direction: column;
  
  @media (max-width: ${props => props.theme.breakpoints.desktop}) {
    grid-column: span 6;
  }
  
  @media (max-width: ${props => props.theme.breakpoints.mobile}) {
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
  color: ${props => props.theme.colors.textSecondary};
  margin: 0;
`;

const StatIcon = styled.div`
  width: 36px;
  height: 36px;
  border-radius: 8px;
  background-color: ${props => props.color || props.theme.colors.primary};
  opacity: 0.1;
  display: flex;
  align-items: center;
  justify-content: center;
  
  svg {
    color: ${props => props.color || props.theme.colors.primary};
    font-size: 20px;
    opacity: 10;
  }
`;

const StatValue = styled.div`
  font-size: 24px;
  font-weight: 700;
  color: ${props => props.theme.colors.textPrimary};
  margin-bottom: 4px;
`;

const StatChange = styled.div`
  font-size: 12px;
  color: ${props => props.isPositive ? props.theme.colors.success : props.theme.colors.danger};
  display: flex;
  align-items: center;
  
  svg {
    margin-right: 4px;
  }
`;

const ChartCard = styled.div`
  grid-column: span 8;
  background-color: ${props => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${props => props.theme.colors.border};
  
  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    grid-column: span 12;
  }
`;

const PortfolioCard = styled.div`
  grid-column: span 4;
  background-color: ${props => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${props => props.theme.colors.border};
  
  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    grid-column: span 12;
  }
`;

const TransactionsCard = styled.div`
  grid-column: span 6;
  background-color: ${props => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${props => props.theme.colors.border};
  
  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    grid-column: span 12;
  }
`;

const MarketCard = styled.div`
  grid-column: span 6;
  background-color: ${props => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${props => props.theme.colors.border};
  
  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    grid-column: span 12;
  }
`;

const CardTitle = styled.h2`
  font-size: 18px;
  font-weight: 600;
  color: ${props => props.theme.colors.textPrimary};
  margin: 0 0 20px 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const Dashboard = () => {
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
