import React from "react";
import styled from "styled-components";
import {
  FiBarChart2,
  FiActivity,
  FiAlertCircle,
  FiTrendingUp,
} from "react-icons/fi";

// Components
import VolatilityChart from "../components/analytics/VolatilityChart";
import GreeksTable from "../components/analytics/GreeksTable";
import RiskMatrix from "../components/analytics/RiskMatrix";
import MarketSentiment from "../components/analytics/MarketSentiment";

const AnalyticsContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 20px;
`;

const MetricsSection = styled.div`
  grid-column: span 12;
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;

  @media (max-width: ${(props) => props.theme.breakpoints.tablet}) {
    grid-template-columns: repeat(2, 1fr);
  }

  @media (max-width: ${(props) => props.theme.breakpoints.mobile}) {
    grid-template-columns: 1fr;
  }
`;

const MetricCard = styled.div`
  background-color: ${(props) => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${(props) => props.theme.colors.border};
  display: flex;
  flex-direction: column;
`;

const MetricHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
`;

const MetricTitle = styled.h3`
  font-size: 14px;
  font-weight: 500;
  color: ${(props) => props.theme.colors.textSecondary};
  margin: 0;
`;

const MetricIcon = styled.div`
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

const MetricValue = styled.div`
  font-size: 24px;
  font-weight: 700;
  color: ${(props) => props.theme.colors.textPrimary};
  margin-bottom: 4px;
`;

const MetricDescription = styled.div`
  font-size: 12px;
  color: ${(props) => props.theme.colors.textSecondary};
`;

const VolatilitySection = styled.div`
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

const GreeksSection = styled.div`
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

const RiskSection = styled.div`
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

const SentimentSection = styled.div`
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

const Analytics = () => {
  return (
    <AnalyticsContainer>
      <MetricsSection>
        <MetricCard>
          <MetricHeader>
            <MetricTitle>Implied Volatility</MetricTitle>
            <MetricIcon color="#2962ff">
              <FiBarChart2 />
            </MetricIcon>
          </MetricHeader>
          <MetricValue>32.45%</MetricValue>
          <MetricDescription>30-day average for BTC-USD</MetricDescription>
        </MetricCard>

        <MetricCard>
          <MetricHeader>
            <MetricTitle>Put/Call Ratio</MetricTitle>
            <MetricIcon color="#26a69a">
              <FiActivity />
            </MetricIcon>
          </MetricHeader>
          <MetricValue>0.85</MetricValue>
          <MetricDescription>
            Slightly bullish market sentiment
          </MetricDescription>
        </MetricCard>

        <MetricCard>
          <MetricHeader>
            <MetricTitle>VIX Index</MetricTitle>
            <MetricIcon color="#ff6d00">
              <FiAlertCircle />
            </MetricIcon>
          </MetricHeader>
          <MetricValue>18.32</MetricValue>
          <MetricDescription>Market volatility index</MetricDescription>
        </MetricCard>

        <MetricCard>
          <MetricHeader>
            <MetricTitle>Options Volume</MetricTitle>
            <MetricIcon color="#ef5350">
              <FiTrendingUp />
            </MetricIcon>
          </MetricHeader>
          <MetricValue>$1.2B</MetricValue>
          <MetricDescription>24h trading volume</MetricDescription>
        </MetricCard>
      </MetricsSection>

      <VolatilitySection>
        <CardTitle>Historical Volatility Analysis</CardTitle>
        <VolatilityChart />
      </VolatilitySection>

      <GreeksSection>
        <CardTitle>Option Greeks</CardTitle>
        <GreeksTable />
      </GreeksSection>

      <RiskSection>
        <CardTitle>Risk Matrix</CardTitle>
        <RiskMatrix />
      </RiskSection>

      <SentimentSection>
        <CardTitle>Market Sentiment</CardTitle>
        <MarketSentiment />
      </SentimentSection>
    </AnalyticsContainer>
  );
};

export default Analytics;
