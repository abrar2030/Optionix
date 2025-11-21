import React from "react";
import styled from "styled-components";
import {
  FiPieChart,
  FiAlertTriangle,
  FiTrendingUp,
  FiTrendingDown,
} from "react-icons/fi";

// Components
import PositionsList from "../components/portfolio/PositionsList";
import PortfolioAllocation from "../components/portfolio/PortfolioAllocation";
import PerformanceChart from "../components/portfolio/PerformanceChart";
import RiskAssessment from "../components/portfolio/RiskAssessment";

const PortfolioContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 20px;
`;

const SummarySection = styled.div`
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

const SummaryCard = styled.div`
  background-color: ${(props) => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${(props) => props.theme.colors.border};
  display: flex;
  flex-direction: column;
`;

const SummaryHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
`;

const SummaryTitle = styled.h3`
  font-size: 14px;
  font-weight: 500;
  color: ${(props) => props.theme.colors.textSecondary};
  margin: 0;
`;

const SummaryIcon = styled.div`
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

const SummaryValue = styled.div`
  font-size: 24px;
  font-weight: 700;
  color: ${(props) => props.theme.colors.textPrimary};
  margin-bottom: 4px;
`;

const SummaryChange = styled.div`
  font-size: 12px;
  color: ${(props) =>
    props.isPositive ? props.theme.colors.success : props.theme.colors.danger};
  display: flex;
  align-items: center;

  svg {
    margin-right: 4px;
  }
`;

const PositionsSection = styled.div`
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

const AllocationSection = styled.div`
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

const PerformanceSection = styled.div`
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

const RiskSection = styled.div`
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

const CardTitle = styled.h2`
  font-size: 18px;
  font-weight: 600;
  color: ${(props) => props.theme.colors.textPrimary};
  margin: 0 0 20px 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const Portfolio = () => {
  return (
    <PortfolioContainer>
      <SummarySection>
        <SummaryCard>
          <SummaryHeader>
            <SummaryTitle>Total Value</SummaryTitle>
            <SummaryIcon color="#2962ff">
              <FiPieChart />
            </SummaryIcon>
          </SummaryHeader>
          <SummaryValue>$24,875.65</SummaryValue>
          <SummaryChange isPositive={true}>
            <FiTrendingUp /> +5.27% today
          </SummaryChange>
        </SummaryCard>

        <SummaryCard>
          <SummaryHeader>
            <SummaryTitle>Profit/Loss</SummaryTitle>
            <SummaryIcon color="#26a69a">
              <FiTrendingUp />
            </SummaryIcon>
          </SummaryHeader>
          <SummaryValue>$1,243.89</SummaryValue>
          <SummaryChange isPositive={true}>
            <FiTrendingUp /> +12.3% this week
          </SummaryChange>
        </SummaryCard>

        <SummaryCard>
          <SummaryHeader>
            <SummaryTitle>Margin Used</SummaryTitle>
            <SummaryIcon color="#ff6d00">
              <FiTrendingDown />
            </SummaryIcon>
          </SummaryHeader>
          <SummaryValue>$5,120.00</SummaryValue>
          <SummaryChange isPositive={false}>
            <FiTrendingDown /> -2.5% available
          </SummaryChange>
        </SummaryCard>

        <SummaryCard>
          <SummaryHeader>
            <SummaryTitle>Risk Level</SummaryTitle>
            <SummaryIcon color="#ef5350">
              <FiAlertTriangle />
            </SummaryIcon>
          </SummaryHeader>
          <SummaryValue>Medium</SummaryValue>
          <SummaryChange isPositive={false}>
            <FiTrendingUp /> +2.1% since yesterday
          </SummaryChange>
        </SummaryCard>
      </SummarySection>

      <PositionsSection>
        <CardTitle>Open Positions</CardTitle>
        <PositionsList />
      </PositionsSection>

      <AllocationSection>
        <CardTitle>Portfolio Allocation</CardTitle>
        <PortfolioAllocation />
      </AllocationSection>

      <PerformanceSection>
        <CardTitle>Performance History</CardTitle>
        <PerformanceChart />
      </PerformanceSection>

      <RiskSection>
        <CardTitle>Risk Assessment</CardTitle>
        <RiskAssessment />
      </RiskSection>
    </PortfolioContainer>
  );
};

export default Portfolio;
