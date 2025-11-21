import React from "react";
import styled from "styled-components";
import { FiAlertTriangle, FiCheckCircle, FiInfo } from "react-icons/fi";

const RiskContainer = styled.div`
  width: 100%;
`;

const RiskCard = styled.div`
  background-color: ${(props) => props.theme.colors.backgroundDark};
  border-radius: 6px;
  padding: 16px;
  margin-bottom: 16px;
  border-left: 4px solid
    ${(props) => {
      if (props.level === "high") return props.theme.colors.danger;
      if (props.level === "medium") return props.theme.colors.warning;
      return props.theme.colors.success;
    }};
`;

const RiskHeader = styled.div`
  display: flex;
  align-items: center;
  margin-bottom: 10px;
`;

const RiskIcon = styled.div`
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background-color: ${(props) => {
    if (props.level === "high") return "rgba(239, 83, 80, 0.1)";
    if (props.level === "medium") return "rgba(255, 202, 40, 0.1)";
    return "rgba(38, 166, 154, 0.1)";
  }};
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 12px;

  svg {
    color: ${(props) => {
      if (props.level === "high") return props.theme.colors.danger;
      if (props.level === "medium") return props.theme.colors.warning;
      return props.theme.colors.success;
    }};
    font-size: 18px;
  }
`;

const RiskTitle = styled.div`
  font-size: 16px;
  font-weight: 600;
  color: ${(props) => props.theme.colors.textPrimary};
`;

const RiskDescription = styled.div`
  font-size: 13px;
  color: ${(props) => props.theme.colors.textSecondary};
  line-height: 1.5;
  padding-left: 44px;
`;

const RiskMetrics = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
  margin-top: 20px;
`;

const MetricCard = styled.div`
  background-color: ${(props) => props.theme.colors.backgroundDark};
  border-radius: 6px;
  padding: 12px;
  text-align: center;
`;

const MetricValue = styled.div`
  font-size: 18px;
  font-weight: 600;
  color: ${(props) => props.theme.colors.textPrimary};
  margin-bottom: 4px;
`;

const MetricLabel = styled.div`
  font-size: 12px;
  color: ${(props) => props.theme.colors.textSecondary};
`;

const RiskAssessment = () => {
  // Sample data
  const riskItems = [
    {
      id: 1,
      level: "high",
      title: "High Volatility Exposure",
      description:
        "Your portfolio has significant exposure to high volatility assets, which could lead to larger than expected price swings.",
      icon: <FiAlertTriangle />,
    },
    {
      id: 2,
      level: "medium",
      title: "Concentration Risk",
      description:
        "Over 60% of your portfolio is concentrated in BTC and ETH options, consider diversifying to reduce risk.",
      icon: <FiInfo />,
    },
    {
      id: 3,
      level: "low",
      title: "Positive Delta Exposure",
      description:
        "Your portfolio has a net positive delta, which aligns well with the current bullish market sentiment.",
      icon: <FiCheckCircle />,
    },
  ];

  return (
    <RiskContainer>
      {riskItems.map((item) => (
        <RiskCard key={item.id} level={item.level}>
          <RiskHeader>
            <RiskIcon level={item.level}>{item.icon}</RiskIcon>
            <RiskTitle>{item.title}</RiskTitle>
          </RiskHeader>
          <RiskDescription>{item.description}</RiskDescription>
        </RiskCard>
      ))}

      <RiskMetrics>
        <MetricCard>
          <MetricValue>0.75</MetricValue>
          <MetricLabel>Portfolio Beta</MetricLabel>
        </MetricCard>
        <MetricCard>
          <MetricValue>32.5%</MetricValue>
          <MetricLabel>Value at Risk (VaR)</MetricLabel>
        </MetricCard>
        <MetricCard>
          <MetricValue>1.25</MetricValue>
          <MetricLabel>Sharpe Ratio</MetricLabel>
        </MetricCard>
        <MetricCard>
          <MetricValue>0.85</MetricValue>
          <MetricLabel>Sortino Ratio</MetricLabel>
        </MetricCard>
      </RiskMetrics>
    </RiskContainer>
  );
};

export default RiskAssessment;
