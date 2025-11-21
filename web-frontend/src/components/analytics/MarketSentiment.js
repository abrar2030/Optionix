import React from "react";
import styled from "styled-components";
import { Doughnut } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";

// Register ChartJS components
ChartJS.register(ArcElement, Tooltip, Legend);

const SentimentContainer = styled.div`
  width: 100%;
`;

const ChartContainer = styled.div`
  height: 200px;
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
`;

const CenterText = styled.div`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
`;

const SentimentValue = styled.div`
  font-size: 24px;
  font-weight: 700;
  color: ${(props) => props.theme.colors.textPrimary};
`;

const SentimentLabel = styled.div`
  font-size: 12px;
  color: ${(props) => props.theme.colors.textSecondary};
`;

const StatsContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-top: 20px;
`;

const StatItem = styled.div`
  background-color: ${(props) => props.theme.colors.backgroundDark};
  border-radius: 6px;
  padding: 12px;
  text-align: center;
`;

const StatValue = styled.div`
  font-size: 18px;
  font-weight: 600;
  color: ${(props) => props.theme.colors.textPrimary};
  margin-bottom: 4px;
`;

const StatLabel = styled.div`
  font-size: 12px;
  color: ${(props) => props.theme.colors.textSecondary};
`;

const MarketSentiment = () => {
  // Sample data
  const sentimentData = {
    labels: ["Bullish", "Neutral", "Bearish"],
    datasets: [
      {
        data: [65, 20, 15],
        backgroundColor: ["#26a69a", "#42a5f5", "#ef5350"],
        borderColor: "transparent",
        borderWidth: 0,
        hoverOffset: 4,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: "#1e222d",
        titleColor: "#b2b5be",
        bodyColor: "white",
        borderColor: "#2a2e39",
        borderWidth: 1,
        padding: 10,
        displayColors: false,
        callbacks: {
          label: function (context) {
            return `${context.label}: ${context.raw}%`;
          },
        },
      },
    },
    cutout: "70%",
  };

  return (
    <SentimentContainer>
      <ChartContainer>
        <Doughnut data={sentimentData} options={options} />
        <CenterText>
          <SentimentValue>65%</SentimentValue>
          <SentimentLabel>Bullish</SentimentLabel>
        </CenterText>
      </ChartContainer>

      <StatsContainer>
        <StatItem>
          <StatValue>0.85</StatValue>
          <StatLabel>Put/Call Ratio</StatLabel>
        </StatItem>
        <StatItem>
          <StatValue>18.32</StatValue>
          <StatLabel>VIX Index</StatLabel>
        </StatItem>
        <StatItem>
          <StatValue>62%</StatValue>
          <StatLabel>Long Positions</StatLabel>
        </StatItem>
      </StatsContainer>
    </SentimentContainer>
  );
};

export default MarketSentiment;
