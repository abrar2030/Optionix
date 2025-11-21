import React from "react";
import styled from "styled-components";
import { Doughnut } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";

// Register ChartJS components
ChartJS.register(ArcElement, Tooltip, Legend);

const ChartContainer = styled.div`
  height: 250px;
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
`;

const LegendContainer = styled.div`
  margin-top: 20px;
`;

const LegendItem = styled.div`
  display: flex;
  align-items: center;
  margin-bottom: 8px;
`;

const LegendColor = styled.div`
  width: 12px;
  height: 12px;
  border-radius: 2px;
  background-color: ${(props) => props.color};
  margin-right: 8px;
`;

const LegendLabel = styled.span`
  font-size: 12px;
  color: ${(props) => props.theme.colors.textSecondary};
  flex: 1;
`;

const LegendValue = styled.span`
  font-size: 12px;
  font-weight: 600;
  color: ${(props) => props.theme.colors.textPrimary};
`;

const PortfolioAllocation = () => {
  // Sample data
  const data = {
    labels: [
      "BTC Options",
      "ETH Options",
      "SOL Options",
      "AAPL Options",
      "TSLA Options",
    ],
    datasets: [
      {
        data: [35, 25, 15, 15, 10],
        backgroundColor: [
          "#2962ff",
          "#26a69a",
          "#ff6d00",
          "#42a5f5",
          "#ec407a",
        ],
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

  const legendItems = [
    { label: "BTC Options", value: "35%", color: "#2962ff" },
    { label: "ETH Options", value: "25%", color: "#26a69a" },
    { label: "SOL Options", value: "15%", color: "#ff6d00" },
    { label: "AAPL Options", value: "15%", color: "#42a5f5" },
    { label: "TSLA Options", value: "10%", color: "#ec407a" },
  ];

  return (
    <>
      <ChartContainer>
        <Doughnut data={data} options={options} />
      </ChartContainer>

      <LegendContainer>
        {legendItems.map((item, index) => (
          <LegendItem key={index}>
            <LegendColor color={item.color} />
            <LegendLabel>{item.label}</LegendLabel>
            <LegendValue>{item.value}</LegendValue>
          </LegendItem>
        ))}
      </LegendContainer>
    </>
  );
};

export default PortfolioAllocation;
