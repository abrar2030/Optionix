import React from "react";
import styled from "styled-components";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
);

const ChartContainer = styled.div`
  height: 300px;
  width: 100%;
`;

const TimeframeSelector = styled.div`
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
`;

const TimeButton = styled.button`
  background-color: ${(props) =>
    props.active ? props.theme.colors.primary : "transparent"};
  color: ${(props) =>
    props.active ? "white" : props.theme.colors.textSecondary};
  border: 1px solid
    ${(props) =>
      props.active ? props.theme.colors.primary : props.theme.colors.border};
  border-radius: 4px;
  padding: 6px 12px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background-color: ${(props) =>
      props.active ? props.theme.colors.primary : "rgba(41, 98, 255, 0.1)"};
    color: ${(props) => (props.active ? "white" : props.theme.colors.primary)};
  }
`;

const VolatilityChart = () => {
  const [timeframe, setTimeframe] = React.useState("1M");

  // Sample data
  const data = {
    labels: [
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec",
    ],
    datasets: [
      {
        label: "Historical Volatility",
        data: [28, 32, 25, 30, 35, 40, 38, 32, 28, 30, 34, 32],
        borderColor: "#2962ff",
        backgroundColor: "rgba(41, 98, 255, 0.1)",
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: "#2962ff",
        pointHoverBorderColor: "white",
        pointHoverBorderWidth: 2,
      },
      {
        label: "Implied Volatility",
        data: [30, 35, 28, 32, 38, 42, 40, 35, 30, 32, 36, 34],
        borderColor: "#ff6d00",
        backgroundColor: "rgba(255, 109, 0, 0.1)",
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: "#ff6d00",
        pointHoverBorderColor: "white",
        pointHoverBorderWidth: 2,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top",
        labels: {
          color: "#b2b5be",
          font: {
            size: 12,
          },
        },
      },
      tooltip: {
        mode: "index",
        intersect: false,
        backgroundColor: "#1e222d",
        titleColor: "#b2b5be",
        bodyColor: "white",
        borderColor: "#2a2e39",
        borderWidth: 1,
        padding: 10,
        displayColors: true,
        callbacks: {
          label: function (context) {
            return `${context.dataset.label}: ${context.raw}%`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: {
          display: false,
          drawBorder: false,
        },
        ticks: {
          color: "#b2b5be",
          font: {
            size: 10,
          },
        },
      },
      y: {
        grid: {
          color: "rgba(42, 46, 57, 0.6)",
          drawBorder: false,
        },
        ticks: {
          color: "#b2b5be",
          font: {
            size: 10,
          },
          callback: function (value) {
            return value + "%";
          },
        },
      },
    },
    interaction: {
      mode: "index",
      intersect: false,
    },
    elements: {
      line: {
        borderWidth: 2,
      },
    },
  };

  return (
    <>
      <TimeframeSelector>
        {["1W", "1M", "3M", "6M", "1Y", "2Y", "5Y"].map((time) => (
          <TimeButton
            key={time}
            active={timeframe === time}
            onClick={() => setTimeframe(time)}
          >
            {time}
          </TimeButton>
        ))}
      </TimeframeSelector>
      <ChartContainer>
        <Line data={data} options={options} />
      </ChartContainer>
    </>
  );
};

export default VolatilityChart;
