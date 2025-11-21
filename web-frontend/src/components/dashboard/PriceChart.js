import React from "react";
import { Line } from "react-chartjs-2";
import styled from "styled-components";
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

const PriceChart = () => {
  const [timeframe, setTimeframe] = React.useState("1D");

  // Sample data
  const data = {
    labels: [
      "9:00",
      "10:00",
      "11:00",
      "12:00",
      "13:00",
      "14:00",
      "15:00",
      "16:00",
    ],
    datasets: [
      {
        label: "Price",
        data: [42300, 42150, 42400, 42800, 42600, 42900, 43100, 42700],
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
        mode: "index",
        intersect: false,
        backgroundColor: "#1e222d",
        titleColor: "#b2b5be",
        bodyColor: "white",
        borderColor: "#2a2e39",
        borderWidth: 1,
        padding: 10,
        displayColors: false,
        callbacks: {
          label: function (context) {
            return `$${context.raw.toLocaleString()}`;
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
            return "$" + value.toLocaleString();
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
        {["1H", "1D", "1W", "1M", "3M", "YTD", "1Y", "ALL"].map((time) => (
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

export default PriceChart;
