import React, { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

export default function RiskVisualization() {
  const [volatilityData, setVolatilityData] = useState([]);

  useEffect(() => {
    fetch("/api/volatility_history")
      .then((res) => res.json())
      .then((data) =>
        setVolatilityData(
          data.map((d, i) => ({
            day: i + 1,
            volatility: d,
          })),
        ),
      );
  }, []);

  return (
    <LineChart width={600} height={300} data={volatilityData}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="day" />
      <YAxis />
      <Tooltip />
      <Line type="monotone" dataKey="volatility" stroke="#8884d8" />
    </LineChart>
  );
}
