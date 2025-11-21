import React from "react";
import styled from "styled-components";

const MatrixContainer = styled.div`
  width: 100%;
`;

const MatrixGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  grid-template-rows: repeat(5, 1fr);
  gap: 8px;
  margin-top: 16px;
`;

const MatrixCell = styled.div`
  aspect-ratio: 1;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 12px;
  color: white;
  background-color: ${(props) => {
    if (props.risk === "very-low") return "#26a69a";
    if (props.risk === "low") return "#66bb6a";
    if (props.risk === "medium") return "#ffca28";
    if (props.risk === "high") return "#ff7043";
    if (props.risk === "very-high") return "#ef5350";
    return props.theme.colors.backgroundLight;
  }};
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }
`;

const LegendContainer = styled.div`
  display: flex;
  justify-content: space-between;
  margin-top: 16px;
`;

const LegendItem = styled.div`
  display: flex;
  align-items: center;
`;

const LegendColor = styled.div`
  width: 12px;
  height: 12px;
  border-radius: 2px;
  background-color: ${(props) => {
    if (props.risk === "very-low") return "#26a69a";
    if (props.risk === "low") return "#66bb6a";
    if (props.risk === "medium") return "#ffca28";
    if (props.risk === "high") return "#ff7043";
    if (props.risk === "very-high") return "#ef5350";
    return props.theme.colors.backgroundLight;
  }};
  margin-right: 6px;
`;

const LegendLabel = styled.span`
  font-size: 12px;
  color: ${(props) => props.theme.colors.textSecondary};
`;

const AxisLabel = styled.div`
  font-size: 12px;
  color: ${(props) => props.theme.colors.textSecondary};
  margin: ${(props) => props.margin || "0"};
  text-align: ${(props) => props.align || "center"};
`;

const RiskMatrix = () => {
  // Generate matrix data
  const generateMatrix = () => {
    const matrix = [];
    const risks = ["very-low", "low", "medium", "high", "very-high"];

    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 5; j++) {
        // Determine risk level based on position
        let risk;
        const sum = i + j;

        if (sum <= 1) risk = "very-low";
        else if (sum <= 3) risk = "low";
        else if (sum <= 5) risk = "medium";
        else if (sum <= 7) risk = "high";
        else risk = "very-high";

        matrix.push({
          id: `${i}-${j}`,
          x: j,
          y: i,
          risk,
        });
      }
    }

    return matrix;
  };

  const matrix = generateMatrix();

  return (
    <MatrixContainer>
      <AxisLabel margin="0 0 8px 0">Impact</AxisLabel>
      <div style={{ display: "flex" }}>
        <AxisLabel
          style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
          margin="0 8px 0 0"
        >
          Probability
        </AxisLabel>
        <div style={{ flex: 1 }}>
          <MatrixGrid>
            {matrix.map((cell) => (
              <MatrixCell
                key={cell.id}
                risk={cell.risk}
                style={{ gridColumn: cell.x + 1, gridRow: cell.y + 1 }}
              />
            ))}
          </MatrixGrid>
        </div>
      </div>

      <LegendContainer>
        <LegendItem>
          <LegendColor risk="very-low" />
          <LegendLabel>Very Low</LegendLabel>
        </LegendItem>
        <LegendItem>
          <LegendColor risk="low" />
          <LegendLabel>Low</LegendLabel>
        </LegendItem>
        <LegendItem>
          <LegendColor risk="medium" />
          <LegendLabel>Medium</LegendLabel>
        </LegendItem>
        <LegendItem>
          <LegendColor risk="high" />
          <LegendLabel>High</LegendLabel>
        </LegendItem>
        <LegendItem>
          <LegendColor risk="very-high" />
          <LegendLabel>Very High</LegendLabel>
        </LegendItem>
      </LegendContainer>
    </MatrixContainer>
  );
};

export default RiskMatrix;
