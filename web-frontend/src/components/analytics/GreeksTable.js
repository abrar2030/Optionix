import React from "react";
import styled from "styled-components";

const TableContainer = styled.div`
  width: 100%;
  overflow-x: auto;
`;

const StyledGreeksTable = styled.table`
  width: 100%;
  border-collapse: collapse;
  min-width: 500px;
`;

const TableHead = styled.thead`
  tr {
    border-bottom: 1px solid ${(props) => props.theme.colors.border};
  }

  th {
    padding: 10px;
    text-align: left;
    font-size: 12px;
    font-weight: 500;
    color: ${(props) => props.theme.colors.textSecondary};
  }
`;

const TableBody = styled.tbody`
  tr {
    border-bottom: 1px solid ${(props) => props.theme.colors.border};

    &:last-child {
      border-bottom: none;
    }

    &:hover {
      background-color: rgba(42, 46, 57, 0.3);
    }
  }

  td {
    padding: 12px 10px;
    font-size: 13px;
    color: ${(props) => props.theme.colors.textPrimary};
  }
`;

const StrikeCell = styled.td`
  font-weight: 600;
`;

const GreeksTable = () => {
  // Sample data
  const greeksData = [
    {
      id: 1,
      strike: "$42,000",
      type: "Call",
      delta: 0.65,
      gamma: 0.0025,
      theta: -15.32,
      vega: 25.45,
      rho: 12.35,
    },
    {
      id: 2,
      strike: "$42,000",
      type: "Put",
      delta: -0.35,
      gamma: 0.0025,
      theta: -12.45,
      vega: 25.45,
      rho: -12.35,
    },
    {
      id: 3,
      strike: "$44,000",
      type: "Call",
      delta: 0.45,
      gamma: 0.0032,
      theta: -18.75,
      vega: 28.65,
      rho: 10.25,
    },
    {
      id: 4,
      strike: "$44,000",
      type: "Put",
      delta: -0.55,
      gamma: 0.0032,
      theta: -15.85,
      vega: 28.65,
      rho: -10.25,
    },
    {
      id: 5,
      strike: "$46,000",
      type: "Call",
      delta: 0.25,
      gamma: 0.0028,
      theta: -16.45,
      vega: 24.35,
      rho: 8.15,
    },
    {
      id: 6,
      strike: "$46,000",
      type: "Put",
      delta: -0.75,
      gamma: 0.0028,
      theta: -14.25,
      vega: 24.35,
      rho: -8.15,
    },
  ];

  return (
    <TableContainer>
      <StyledGreeksTable>
        <TableHead>
          <tr>
            <th>Strike</th>
            <th>Type</th>
            <th>Delta</th>
            <th>Gamma</th>
            <th>Theta</th>
            <th>Vega</th>
            <th>Rho</th>
          </tr>
        </TableHead>
        <TableBody>
          {greeksData.map((row) => (
            <tr key={row.id}>
              <StrikeCell>{row.strike}</StrikeCell>
              <td>{row.type}</td>
              <td>{row.delta.toFixed(2)}</td>
              <td>{row.gamma.toFixed(4)}</td>
              <td>{row.theta.toFixed(2)}</td>
              <td>{row.vega.toFixed(2)}</td>
              <td>{row.rho.toFixed(2)}</td>
            </tr>
          ))}
        </TableBody>
      </StyledGreeksTable>
    </TableContainer>
  );
};

export default GreeksTable;
