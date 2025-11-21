import React from "react";
import styled from "styled-components";

const OptionChainContainer = styled.div`
  width: 100%;
  overflow-x: auto;
`;

const OptionTable = styled.table`
  width: 100%;
  border-collapse: collapse;
  min-width: 700px;
`;

const TableHead = styled.thead`
  tr {
    border-bottom: 1px solid ${(props) => props.theme.colors.border};
  }

  th {
    padding: 10px;
    text-align: center;
    font-size: 12px;
    font-weight: 500;
    color: ${(props) => props.theme.colors.textSecondary};
    position: sticky;
    top: 0;
    background-color: ${(props) => props.theme.colors.cardBg};
    z-index: 1;
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
    padding: 10px;
    font-size: 12px;
    text-align: center;
    color: ${(props) => props.theme.colors.textPrimary};
  }
`;

const StrikeCell = styled.td`
  font-weight: 600;
  background-color: ${(props) => props.theme.colors.backgroundDark};
  color: ${(props) => props.theme.colors.textPrimary};
`;

const CallCell = styled.td`
  color: ${(props) => props.theme.colors.success};
`;

const PutCell = styled.td`
  color: ${(props) => props.theme.colors.danger};
`;

const ExpirySelector = styled.div`
  display: flex;
  gap: 10px;
  margin-bottom: 16px;
  overflow-x: auto;
  padding-bottom: 6px;

  &::-webkit-scrollbar {
    height: 4px;
  }

  &::-webkit-scrollbar-track {
    background: ${(props) => props.theme.colors.backgroundDark};
    border-radius: 2px;
  }

  &::-webkit-scrollbar-thumb {
    background: ${(props) => props.theme.colors.border};
    border-radius: 2px;
  }
`;

const ExpiryButton = styled.button`
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
  white-space: nowrap;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background-color: ${(props) =>
      props.active ? props.theme.colors.primary : "rgba(41, 98, 255, 0.1)"};
    color: ${(props) => (props.active ? "white" : props.theme.colors.primary)};
  }
`;

const OptionChain = () => {
  const [selectedExpiry, setSelectedExpiry] = React.useState("20250430");

  // Sample data
  const expiryDates = [
    { value: "20250430", label: "Apr 30, 2025" },
    { value: "20250515", label: "May 15, 2025" },
    { value: "20250530", label: "May 30, 2025" },
    { value: "20250620", label: "Jun 20, 2025" },
    { value: "20250720", label: "Jul 20, 2025" },
    { value: "20250820", label: "Aug 20, 2025" },
    { value: "20250920", label: "Sep 20, 2025" },
  ];

  const optionChainData = [
    {
      strike: 40000,
      call: { bid: 3250.45, ask: 3275.2, iv: 32.5, volume: 145, oi: 1250 },
      put: { bid: 750.3, ask: 765.45, iv: 30.2, volume: 98, oi: 850 },
    },
    {
      strike: 42000,
      call: { bid: 1850.75, ask: 1875.3, iv: 31.8, volume: 210, oi: 1850 },
      put: { bid: 1350.2, ask: 1375.45, iv: 31.5, volume: 175, oi: 1450 },
    },
    {
      strike: 44000,
      call: { bid: 950.25, ask: 975.5, iv: 30.5, volume: 320, oi: 2250 },
      put: { bid: 2450.75, ask: 2475.3, iv: 32.8, volume: 280, oi: 1950 },
    },
    {
      strike: 46000,
      call: { bid: 450.8, ask: 465.25, iv: 29.2, volume: 185, oi: 1650 },
      put: { bid: 3950.45, ask: 3975.3, iv: 33.5, volume: 145, oi: 1250 },
    },
    {
      strike: 48000,
      call: { bid: 175.45, ask: 185.3, iv: 28.5, volume: 95, oi: 850 },
      put: { bid: 5675.25, ask: 5700.5, iv: 34.2, volume: 75, oi: 650 },
    },
  ];

  return (
    <OptionChainContainer>
      <ExpirySelector>
        {expiryDates.map((date) => (
          <ExpiryButton
            key={date.value}
            active={selectedExpiry === date.value}
            onClick={() => setSelectedExpiry(date.value)}
          >
            {date.label}
          </ExpiryButton>
        ))}
      </ExpirySelector>

      <OptionTable>
        <TableHead>
          <tr>
            <th colSpan="5">Calls</th>
            <th rowSpan="2">Strike</th>
            <th colSpan="5">Puts</th>
          </tr>
          <tr>
            <th>Bid</th>
            <th>Ask</th>
            <th>IV%</th>
            <th>Vol</th>
            <th>OI</th>
            <th>Bid</th>
            <th>Ask</th>
            <th>IV%</th>
            <th>Vol</th>
            <th>OI</th>
          </tr>
        </TableHead>
        <TableBody>
          {optionChainData.map((row, index) => (
            <tr key={index}>
              <CallCell>{row.call.bid.toFixed(2)}</CallCell>
              <CallCell>{row.call.ask.toFixed(2)}</CallCell>
              <CallCell>{row.call.iv.toFixed(1)}</CallCell>
              <CallCell>{row.call.volume}</CallCell>
              <CallCell>{row.call.oi}</CallCell>
              <StrikeCell>${row.strike.toLocaleString()}</StrikeCell>
              <PutCell>{row.put.bid.toFixed(2)}</PutCell>
              <PutCell>{row.put.ask.toFixed(2)}</PutCell>
              <PutCell>{row.put.iv.toFixed(1)}</PutCell>
              <PutCell>{row.put.volume}</PutCell>
              <PutCell>{row.put.oi}</PutCell>
            </tr>
          ))}
        </TableBody>
      </OptionTable>
    </OptionChainContainer>
  );
};

export default OptionChain;
