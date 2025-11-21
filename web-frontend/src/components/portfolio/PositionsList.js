import React from "react";
import styled from "styled-components";
import { FiArrowUp, FiArrowDown, FiClock } from "react-icons/fi";

const PositionsContainer = styled.div`
  width: 100%;
`;

const PositionsTable = styled.table`
  width: 100%;
  border-collapse: collapse;
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

const AssetCell = styled.div`
  display: flex;
  align-items: center;
`;

const AssetIcon = styled.div`
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background-color: ${(props) => props.color || props.theme.colors.primary};
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 10px;
  font-size: 12px;
  font-weight: 600;
  color: white;
`;

const AssetName = styled.div`
  font-weight: 500;
`;

const AssetDetails = styled.div`
  font-size: 11px;
  color: ${(props) => props.theme.colors.textSecondary};
  margin-top: 2px;
`;

const ProfitLoss = styled.div`
  display: flex;
  align-items: center;
  color: ${(props) =>
    props.isPositive ? props.theme.colors.success : props.theme.colors.danger};

  svg {
    margin-right: 4px;
  }
`;

const ActionButton = styled.button`
  background-color: transparent;
  border: 1px solid ${(props) => props.theme.colors.border};
  border-radius: 4px;
  padding: 6px 10px;
  font-size: 12px;
  color: ${(props) => props.theme.colors.textSecondary};
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background-color: rgba(41, 98, 255, 0.1);
    color: ${(props) => props.theme.colors.primary};
    border-color: ${(props) => props.theme.colors.primary};
  }
`;

const EmptyState = styled.div`
  padding: 40px 20px;
  text-align: center;
  color: ${(props) => props.theme.colors.textSecondary};
  font-size: 14px;
`;

const PositionsList = () => {
  // Sample data
  const positions = [
    {
      id: 1,
      asset: "BTC",
      assetName: "Bitcoin",
      type: "Call Option",
      details: "Strike: $45,000 | Exp: Apr 30, 2025",
      size: "2 Contracts",
      entryPrice: "$1,250.00",
      currentPrice: "$1,450.00",
      profitLoss: "+$400.00",
      profitLossPercent: "+16.00%",
      isPositive: true,
      color: "#f7931a",
    },
    {
      id: 2,
      asset: "ETH",
      assetName: "Ethereum",
      type: "Put Option",
      details: "Strike: $3,200 | Exp: May 15, 2025",
      size: "5 Contracts",
      entryPrice: "$320.00",
      currentPrice: "$280.00",
      profitLoss: "+$200.00",
      profitLossPercent: "+12.50%",
      isPositive: true,
      color: "#627eea",
    },
    {
      id: 3,
      asset: "AAPL",
      assetName: "Apple Inc.",
      type: "Call Option",
      details: "Strike: $180 | Exp: Jun 20, 2025",
      size: "10 Contracts",
      entryPrice: "$4.50",
      currentPrice: "$5.25",
      profitLoss: "+$750.00",
      profitLossPercent: "+16.67%",
      isPositive: true,
      color: "#a2aaad",
    },
    {
      id: 4,
      asset: "TSLA",
      assetName: "Tesla Inc.",
      type: "Put Option",
      details: "Strike: $210 | Exp: May 5, 2025",
      size: "3 Contracts",
      entryPrice: "$12.50",
      currentPrice: "$10.75",
      profitLoss: "-$525.00",
      profitLossPercent: "-14.00%",
      isPositive: false,
      color: "#cc0000",
    },
    {
      id: 5,
      asset: "SOL",
      assetName: "Solana",
      type: "Call Option",
      details: "Strike: $120 | Exp: Apr 25, 2025",
      size: "8 Contracts",
      entryPrice: "$8.75",
      currentPrice: "$9.50",
      profitLoss: "+$600.00",
      profitLossPercent: "+8.57%",
      isPositive: true,
      color: "#00ffbd",
    },
  ];

  return (
    <PositionsContainer>
      {positions.length > 0 ? (
        <PositionsTable>
          <TableHead>
            <tr>
              <th>Asset</th>
              <th>Size</th>
              <th>Entry Price</th>
              <th>Current Price</th>
              <th>Profit/Loss</th>
              <th>Action</th>
            </tr>
          </TableHead>
          <TableBody>
            {positions.map((position) => (
              <tr key={position.id}>
                <td>
                  <AssetCell>
                    <AssetIcon color={position.color}>
                      {position.asset.substring(0, 1)}
                    </AssetIcon>
                    <div>
                      <AssetName>
                        {position.assetName} {position.type}
                      </AssetName>
                      <AssetDetails>{position.details}</AssetDetails>
                    </div>
                  </AssetCell>
                </td>
                <td>{position.size}</td>
                <td>{position.entryPrice}</td>
                <td>{position.currentPrice}</td>
                <td>
                  <ProfitLoss isPositive={position.isPositive}>
                    {position.isPositive ? <FiArrowUp /> : <FiArrowDown />}
                    {position.profitLoss} ({position.profitLossPercent})
                  </ProfitLoss>
                </td>
                <td>
                  <ActionButton>Close Position</ActionButton>
                </td>
              </tr>
            ))}
          </TableBody>
        </PositionsTable>
      ) : (
        <EmptyState>No open positions</EmptyState>
      )}
    </PositionsContainer>
  );
};

export default PositionsList;
