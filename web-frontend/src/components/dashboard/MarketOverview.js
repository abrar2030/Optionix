import React from "react";
import styled from "styled-components";
import { FiArrowUp, FiArrowDown, FiInfo } from "react-icons/fi";

const MarketOverviewContainer = styled.div`
  width: 100%;
`;

const MarketTable = styled.table`
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

const PriceChange = styled.div`
  display: flex;
  align-items: center;
  color: ${(props) =>
    props.isPositive ? props.theme.colors.success : props.theme.colors.danger};

  svg {
    margin-right: 4px;
  }
`;

const InfoIcon = styled.div`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-left: 6px;
  color: ${(props) => props.theme.colors.textSecondary};
  cursor: help;
  position: relative;

  &:hover::after {
    content: "${(props) => props.tooltip}";
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background-color: ${(props) => props.theme.colors.backgroundLight};
    color: ${(props) => props.theme.colors.textPrimary};
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 12px;
    white-space: nowrap;
    z-index: 10;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    border: 1px solid ${(props) => props.theme.colors.border};
  }
`;

const MarketOverview = () => {
  // Sample data
  const marketData = [
    {
      id: 1,
      name: "BTC",
      fullName: "Bitcoin",
      price: "$42,567.89",
      change: "+2.34%",
      isPositive: true,
      volume: "$28.5B",
      ivRank: "45%",
      color: "#f7931a",
    },
    {
      id: 2,
      name: "ETH",
      fullName: "Ethereum",
      price: "$3,245.67",
      change: "+1.56%",
      isPositive: true,
      volume: "$15.2B",
      ivRank: "38%",
      color: "#627eea",
    },
    {
      id: 3,
      name: "SOL",
      fullName: "Solana",
      price: "$118.45",
      change: "-0.78%",
      isPositive: false,
      volume: "$4.8B",
      ivRank: "52%",
      color: "#00ffbd",
    },
    {
      id: 4,
      name: "AAPL",
      fullName: "Apple Inc.",
      price: "$178.32",
      change: "+0.45%",
      isPositive: true,
      volume: "$3.2B",
      ivRank: "32%",
      color: "#a2aaad",
    },
    {
      id: 5,
      name: "TSLA",
      fullName: "Tesla Inc.",
      price: "$215.67",
      change: "-1.23%",
      isPositive: false,
      volume: "$5.1B",
      ivRank: "65%",
      color: "#cc0000",
    },
  ];

  return (
    <MarketOverviewContainer>
      <MarketTable>
        <TableHead>
          <tr>
            <th>Asset</th>
            <th>Price</th>
            <th>24h Change</th>
            <th>Volume</th>
            <th>
              IV Rank{" "}
              <InfoIcon tooltip="Implied Volatility Rank">
                <FiInfo size={12} />
              </InfoIcon>
            </th>
          </tr>
        </TableHead>
        <TableBody>
          {marketData.map((asset) => (
            <tr key={asset.id}>
              <td>
                <AssetCell>
                  <AssetIcon color={asset.color}>
                    {asset.name.substring(0, 1)}
                  </AssetIcon>
                  <AssetName>{asset.fullName}</AssetName>
                </AssetCell>
              </td>
              <td>{asset.price}</td>
              <td>
                <PriceChange isPositive={asset.isPositive}>
                  {asset.isPositive ? <FiArrowUp /> : <FiArrowDown />}
                  {asset.change}
                </PriceChange>
              </td>
              <td>{asset.volume}</td>
              <td>{asset.ivRank}</td>
            </tr>
          ))}
        </TableBody>
      </MarketTable>
    </MarketOverviewContainer>
  );
};

export default MarketOverview;
