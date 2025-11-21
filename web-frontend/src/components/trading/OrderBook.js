import React from "react";
import styled from "styled-components";
import { FiArrowUp, FiArrowDown } from "react-icons/fi";

const OrderBookContainer = styled.div`
  width: 100%;
`;

const OrderBookHeader = styled.div`
  display: flex;
  justify-content: space-between;
  margin-bottom: 12px;
`;

const HeaderItem = styled.div`
  font-size: 12px;
  color: ${(props) => props.theme.colors.textSecondary};
`;

const OrderBookContent = styled.div`
  display: flex;
  flex-direction: column;
  height: 300px;
`;

const OrderSection = styled.div`
  flex: 1;
  overflow-y: auto;

  &::-webkit-scrollbar {
    width: 4px;
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

const OrderRow = styled.div`
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
  position: relative;
`;

const OrderPrice = styled.div`
  font-size: 12px;
  font-weight: 500;
  color: ${(props) =>
    props.type === "ask"
      ? props.theme.colors.danger
      : props.theme.colors.success};
  z-index: 1;
`;

const OrderSize = styled.div`
  font-size: 12px;
  color: ${(props) => props.theme.colors.textPrimary};
  z-index: 1;
`;

const OrderTotal = styled.div`
  font-size: 12px;
  color: ${(props) => props.theme.colors.textSecondary};
  z-index: 1;
`;

const OrderBackground = styled.div`
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: ${(props) => props.width}%;
  background-color: ${(props) =>
    props.type === "ask"
      ? "rgba(239, 83, 80, 0.1)"
      : "rgba(38, 166, 154, 0.1)"};
`;

const Divider = styled.div`
  height: 1px;
  background-color: ${(props) => props.theme.colors.border};
  margin: 8px 0;
`;

const SpreadRow = styled.div`
  display: flex;
  justify-content: center;
  padding: 6px 0;
  font-size: 12px;
  color: ${(props) => props.theme.colors.textSecondary};
`;

const OrderBook = () => {
  // Sample data
  const asks = [
    { price: 42950.5, size: 0.85, total: 36508.93 },
    { price: 42900.25, size: 1.2, total: 51480.3 },
    { price: 42875.75, size: 0.65, total: 27869.24 },
    { price: 42850.0, size: 2.1, total: 89985.0 },
    { price: 42825.5, size: 1.45, total: 62097.98 },
    { price: 42800.75, size: 0.95, total: 40660.71 },
    { price: 42775.25, size: 1.75, total: 74856.69 },
    { price: 42750.0, size: 1.3, total: 55575.0 },
  ].reverse();

  const bids = [
    { price: 42700.0, size: 1.5, total: 64050.0 },
    { price: 42675.5, size: 0.9, total: 38407.95 },
    { price: 42650.25, size: 1.85, total: 78902.96 },
    { price: 42625.75, size: 1.1, total: 46888.33 },
    { price: 42600.0, size: 2.25, total: 95850.0 },
    { price: 42575.5, size: 0.75, total: 31931.63 },
    { price: 42550.25, size: 1.4, total: 59570.35 },
    { price: 42525.0, size: 1.05, total: 44651.25 },
  ];

  const maxSize = Math.max(...[...asks, ...bids].map((order) => order.size));
  const spread = asks[0].price - bids[0].price;
  const spreadPercentage = (spread / asks[0].price) * 100;

  return (
    <OrderBookContainer>
      <OrderBookHeader>
        <HeaderItem>Price (USD)</HeaderItem>
        <HeaderItem>Size (BTC)</HeaderItem>
        <HeaderItem>Total (USD)</HeaderItem>
      </OrderBookHeader>

      <OrderBookContent>
        <OrderSection>
          {asks.map((ask, index) => (
            <OrderRow key={index}>
              <OrderPrice type="ask">{ask.price.toFixed(2)}</OrderPrice>
              <OrderSize>{ask.size.toFixed(2)}</OrderSize>
              <OrderTotal>{ask.total.toFixed(2)}</OrderTotal>
              <OrderBackground type="ask" width={(ask.size / maxSize) * 100} />
            </OrderRow>
          ))}
        </OrderSection>

        <Divider />

        <SpreadRow>
          Spread: ${spread.toFixed(2)} ({spreadPercentage.toFixed(3)}%)
        </SpreadRow>

        <Divider />

        <OrderSection>
          {bids.map((bid, index) => (
            <OrderRow key={index}>
              <OrderPrice type="bid">{bid.price.toFixed(2)}</OrderPrice>
              <OrderSize>{bid.size.toFixed(2)}</OrderSize>
              <OrderTotal>{bid.total.toFixed(2)}</OrderTotal>
              <OrderBackground type="bid" width={(bid.size / maxSize) * 100} />
            </OrderRow>
          ))}
        </OrderSection>
      </OrderBookContent>
    </OrderBookContainer>
  );
};

export default OrderBook;
