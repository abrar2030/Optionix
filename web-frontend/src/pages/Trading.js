import React, { useState } from "react";
import styled from "styled-components";
import { FiDollarSign, FiArrowUp, FiArrowDown, FiClock } from "react-icons/fi";

// Components for Dashboard
import PriceChart from "../components/dashboard/PriceChart";
import TradingForm from "../components/trading/TradingForm";
import OptionChain from "../components/trading/OptionChain";
import OrderBook from "../components/trading/OrderBook";

const TradingContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 20px;
`;

const ChartSection = styled.div`
  grid-column: span 8;
  background-color: ${(props) => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${(props) => props.theme.colors.border};

  @media (max-width: ${(props) => props.theme.breakpoints.tablet}) {
    grid-column: span 12;
  }
`;

const TradingFormSection = styled.div`
  grid-column: span 4;
  background-color: ${(props) => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${(props) => props.theme.colors.border};

  @media (max-width: ${(props) => props.theme.breakpoints.tablet}) {
    grid-column: span 12;
  }
`;

const OptionChainSection = styled.div`
  grid-column: span 8;
  background-color: ${(props) => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${(props) => props.theme.colors.border};

  @media (max-width: ${(props) => props.theme.breakpoints.tablet}) {
    grid-column: span 12;
  }
`;

const OrderBookSection = styled.div`
  grid-column: span 4;
  background-color: ${(props) => props.theme.colors.cardBg};
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 1px solid ${(props) => props.theme.colors.border};

  @media (max-width: ${(props) => props.theme.breakpoints.tablet}) {
    grid-column: span 12;
  }
`;

const CardTitle = styled.h2`
  font-size: 18px;
  font-weight: 600;
  color: ${(props) => props.theme.colors.textPrimary};
  margin: 0 0 20px 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const MarketInfo = styled.div`
  display: flex;
  gap: 15px;
  margin-bottom: 20px;
  flex-wrap: wrap;
`;

const MarketItem = styled.div`
  background-color: ${(props) => props.theme.colors.backgroundDark};
  border-radius: 6px;
  padding: 10px 15px;
  flex: 1;
  min-width: 120px;

  h4 {
    font-size: 12px;
    color: ${(props) => props.theme.colors.textSecondary};
    margin: 0 0 5px 0;
    display: flex;
    align-items: center;

    svg {
      margin-right: 5px;
    }
  }

  p {
    font-size: 16px;
    font-weight: 600;
    margin: 0;
    color: ${(props) => props.color || props.theme.colors.textPrimary};
  }
`;

const Trading = () => {
  const [selectedAsset, setSelectedAsset] = useState("BTC-USD");

  return (
    <TradingContainer>
      <ChartSection>
        <CardTitle>{selectedAsset} Price Chart</CardTitle>

        <MarketInfo>
          <MarketItem>
            <h4>
              <FiDollarSign /> Price
            </h4>
            <p>$42,567.89</p>
          </MarketItem>
          <MarketItem color={(props) => props.theme.colors.success}>
            <h4>
              <FiArrowUp /> 24h High
            </h4>
            <p>$43,120.45</p>
          </MarketItem>
          <MarketItem color={(props) => props.theme.colors.danger}>
            <h4>
              <FiArrowDown /> 24h Low
            </h4>
            <p>$41,890.32</p>
          </MarketItem>
          <MarketItem>
            <h4>
              <FiClock /> 24h Volume
            </h4>
            <p>$1.2B</p>
          </MarketItem>
        </MarketInfo>

        <PriceChart />
      </ChartSection>

      <TradingFormSection>
        <CardTitle>Trade Options</CardTitle>
        <TradingForm />
      </TradingFormSection>

      <OptionChainSection>
        <CardTitle>Option Chain</CardTitle>
        <OptionChain />
      </OptionChainSection>

      <OrderBookSection>
        <CardTitle>Order Book</CardTitle>
        <OrderBook />
      </OrderBookSection>
    </TradingContainer>
  );
};

export default Trading;
