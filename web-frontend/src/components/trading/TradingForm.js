import React, { useState } from "react";
import styled from "styled-components";
import {
  FiDollarSign,
  FiPercent,
  FiCalendar,
  FiCheckCircle,
} from "react-icons/fi";

const TradingFormContainer = styled.div`
  width: 100%;
`;

const FormGroup = styled.div`
  margin-bottom: 16px;
`;

const Label = styled.label`
  display: block;
  font-size: 12px;
  font-weight: 500;
  color: ${(props) => props.theme.colors.textSecondary};
  margin-bottom: 6px;
`;

const Input = styled.input`
  width: 100%;
  background-color: ${(props) => props.theme.colors.backgroundDark};
  border: 1px solid ${(props) => props.theme.colors.border};
  border-radius: 4px;
  padding: 10px 12px;
  color: ${(props) => props.theme.colors.textPrimary};
  font-size: 14px;

  &:focus {
    outline: none;
    border-color: ${(props) => props.theme.colors.primary};
  }
`;

const Select = styled.select`
  width: 100%;
  background-color: ${(props) => props.theme.colors.backgroundDark};
  border: 1px solid ${(props) => props.theme.colors.border};
  border-radius: 4px;
  padding: 10px 12px;
  color: ${(props) => props.theme.colors.textPrimary};
  font-size: 14px;
  appearance: none;
  background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23b2b5be' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
  background-repeat: no-repeat;
  background-position: right 12px center;
  background-size: 16px;

  &:focus {
    outline: none;
    border-color: ${(props) => props.theme.colors.primary};
  }
`;

const ToggleContainer = styled.div`
  display: flex;
  margin-bottom: 16px;
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid ${(props) => props.theme.colors.border};
`;

const ToggleButton = styled.button`
  flex: 1;
  padding: 10px;
  background-color: ${(props) =>
    props.active ? props.activeColor : props.theme.colors.backgroundDark};
  color: ${(props) =>
    props.active ? "white" : props.theme.colors.textSecondary};
  border: none;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background-color: ${(props) =>
      props.active ? props.activeColor : "rgba(42, 46, 57, 0.8)"};
  }
`;

const SubmitButton = styled.button`
  width: 100%;
  padding: 12px;
  background-color: ${(props) => props.theme.colors.primary};
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover {
    background-color: ${(props) => props.theme.colors.primaryDark};
  }

  svg {
    margin-right: 8px;
  }
`;

const PricePreview = styled.div`
  background-color: ${(props) => props.theme.colors.backgroundDark};
  border: 1px solid ${(props) => props.theme.colors.border};
  border-radius: 4px;
  padding: 12px;
  margin-bottom: 16px;
`;

const PriceRow = styled.div`
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;

  &:last-child {
    margin-bottom: 0;
    padding-top: 8px;
    border-top: 1px solid ${(props) => props.theme.colors.border};
  }
`;

const PriceLabel = styled.span`
  font-size: 12px;
  color: ${(props) => props.theme.colors.textSecondary};
`;

const PriceValue = styled.span`
  font-size: 12px;
  font-weight: 500;
  color: ${(props) => props.theme.colors.textPrimary};
`;

const TotalValue = styled.span`
  font-size: 14px;
  font-weight: 600;
  color: ${(props) => props.theme.colors.textPrimary};
`;

const TradingForm = () => {
  const [optionType, setOptionType] = useState("call");
  const [positionType, setPositionType] = useState("buy");

  return (
    <TradingFormContainer>
      <ToggleContainer>
        <ToggleButton
          active={optionType === "call"}
          activeColor="#26a69a"
          onClick={() => setOptionType("call")}
        >
          Call
        </ToggleButton>
        <ToggleButton
          active={optionType === "put"}
          activeColor="#ef5350"
          onClick={() => setOptionType("put")}
        >
          Put
        </ToggleButton>
      </ToggleContainer>

      <ToggleContainer>
        <ToggleButton
          active={positionType === "buy"}
          activeColor="#26a69a"
          onClick={() => setPositionType("buy")}
        >
          Buy
        </ToggleButton>
        <ToggleButton
          active={positionType === "sell"}
          activeColor="#ef5350"
          onClick={() => setPositionType("sell")}
        >
          Sell
        </ToggleButton>
      </ToggleContainer>

      <FormGroup>
        <Label>Strike Price</Label>
        <Input
          type="number"
          placeholder="Enter strike price"
          defaultValue="45000"
        />
      </FormGroup>

      <FormGroup>
        <Label>Expiration Date</Label>
        <Select>
          <option value="20250430">Apr 30, 2025</option>
          <option value="20250515">May 15, 2025</option>
          <option value="20250530">May 30, 2025</option>
          <option value="20250620">Jun 20, 2025</option>
          <option value="20250720">Jul 20, 2025</option>
        </Select>
      </FormGroup>

      <FormGroup>
        <Label>Quantity (Contracts)</Label>
        <Input
          type="number"
          placeholder="Enter quantity"
          defaultValue="1"
          min="1"
        />
      </FormGroup>

      <PricePreview>
        <PriceRow>
          <PriceLabel>Premium per Contract</PriceLabel>
          <PriceValue>$1,250.00</PriceValue>
        </PriceRow>
        <PriceRow>
          <PriceLabel>Implied Volatility</PriceLabel>
          <PriceValue>32.5%</PriceValue>
        </PriceRow>
        <PriceRow>
          <PriceLabel>Max Profit</PriceLabel>
          <PriceValue>Unlimited</PriceValue>
        </PriceRow>
        <PriceRow>
          <PriceLabel>Max Loss</PriceLabel>
          <PriceValue>$1,250.00</PriceValue>
        </PriceRow>
        <PriceRow>
          <PriceLabel>Total Cost</PriceLabel>
          <TotalValue>$1,250.00</TotalValue>
        </PriceRow>
      </PricePreview>

      <SubmitButton>
        <FiCheckCircle />
        {positionType === "buy" ? "Buy" : "Sell"} {optionType.toUpperCase()}
      </SubmitButton>
    </TradingFormContainer>
  );
};

export default TradingForm;
