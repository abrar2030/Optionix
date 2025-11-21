import React from "react";
import styled from "styled-components";
import { FiArrowUp, FiArrowDown } from "react-icons/fi";

const TransactionsContainer = styled.div`
  width: 100%;
`;

const TransactionsList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: 300px;
  overflow-y: auto;

  &::-webkit-scrollbar {
    width: 6px;
  }

  &::-webkit-scrollbar-track {
    background: ${(props) => props.theme.colors.backgroundDark};
    border-radius: 3px;
  }

  &::-webkit-scrollbar-thumb {
    background: ${(props) => props.theme.colors.border};
    border-radius: 3px;
  }

  &::-webkit-scrollbar-thumb:hover {
    background: ${(props) => props.theme.colors.textSecondary};
  }
`;

const Transaction = styled.div`
  display: flex;
  align-items: center;
  padding: 12px;
  border-radius: 6px;
  background-color: ${(props) => props.theme.colors.backgroundDark};
  border: 1px solid ${(props) => props.theme.colors.border};
`;

const TransactionIcon = styled.div`
  width: 36px;
  height: 36px;
  border-radius: 6px;
  background-color: ${(props) =>
    props.type === "buy"
      ? "rgba(38, 166, 154, 0.1)"
      : "rgba(239, 83, 80, 0.1)"};
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 12px;

  svg {
    color: ${(props) =>
      props.type === "buy"
        ? props.theme.colors.success
        : props.theme.colors.danger};
    font-size: 18px;
  }
`;

const TransactionInfo = styled.div`
  flex: 1;
`;

const TransactionTitle = styled.div`
  font-size: 14px;
  font-weight: 500;
  color: ${(props) => props.theme.colors.textPrimary};
  margin-bottom: 4px;
`;

const TransactionDetails = styled.div`
  font-size: 12px;
  color: ${(props) => props.theme.colors.textSecondary};
`;

const TransactionAmount = styled.div`
  font-size: 14px;
  font-weight: 600;
  color: ${(props) =>
    props.type === "buy"
      ? props.theme.colors.success
      : props.theme.colors.danger};
`;

const EmptyState = styled.div`
  padding: 20px;
  text-align: center;
  color: ${(props) => props.theme.colors.textSecondary};
  font-size: 14px;
`;

const RecentTransactions = () => {
  // Sample data
  const transactions = [
    {
      id: 1,
      type: "buy",
      title: "BTC-USD Call Option",
      details: "Strike: $45,000 | Exp: Apr 30, 2025",
      amount: "+2 Contracts",
      timestamp: "2 hours ago",
    },
    {
      id: 2,
      type: "sell",
      title: "ETH-USD Put Option",
      details: "Strike: $3,200 | Exp: May 15, 2025",
      amount: "-5 Contracts",
      timestamp: "5 hours ago",
    },
    {
      id: 3,
      type: "buy",
      title: "AAPL Call Option",
      details: "Strike: $180 | Exp: Jun 20, 2025",
      amount: "+10 Contracts",
      timestamp: "1 day ago",
    },
    {
      id: 4,
      type: "sell",
      title: "TSLA Put Option",
      details: "Strike: $210 | Exp: May 5, 2025",
      amount: "-3 Contracts",
      timestamp: "2 days ago",
    },
    {
      id: 5,
      type: "buy",
      title: "SOL-USD Call Option",
      details: "Strike: $120 | Exp: Apr 25, 2025",
      amount: "+8 Contracts",
      timestamp: "3 days ago",
    },
  ];

  return (
    <TransactionsContainer>
      {transactions.length > 0 ? (
        <TransactionsList>
          {transactions.map((transaction) => (
            <Transaction key={transaction.id}>
              <TransactionIcon type={transaction.type}>
                {transaction.type === "buy" ? <FiArrowUp /> : <FiArrowDown />}
              </TransactionIcon>
              <TransactionInfo>
                <TransactionTitle>{transaction.title}</TransactionTitle>
                <TransactionDetails>
                  {transaction.details} • {transaction.timestamp}
                </TransactionDetails>
              </TransactionInfo>
              <TransactionAmount type={transaction.type}>
                {transaction.amount}
              </TransactionAmount>
            </Transaction>
          ))}
        </TransactionsList>
      ) : (
        <EmptyState>No recent transactions</EmptyState>
      )}
    </TransactionsContainer>
  );
};

export default RecentTransactions;
