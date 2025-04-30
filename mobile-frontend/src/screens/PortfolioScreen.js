import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, ActivityIndicator, FlatList } from 'react-native';
import { portfolioService } from '../services/api'; // Corrected import path

const PortfolioScreen = () => {
  const [portfolioSummary, setPortfolioSummary] = useState(null);
  const [positions, setPositions] = useState([]);
  const [performance, setPerformance] = useState([]);
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [loadingPositions, setLoadingPositions] = useState(true);
  const [loadingPerformance, setLoadingPerformance] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPortfolioData = async () => {
      setLoadingSummary(true);
      setLoadingPositions(true);
      setLoadingPerformance(true);
      setError(null);
      try {
        // Using placeholder data as backend might not be running
        // const summaryData = await portfolioService.getPortfolioSummary();
        // const positionsData = await portfolioService.getPositions();
        // const performanceData = await portfolioService.getPerformanceHistory('1M'); // Example timeframe

        const placeholderSummary = {
          totalValue: 150000.00,
          dayChange: '+1.5%',
          cashBalance: 25000.00,
        };
        const placeholderPositions = [
          { symbol: 'AAPL', quantity: 100, value: 17550.00, type: 'Stock' },
          { symbol: 'GOOGL', quantity: 10, value: 28000.00, type: 'Stock' },
          { symbol: 'BTC-USD', quantity: 0.5, value: 30000.00, type: 'Crypto' },
          { symbol: 'AAPL 251219C180', quantity: 5, value: 2400.00, type: 'Option' },
        ];
        const placeholderPerformance = [
          { date: '2025-04-01', value: 145000 },
          { date: '2025-04-15', value: 148000 },
          { date: '2025-04-29', value: 150000 },
        ];

        await new Promise(resolve => setTimeout(resolve, 1100)); // Simulate network delay

        setPortfolioSummary(placeholderSummary);
        setPositions(placeholderPositions);
        setPerformance(placeholderPerformance);

      } catch (err) {
        console.error('Error fetching portfolio data:', err);
        setError('Failed to load portfolio data. Please ensure the backend is running.');
        setPortfolioSummary(null);
        setPositions([]);
        setPerformance([]);
      } finally {
        setLoadingSummary(false);
        setLoadingPositions(false);
        setLoadingPerformance(false);
      }
    };

    fetchPortfolioData();
  }, []);

  const renderPositionItem = ({ item }) => (
    <View style={styles.listItem}>
      <Text>{item.symbol} ({item.type})</Text>
      <Text>Qty: {item.quantity}</Text>
      <Text>Value: ${item.value.toFixed(2)}</Text>
    </View>
  );

  const renderPerformanceItem = ({ item }) => (
    <View style={styles.listItem}>
      <Text>{item.date}</Text>
      <Text>Value: ${item.value.toFixed(2)}</Text>
    </View>
  );

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Portfolio Management</Text>

      {error && <Text style={styles.errorText}>{error}</Text>}

      {/* Portfolio Summary Section */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Summary</Text>
        {loadingSummary ? (
          <ActivityIndicator size="small" color="#007AFF" />
        ) : portfolioSummary ? (
          <View>
            <Text style={styles.summaryText}>Total Value: ${portfolioSummary.totalValue.toFixed(2)}</Text>
            <Text style={styles.summaryText}>Day's Change: {portfolioSummary.dayChange}</Text>
            <Text style={styles.summaryText}>Cash Balance: ${portfolioSummary.cashBalance.toFixed(2)}</Text>
          </View>
        ) : (
          <Text>Could not load summary.</Text>
        )}
      </View>

      {/* Positions Section */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Positions</Text>
        {loadingPositions ? (
          <ActivityIndicator size="large" color="#007AFF" />
        ) : (
          <FlatList
            data={positions}
            renderItem={renderPositionItem}
            keyExtractor={(item, index) => `${item.symbol}-${index}`}
            ListEmptyComponent={<Text>No positions found.</Text>}
          />
        )}
      </View>

      {/* Performance History Section */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Performance (Last Month)</Text>
        {loadingPerformance ? (
          <ActivityIndicator size="large" color="#007AFF" />
        ) : (
          <FlatList
            data={performance}
            renderItem={renderPerformanceItem}
            keyExtractor={(item) => item.date}
            ListEmptyComponent={<Text>No performance data available.</Text>}
          />
        )}
      </View>

    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 15,
    backgroundColor: '#F5F5F7',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#1C1C1E',
    textAlign: 'center',
  },
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 10,
    padding: 15,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#007AFF',
  },
  summaryText: {
    fontSize: 16,
    marginBottom: 5,
  },
  listItem: {
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  errorText: {
    color: '#FF3B30',
    textAlign: 'center',
    marginVertical: 10,
    fontSize: 16,
  },
});

export default PortfolioScreen;

