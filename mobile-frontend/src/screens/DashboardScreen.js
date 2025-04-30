import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, ScrollView } from 'react-native';
import { marketService } from '../services/api'; // Corrected import path

const DashboardScreen = () => {
  const [marketOverview, setMarketOverview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMarketData = async () => {
      try {
        setLoading(true);
        // NOTE: This assumes the backend API endpoint '/market/overview' exists and returns data.
        // In a real scenario, ensure the backend is running and accessible.
        // Using placeholder data for now as backend might not be running.
        // const data = await marketService.getMarketOverview();
        const placeholderData = {
          marketStatus: 'Open',
          majorIndices: [
            { name: 'S&P 500', value: 4500.50, change: '+0.5%' },
            { name: 'Dow Jones', value: 35000.75, change: '+0.3%' },
            { name: 'NASDAQ', value: 14000.25, change: '+0.8%' },
          ],
          topMovers: [
            { symbol: 'AAPL', price: 175.50, change: '+1.2%' },
            { symbol: 'GOOGL', price: 2800.00, change: '+0.9%' },
            { symbol: 'TSLA', price: 700.00, change: '-0.5%' },
          ]
        };
        await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate network delay
        setMarketOverview(placeholderData);
        setError(null);
      } catch (err) {
        console.error('Error fetching market overview:', err);
        setError('Failed to load market data. Please ensure the backend is running.');
        setMarketOverview(null); // Clear data on error
      } finally {
        setLoading(false);
      }
    };

    fetchMarketData();
  }, []);

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Market Dashboard</Text>

      {loading && <ActivityIndicator size="large" color="#007AFF" />}

      {error && <Text style={styles.errorText}>{error}</Text>}

      {marketOverview && (
        <View>
          <Text style={styles.sectionTitle}>Market Status: {marketOverview.marketStatus}</Text>
          
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Major Indices</Text>
            {marketOverview.majorIndices.map((index, i) => (
              <View key={i} style={styles.listItem}>
                <Text>{index.name}: {index.value}</Text>
                <Text style={index.change.startsWith('+') ? styles.positiveChange : styles.negativeChange}>
                  {index.change}
                </Text>
              </View>
            ))}
          </View>

          <View style={styles.card}>
            <Text style={styles.cardTitle}>Top Movers</Text>
            {marketOverview.topMovers.map((stock, i) => (
              <View key={i} style={styles.listItem}>
                <Text>{stock.symbol}: {stock.price}</Text>
                <Text style={stock.change.startsWith('+') ? styles.positiveChange : styles.negativeChange}>
                  {stock.change}
                </Text>
              </View>
            ))}
          </View>
        </View>
      )}
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
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 15,
    color: '#3A3A3C',
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
    elevation: 2, // for Android shadow
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#007AFF',
  },
  listItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  positiveChange: {
    color: '#34C759',
    fontWeight: '600',
  },
  negativeChange: {
    color: '#FF3B30',
    fontWeight: '600',
  },
  errorText: {
    color: '#FF3B30',
    textAlign: 'center',
    marginTop: 20,
    fontSize: 16,
  },
});

export default DashboardScreen;

