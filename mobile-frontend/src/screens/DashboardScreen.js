import React, { useState, useEffect } from 'react';
import { StyleSheet, ScrollView, View } from 'react-native';
import { ActivityIndicator, Card, Title, Paragraph, List, Divider, Text as PaperText, useTheme } from 'react-native-paper';
// Removed unused import: import { marketService } from '../services/api';

const DashboardScreen = () => {
  const [marketOverview, setMarketOverview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const theme = useTheme(); // Access theme colors

  useEffect(() => {
    const fetchMarketData = async () => {
      try {
        setLoading(true);
        // Using placeholder data
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
        setError('Failed to load market data. Please try again later.');
        setMarketOverview(null);
      } finally {
        setLoading(false);
      }
    };

    fetchMarketData();
  }, []);

  const renderChange = (change) => {
    const isPositive = change.startsWith('+');
    return (
      <PaperText style={isPositive ? styles.positiveChange : styles.negativeChange}>
        {change}
      </PaperText>
    );
  };

  return (
    <ScrollView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <Title style={styles.title}>Market Dashboard</Title>

      {loading && <ActivityIndicator animating={true} size="large" style={styles.loadingIndicator} />}

      {error && <Paragraph style={styles.errorText}>{error}</Paragraph>}

      {marketOverview && (
        <View>
          <Card style={styles.card}>
            <Card.Content>
              <Title>Market Status: {marketOverview.marketStatus}</Title>
            </Card.Content>
          </Card>

          <Card style={styles.card}>
            <Card.Title title="Major Indices" />
            <Card.Content>
              {marketOverview.majorIndices.map((index, i) => (
                <React.Fragment key={i}>
                  <List.Item
                    title={`${index.name}: ${index.value.toFixed(2)}`}
                    right={() => renderChange(index.change)}
                    titleStyle={styles.listItemTitle}
                  />
                  {i < marketOverview.majorIndices.length - 1 && <Divider />}
                </React.Fragment>
              ))}
            </Card.Content>
          </Card>

          <Card style={styles.card}>
            <Card.Title title="Top Movers" />
            <Card.Content>
              {marketOverview.topMovers.map((stock, i) => (
                <React.Fragment key={i}>
                  <List.Item
                    title={`${stock.symbol}: ${stock.price.toFixed(2)}`}
                    right={() => renderChange(stock.change)}
                    titleStyle={styles.listItemTitle}
                  />
                  {i < marketOverview.topMovers.length - 1 && <Divider />}
                </React.Fragment>
              ))}
            </Card.Content>
          </Card>
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 15,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  loadingIndicator: {
    marginTop: 30,
    marginBottom: 30,
  },
  card: {
    marginBottom: 20,
    elevation: 4, // Added elevation for Paper Card shadow
  },
  listItemTitle: {
    fontSize: 16,
  },
  positiveChange: {
    color: '#34C759', // Green color for positive change
    fontWeight: 'bold',
    fontSize: 16,
  },
  negativeChange: {
    color: '#FF3B30', // Red color for negative change
    fontWeight: 'bold',
    fontSize: 16,
  },
  errorText: {
    color: '#FF3B30', // Use theme error color if available
    textAlign: 'center',
    marginTop: 20,
    fontSize: 16,
    padding: 10,
  },
});

export default DashboardScreen;
