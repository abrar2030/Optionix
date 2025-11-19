import React, { useState, useEffect } from 'react';
import { StyleSheet, ScrollView, View, FlatList } from 'react-native';
import { ActivityIndicator, Card, Title, Paragraph, List, Divider, Text as PaperText, useTheme } from 'react-native-paper';
// Removed unused import: import { portfolioService } from '../services/api';

const PortfolioScreen = () => {
  const [portfolioSummary, setPortfolioSummary] = useState(null);
  const [positions, setPositions] = useState([]);
  const [performance, setPerformance] = useState([]);
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [loadingPositions, setLoadingPositions] = useState(true);
  const [loadingPerformance, setLoadingPerformance] = useState(true);
  const [error, setError] = useState(null);
  const theme = useTheme();

  useEffect(() => {
    const fetchPortfolioData = async () => {
      setLoadingSummary(true);
      setLoadingPositions(true);
      setLoadingPerformance(true);
      setError(null);
      try {
        // Using placeholder data
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
        setError('Failed to load portfolio data. Please try again later.');
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
    <List.Item
      title={`${item.symbol} (${item.type})`}
      description={`Qty: ${item.quantity}`}
      right={() => <PaperText style={styles.listItemValue}>${item.value.toFixed(2)}</PaperText>}
      titleStyle={styles.listItemTitle}
    />
  );

  const renderPerformanceItem = ({ item }) => (
    <List.Item
      title={item.date}
      right={() => <PaperText style={styles.listItemValue}>${item.value.toFixed(2)}</PaperText>}
      titleStyle={styles.listItemTitle}
    />
  );

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
      <Title style={styles.title}>Portfolio Management</Title>

      {error && <Paragraph style={styles.errorText}>{error}</Paragraph>}

      {/* Portfolio Summary Section */}
      <Card style={styles.card}>
        <Card.Title title="Summary" />
        <Card.Content>
          {loadingSummary ? (
            <ActivityIndicator animating={true} size="small" style={styles.loadingIndicator} />
          ) : portfolioSummary ? (
            <View>
              <Paragraph style={styles.summaryText}>Total Value: ${portfolioSummary.totalValue.toFixed(2)}</Paragraph>
              <View style={styles.summaryChangeContainer}>
                 <Paragraph style={styles.summaryText}>Day's Change: </Paragraph>
                 {renderChange(portfolioSummary.dayChange)}
              </View>
              <Paragraph style={styles.summaryText}>Cash Balance: ${portfolioSummary.cashBalance.toFixed(2)}</Paragraph>
            </View>
          ) : (
            <Paragraph>Could not load summary.</Paragraph>
          )}
        </Card.Content>
      </Card>

      {/* Positions Section */}
      <Card style={styles.card}>
        <Card.Title title="Positions" />
        <Card.Content>
          {loadingPositions ? (
            <ActivityIndicator animating={true} size="large" style={styles.loadingIndicator} />
          ) : (
            <FlatList
              data={positions}
              renderItem={renderPositionItem}
              keyExtractor={(item, index) => `${item.symbol}-${index}`}
              ItemSeparatorComponent={() => <Divider />}
              ListEmptyComponent={<Paragraph style={styles.emptyListText}>No positions found.</Paragraph>}
            />
          )}
        </Card.Content>
      </Card>

      {/* Performance History Section */}
      <Card style={styles.card}>
        <Card.Title title="Performance (Last Month)" />
        <Card.Content>
          {loadingPerformance ? (
            <ActivityIndicator animating={true} size="large" style={styles.loadingIndicator} />
          ) : (
            <FlatList
              data={performance}
              renderItem={renderPerformanceItem}
              keyExtractor={(item) => item.date}
              ItemSeparatorComponent={() => <Divider />}
              ListEmptyComponent={<Paragraph style={styles.emptyListText}>No performance data available.</Paragraph>}
            />
          )}
        </Card.Content>
      </Card>

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
  card: {
    marginBottom: 20,
    elevation: 4,
  },
  loadingIndicator: {
    marginVertical: 20,
  },
  summaryText: {
    fontSize: 16,
    marginBottom: 8,
  },
   summaryChangeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  listItemTitle: {
    fontSize: 16,
  },
  listItemValue: {
    fontSize: 16,
    fontWeight: 'bold',
    alignSelf: 'center',
  },
  positiveChange: {
    color: '#34C759',
    fontWeight: 'bold',
    fontSize: 16,
  },
  negativeChange: {
    color: '#FF3B30',
    fontWeight: 'bold',
    fontSize: 16,
  },
  errorText: {
    color: '#FF3B30',
    textAlign: 'center',
    marginVertical: 10,
    fontSize: 16,
    padding: 10,
  },
  emptyListText: {
    textAlign: 'center',
    marginVertical: 10,
    fontStyle: 'italic',
  },
});

export default PortfolioScreen;
