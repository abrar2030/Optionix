import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, ActivityIndicator, FlatList } from 'react-native';
import { analyticsService } from '../services/api'; // Corrected import path

const AnalyticsScreen = () => {
  const [riskAssessment, setRiskAssessment] = useState(null);
  const [volatilityAnalysis, setVolatilityAnalysis] = useState([]);
  const [marketSentiment, setMarketSentiment] = useState(null);
  const [loadingRisk, setLoadingRisk] = useState(true);
  const [loadingVolatility, setLoadingVolatility] = useState(true);
  const [loadingSentiment, setLoadingSentiment] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAnalyticsData = async () => {
      setLoadingRisk(true);
      setLoadingVolatility(true);
      setLoadingSentiment(true);
      setError(null);
      try {
        // Using placeholder data as backend might not be running
        // const riskData = await analyticsService.getRiskAssessment();
        // const volatilityData = await analyticsService.getVolatilityAnalysis('AAPL', '1M'); // Example symbol/timeframe
        // const sentimentData = await analyticsService.getMarketSentiment();

        const placeholderRisk = {
          overallScore: 75,
          factors: [
            { name: 'Market Risk', level: 'High' },
            { name: 'Credit Risk', level: 'Low' },
            { name: 'Liquidity Risk', level: 'Medium' },
          ],
          recommendation: 'Consider hedging strategies for market exposure.',
        };
        const placeholderVolatility = [
          { date: '2025-04-01', impliedVol: 0.25, historicalVol: 0.22 },
          { date: '2025-04-15', impliedVol: 0.28, historicalVol: 0.24 },
          { date: '2025-04-29', impliedVol: 0.26, historicalVol: 0.25 },
        ];
        const placeholderSentiment = {
          index: 65, // Example: Fear & Greed Index
          status: 'Greed',
          summary: 'Market sentiment is leaning towards greed, potentially indicating overvaluation.',
        };

        await new Promise(resolve => setTimeout(resolve, 1300)); // Simulate network delay

        setRiskAssessment(placeholderRisk);
        setVolatilityAnalysis(placeholderVolatility);
        setMarketSentiment(placeholderSentiment);

      } catch (err) {
        console.error('Error fetching analytics data:', err);
        setError('Failed to load analytics data. Please ensure the backend is running.');
        setRiskAssessment(null);
        setVolatilityAnalysis([]);
        setMarketSentiment(null);
      } finally {
        setLoadingRisk(false);
        setLoadingVolatility(false);
        setLoadingSentiment(false);
      }
    };

    fetchAnalyticsData();
  }, []);

  const renderVolatilityItem = ({ item }) => (
    <View style={styles.listItem}>
      <Text>{item.date}</Text>
      <Text>Implied: {item.impliedVol.toFixed(2)}</Text>
      <Text>Historical: {item.historicalVol.toFixed(2)}</Text>
    </View>
  );

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Analytics & Insights</Text>

      {error && <Text style={styles.errorText}>{error}</Text>}

      {/* Risk Assessment Section */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Risk Assessment</Text>
        {loadingRisk ? (
          <ActivityIndicator size="small" color="#007AFF" />
        ) : riskAssessment ? (
          <View>
            <Text style={styles.metricText}>Overall Score: {riskAssessment.overallScore}</Text>
            {riskAssessment.factors.map((factor, index) => (
              <Text key={index} style={styles.factorText}>{factor.name}: {factor.level}</Text>
            ))}
            <Text style={styles.recommendationText}>Recommendation: {riskAssessment.recommendation}</Text>
          </View>
        ) : (
          <Text>Could not load risk assessment.</Text>
        )}
      </View>

      {/* Volatility Analysis Section */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Volatility Analysis (AAPL - 1M)</Text>
        {loadingVolatility ? (
          <ActivityIndicator size="large" color="#007AFF" />
        ) : (
          <FlatList
            data={volatilityAnalysis}
            renderItem={renderVolatilityItem}
            keyExtractor={(item) => item.date}
            ListEmptyComponent={<Text>No volatility data available.</Text>}
          />
        )}
      </View>

      {/* Market Sentiment Section */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Market Sentiment</Text>
        {loadingSentiment ? (
          <ActivityIndicator size="small" color="#007AFF" />
        ) : marketSentiment ? (
          <View>
            <Text style={styles.metricText}>Index: {marketSentiment.index} ({marketSentiment.status})</Text>
            <Text style={styles.summaryText}>{marketSentiment.summary}</Text>
          </View>
        ) : (
          <Text>Could not load market sentiment.</Text>
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
  metricText: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 5,
  },
  factorText: {
    fontSize: 15,
    marginLeft: 10,
    marginBottom: 3,
  },
  recommendationText: {
    fontSize: 15,
    fontStyle: 'italic',
    marginTop: 10,
    color: '#3A3A3C',
  },
  summaryText: {
    fontSize: 15,
    marginTop: 5,
    color: '#3A3A3C',
  },
  listItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
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

export default AnalyticsScreen;

