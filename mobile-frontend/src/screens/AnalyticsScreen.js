import React, { useState, useEffect } from "react";
import { StyleSheet, ScrollView, View, FlatList } from "react-native";
import {
  ActivityIndicator,
  Card,
  Title,
  Paragraph,
  List,
  Divider,
  Text as PaperText,
  useTheme,
} from "react-native-paper";
// Removed unused import: import { analyticsService } from '../services/api';

const AnalyticsScreen = () => {
  const [riskAssessment, setRiskAssessment] = useState(null);
  const [volatilityAnalysis, setVolatilityAnalysis] = useState([]);
  const [marketSentiment, setMarketSentiment] = useState(null);
  const [loadingRisk, setLoadingRisk] = useState(true);
  const [loadingVolatility, setLoadingVolatility] = useState(true);
  const [loadingSentiment, setLoadingSentiment] = useState(true);
  const [error, setError] = useState(null);
  const theme = useTheme();

  useEffect(() => {
    const fetchAnalyticsData = async () => {
      setLoadingRisk(true);
      setLoadingVolatility(true);
      setLoadingSentiment(true);
      setError(null);
      try {
        // Using placeholder data
        const placeholderRisk = {
          overallScore: 75,
          factors: [
            { name: "Market Risk", level: "High" },
            { name: "Credit Risk", level: "Low" },
            { name: "Liquidity Risk", level: "Medium" },
          ],
          recommendation: "Consider hedging strategies for market exposure.",
        };
        const placeholderVolatility = [
          { date: "2025-04-01", impliedVol: 0.25, historicalVol: 0.22 },
          { date: "2025-04-15", impliedVol: 0.28, historicalVol: 0.24 },
          { date: "2025-04-29", impliedVol: 0.26, historicalVol: 0.25 },
        ];
        const placeholderSentiment = {
          index: 65, // Example: Fear & Greed Index
          status: "Greed",
          summary:
            "Market sentiment is leaning towards greed, potentially indicating overvaluation.",
        };

        await new Promise((resolve) => setTimeout(resolve, 1300)); // Simulate network delay

        setRiskAssessment(placeholderRisk);
        setVolatilityAnalysis(placeholderVolatility);
        setMarketSentiment(placeholderSentiment);
      } catch (err) {
        console.error("Error fetching analytics data:", err);
        setError("Failed to load analytics data. Please try again later.");
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
    <List.Item
      title={item.date}
      description={`Implied: ${item.impliedVol.toFixed(2)} | Historical: ${item.historicalVol.toFixed(2)}`}
      titleStyle={styles.listItemTitle}
    />
  );

  const getRiskLevelColor = (level) => {
    switch (level.toLowerCase()) {
      case "high":
        return theme.colors.error; // Red
      case "medium":
        return "#FFA500"; // Orange (adjust as needed)
      case "low":
        return "#34C759"; // Green
      default:
        return theme.colors.text;
    }
  };

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: theme.colors.background }]}
    >
      <Title style={styles.title}>Analytics & Insights</Title>

      {error && <Paragraph style={styles.errorText}>{error}</Paragraph>}

      {/* Risk Assessment Section */}
      <Card style={styles.card}>
        <Card.Title title="Risk Assessment" />
        <Card.Content>
          {loadingRisk ? (
            <ActivityIndicator
              animating={true}
              size="small"
              style={styles.loadingIndicator}
            />
          ) : riskAssessment ? (
            <View>
              <Paragraph style={styles.metricText}>
                Overall Score: {riskAssessment.overallScore}
              </Paragraph>
              {riskAssessment.factors.map((factor, index) => (
                <View key={index} style={styles.factorContainer}>
                  <Paragraph style={styles.factorText}>
                    {factor.name}:{" "}
                  </Paragraph>
                  <PaperText
                    style={[
                      styles.factorLevel,
                      { color: getRiskLevelColor(factor.level) },
                    ]}
                  >
                    {factor.level}
                  </PaperText>
                </View>
              ))}
              <Paragraph style={styles.recommendationText}>
                Recommendation: {riskAssessment.recommendation}
              </Paragraph>
            </View>
          ) : (
            <Paragraph>Could not load risk assessment.</Paragraph>
          )}
        </Card.Content>
      </Card>

      {/* Volatility Analysis Section */}
      <Card style={styles.card}>
        <Card.Title title="Volatility Analysis (AAPL - 1M)" />
        <Card.Content>
          {loadingVolatility ? (
            <ActivityIndicator
              animating={true}
              size="large"
              style={styles.loadingIndicator}
            />
          ) : (
            <FlatList
              data={volatilityAnalysis}
              renderItem={renderVolatilityItem}
              keyExtractor={(item) => item.date}
              ItemSeparatorComponent={() => <Divider />}
              ListEmptyComponent={
                <Paragraph style={styles.emptyListText}>
                  No volatility data available.
                </Paragraph>
              }
            />
          )}
        </Card.Content>
      </Card>

      {/* Market Sentiment Section */}
      <Card style={styles.card}>
        <Card.Title title="Market Sentiment" />
        <Card.Content>
          {loadingSentiment ? (
            <ActivityIndicator
              animating={true}
              size="small"
              style={styles.loadingIndicator}
            />
          ) : marketSentiment ? (
            <View>
              <Paragraph style={styles.metricText}>
                Index: {marketSentiment.index} ({marketSentiment.status})
              </Paragraph>
              <Paragraph style={styles.summaryText}>
                {marketSentiment.summary}
              </Paragraph>
            </View>
          ) : (
            <Paragraph>Could not load market sentiment.</Paragraph>
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
    fontWeight: "bold",
    marginBottom: 20,
    textAlign: "center",
  },
  card: {
    marginBottom: 20,
    elevation: 4,
  },
  loadingIndicator: {
    marginVertical: 20,
  },
  metricText: {
    fontSize: 16,
    fontWeight: "600",
    marginBottom: 10,
  },
  factorContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 5,
    marginLeft: 10,
  },
  factorText: {
    fontSize: 15,
  },
  factorLevel: {
    fontSize: 15,
    fontWeight: "bold",
    marginLeft: 5,
  },
  recommendationText: {
    fontSize: 15,
    fontStyle: "italic",
    marginTop: 15,
  },
  summaryText: {
    fontSize: 15,
    marginTop: 5,
  },
  listItemTitle: {
    fontSize: 16,
  },
  errorText: {
    color: "#FF3B30",
    textAlign: "center",
    marginVertical: 10,
    fontSize: 16,
    padding: 10,
  },
  emptyListText: {
    textAlign: "center",
    marginVertical: 10,
    fontStyle: "italic",
  },
});

export default AnalyticsScreen;
