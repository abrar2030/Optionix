import React, { useState, useEffect, useCallback } from 'react';
import { StyleSheet, ScrollView, View, FlatList, RefreshControl } from 'react-native';
import {
    ActivityIndicator,
    Card,
    Title,
    Paragraph,
    List,
    Divider,
    Text as PaperText,
    useTheme,
} from 'react-native-paper';
import { analyticsService } from '../services/api';

const AnalyticsScreen = () => {
    const [riskAssessment, setRiskAssessment] = useState(null);
    const [volatilityAnalysis, setVolatilityAnalysis] = useState([]);
    const [marketSentiment, setMarketSentiment] = useState(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState(null);
    const theme = useTheme();

    const getFallbackData = () => ({
        risk: {
            overallScore: 75,
            factors: [
                { name: 'Market Risk', level: 'High' },
                { name: 'Credit Risk', level: 'Low' },
                { name: 'Liquidity Risk', level: 'Medium' },
            ],
            recommendation: 'Consider hedging strategies for market exposure.',
        },
        volatility: [
            { date: '2025-04-01', impliedVol: 0.25, historicalVol: 0.22 },
            { date: '2025-04-15', impliedVol: 0.28, historicalVol: 0.24 },
            { date: '2025-04-29', impliedVol: 0.26, historicalVol: 0.25 },
        ],
        sentiment: {
            index: 65,
            status: 'Greed',
            summary:
                'Market sentiment is leaning towards greed, potentially indicating overvaluation.',
        },
    });

    const fetchAnalyticsData = async () => {
        setLoading(true);
        setError(null);

        try {
            const [riskData, volatilityData, sentimentData] = await Promise.all([
                analyticsService.getRiskAssessment().catch(() => null),
                analyticsService.getVolatilityAnalysis('AAPL').catch(() => null),
                analyticsService.getMarketSentiment().catch(() => null),
            ]);

            if (riskData || volatilityData || sentimentData) {
                setRiskAssessment(riskData);
                setVolatilityAnalysis(volatilityData?.analysis || volatilityData || []);
                setMarketSentiment(sentimentData);
                if (!riskData || !volatilityData || !sentimentData) {
                    setError('Using demo data. Connect to backend for full analytics.');
                }
            } else {
                const fallback = getFallbackData();
                setRiskAssessment(fallback.risk);
                setVolatilityAnalysis(fallback.volatility);
                setMarketSentiment(fallback.sentiment);
                setError('Using demo data. Connect to backend for real analytics.');
            }
        } catch (err) {
            console.error('Error fetching analytics data:', err);
            const fallback = getFallbackData();
            setRiskAssessment(fallback.risk);
            setVolatilityAnalysis(fallback.volatility);
            setMarketSentiment(fallback.sentiment);
            setError('Using demo data. Please check your connection.');
        } finally {
            setLoading(false);
        }
    };

    const onRefresh = useCallback(async () => {
        setRefreshing(true);
        await fetchAnalyticsData();
        setRefreshing(false);
    }, []);

    useEffect(() => {
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
            case 'high':
                return theme.colors.error;
            case 'medium':
                return '#FFA500';
            case 'low':
                return '#34C759';
            default:
                return theme.colors.text;
        }
    };

    return (
        <ScrollView
            style={[styles.container, { backgroundColor: theme.colors.background }]}
            refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        >
            <Title style={styles.title}>Analytics & Insights</Title>

            {error && (
                <Card style={[styles.card, styles.warningCard]}>
                    <Card.Content>
                        <Paragraph style={styles.warningText}>{error}</Paragraph>
                    </Card.Content>
                </Card>
            )}

            <Card style={styles.card}>
                <Card.Title title="Risk Assessment" />
                <Card.Content>
                    {loading && !riskAssessment ? (
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
                                    <Paragraph style={styles.factorText}>{factor.name}: </Paragraph>
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

            <Card style={styles.card}>
                <Card.Title title="Volatility Analysis (AAPL - 1M)" />
                <Card.Content>
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
                        scrollEnabled={false}
                    />
                </Card.Content>
            </Card>

            <Card style={styles.card}>
                <Card.Title title="Market Sentiment" />
                <Card.Content>
                    {loading && !marketSentiment ? (
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
        fontWeight: 'bold',
        marginBottom: 20,
        textAlign: 'center',
    },
    card: {
        marginBottom: 20,
        elevation: 4,
    },
    warningCard: {
        backgroundColor: '#FFF3CD',
        borderLeftWidth: 4,
        borderLeftColor: '#FFA500',
    },
    warningText: {
        color: '#856404',
    },
    loadingIndicator: {
        marginVertical: 20,
    },
    metricText: {
        fontSize: 16,
        fontWeight: '600',
        marginBottom: 10,
    },
    factorContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 5,
        marginLeft: 10,
    },
    factorText: {
        fontSize: 15,
    },
    factorLevel: {
        fontSize: 15,
        fontWeight: 'bold',
        marginLeft: 5,
    },
    recommendationText: {
        fontSize: 15,
        fontStyle: 'italic',
        marginTop: 15,
    },
    summaryText: {
        fontSize: 15,
        marginTop: 5,
    },
    listItemTitle: {
        fontSize: 16,
    },
    emptyListText: {
        textAlign: 'center',
        marginVertical: 10,
        fontStyle: 'italic',
    },
});

export default AnalyticsScreen;
