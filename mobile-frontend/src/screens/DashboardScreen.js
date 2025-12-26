import React, { useState, useEffect, useCallback } from 'react';
import { StyleSheet, ScrollView, View, RefreshControl } from 'react-native';
import {
    ActivityIndicator,
    Card,
    Title,
    Paragraph,
    List,
    Divider,
    Text as PaperText,
    useTheme,
    Button,
} from 'react-native-paper';
import { marketService } from '../services/api';

const DashboardScreen = () => {
    const [marketOverview, setMarketOverview] = useState(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState(null);
    const [usingFallbackData, setUsingFallbackData] = useState(false);
    const theme = useTheme();

    const getFallbackData = () => ({
        marketStatus: 'Open',
        majorIndices: [
            { name: 'S&P 500', value: 4500.5, change: '+0.5%' },
            { name: 'Dow Jones', value: 35000.75, change: '+0.3%' },
            { name: 'NASDAQ', value: 14000.25, change: '+0.8%' },
        ],
        topMovers: [
            { symbol: 'AAPL', price: 175.5, change: '+1.2%' },
            { symbol: 'GOOGL', price: 2800.0, change: '+0.9%' },
            { symbol: 'TSLA', price: 700.0, change: '-0.5%' },
        ],
    });

    const fetchMarketData = async () => {
        try {
            setLoading(true);
            setError(null);

            // Try to fetch from API
            const data = await marketService.getMarketOverview();
            setMarketOverview(data);
            setUsingFallbackData(false);
        } catch (err) {
            console.error('Error fetching market overview:', err);

            // Use fallback data when API fails
            setMarketOverview(getFallbackData());
            setUsingFallbackData(true);
            setError('Using demo data. Check your connection to see live market data.');
        } finally {
            setLoading(false);
        }
    };

    const onRefresh = useCallback(async () => {
        setRefreshing(true);
        await fetchMarketData();
        setRefreshing(false);
    }, []);

    useEffect(() => {
        fetchMarketData();

        // Refresh data every 30 seconds if not using fallback
        const interval = setInterval(() => {
            if (!usingFallbackData) {
                fetchMarketData();
            }
        }, 30000);

        return () => clearInterval(interval);
    }, [usingFallbackData]);

    const renderChange = (change) => {
        const changeStr =
            typeof change === 'string' ? change : `${change > 0 ? '+' : ''}${change}%`;
        const isPositive = changeStr.startsWith('+');
        return (
            <PaperText style={isPositive ? styles.positiveChange : styles.negativeChange}>
                {changeStr}
            </PaperText>
        );
    };

    return (
        <ScrollView
            style={[styles.container, { backgroundColor: theme.colors.background }]}
            refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        >
            <Title style={styles.title}>Market Dashboard</Title>

            {usingFallbackData && (
                <Card style={[styles.card, styles.warningCard]}>
                    <Card.Content>
                        <Paragraph style={styles.warningText}>{error}</Paragraph>
                        <Button
                            mode="outlined"
                            onPress={fetchMarketData}
                            style={styles.retryButton}
                            compact
                        >
                            Retry Connection
                        </Button>
                    </Card.Content>
                </Card>
            )}

            {loading && !marketOverview && (
                <ActivityIndicator animating={true} size="large" style={styles.loadingIndicator} />
            )}

            {marketOverview && (
                <View>
                    <Card style={styles.card}>
                        <Card.Content>
                            <Title>
                                Market Status:{' '}
                                {marketOverview.marketStatus || marketOverview.status || 'Open'}
                            </Title>
                        </Card.Content>
                    </Card>

                    <Card style={styles.card}>
                        <Card.Title title="Major Indices" />
                        <Card.Content>
                            {(marketOverview.majorIndices || marketOverview.indices || []).map(
                                (index, i) => (
                                    <React.Fragment key={i}>
                                        <List.Item
                                            title={`${index.name || index.symbol}: ${(index.value || index.price).toFixed(2)}`}
                                            right={() =>
                                                renderChange(index.change || index.change_percent)
                                            }
                                            titleStyle={styles.listItemTitle}
                                        />
                                        {i <
                                            (
                                                marketOverview.majorIndices ||
                                                marketOverview.indices ||
                                                []
                                            ).length -
                                                1 && <Divider />}
                                    </React.Fragment>
                                ),
                            )}
                        </Card.Content>
                    </Card>

                    <Card style={styles.card}>
                        <Card.Title title="Top Movers" />
                        <Card.Content>
                            {(marketOverview.topMovers || marketOverview.movers || []).map(
                                (stock, i) => (
                                    <React.Fragment key={i}>
                                        <List.Item
                                            title={`${stock.symbol}: $${(stock.price || stock.value).toFixed(2)}`}
                                            right={() =>
                                                renderChange(stock.change || stock.change_percent)
                                            }
                                            titleStyle={styles.listItemTitle}
                                        />
                                        {i <
                                            (
                                                marketOverview.topMovers ||
                                                marketOverview.movers ||
                                                []
                                            ).length -
                                                1 && <Divider />}
                                    </React.Fragment>
                                ),
                            )}
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
        elevation: 4,
    },
    warningCard: {
        backgroundColor: '#FFF3CD',
        borderLeftWidth: 4,
        borderLeftColor: '#FFA500',
    },
    warningText: {
        color: '#856404',
        marginBottom: 10,
    },
    retryButton: {
        borderColor: '#FFA500',
    },
    listItemTitle: {
        fontSize: 16,
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
});

export default DashboardScreen;
