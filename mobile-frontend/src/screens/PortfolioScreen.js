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
import { portfolioService } from '../services/api';

const PortfolioScreen = () => {
    const [portfolioSummary, setPortfolioSummary] = useState(null);
    const [positions, setPositions] = useState([]);
    const [performance, setPerformance] = useState([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState(null);
    const [usingFallbackData, setUsingFallbackData] = useState(false);
    const theme = useTheme();

    const getFallbackData = () => ({
        summary: {
            totalValue: 150000.0,
            dayChange: '+1.5%',
            cashBalance: 25000.0,
        },
        positions: [
            { symbol: 'AAPL', quantity: 100, value: 17550.0, type: 'Stock' },
            { symbol: 'GOOGL', quantity: 10, value: 28000.0, type: 'Stock' },
            { symbol: 'BTC-USD', quantity: 0.5, value: 30000.0, type: 'Crypto' },
            { symbol: 'AAPL 251219C180', quantity: 5, value: 2400.0, type: 'Option' },
        ],
        performance: [
            { date: '2025-04-01', value: 145000 },
            { date: '2025-04-15', value: 148000 },
            { date: '2025-04-29', value: 150000 },
        ],
    });

    const fetchPortfolioData = async () => {
        setLoading(true);
        setError(null);

        try {
            const [summaryData, positionsData, performanceData] = await Promise.all([
                portfolioService.getPortfolioSummary().catch(() => null),
                portfolioService.getPositions().catch(() => null),
                portfolioService.getPerformanceHistory().catch(() => null),
            ]);

            if (summaryData && positionsData) {
                setPortfolioSummary(summaryData);
                setPositions(positionsData.positions || positionsData);
                setPerformance(performanceData?.performance || performanceData || []);
                setUsingFallbackData(false);
            } else {
                const fallback = getFallbackData();
                setPortfolioSummary(fallback.summary);
                setPositions(fallback.positions);
                setPerformance(fallback.performance);
                setUsingFallbackData(true);
                setError('Using demo data. Connect to backend for your portfolio.');
            }
        } catch (err) {
            console.error('Error fetching portfolio data:', err);
            const fallback = getFallbackData();
            setPortfolioSummary(fallback.summary);
            setPositions(fallback.positions);
            setPerformance(fallback.performance);
            setUsingFallbackData(true);
            setError('Using demo data. Please check your connection.');
        } finally {
            setLoading(false);
        }
    };

    const onRefresh = useCallback(async () => {
        setRefreshing(true);
        await fetchPortfolioData();
        setRefreshing(false);
    }, []);

    useEffect(() => {
        fetchPortfolioData();
    }, []);

    const renderPositionItem = ({ item }) => (
        <List.Item
            title={`${item.symbol} (${item.type})`}
            description={`Qty: ${item.quantity}`}
            right={() => (
                <PaperText style={styles.listItemValue}>${item.value.toFixed(2)}</PaperText>
            )}
            titleStyle={styles.listItemTitle}
        />
    );

    const renderPerformanceItem = ({ item }) => (
        <List.Item
            title={item.date}
            right={() => (
                <PaperText style={styles.listItemValue}>${item.value.toFixed(2)}</PaperText>
            )}
            titleStyle={styles.listItemTitle}
        />
    );

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
            <Title style={styles.title}>Portfolio Management</Title>

            {error && (
                <Card style={[styles.card, styles.warningCard]}>
                    <Card.Content>
                        <Paragraph style={styles.warningText}>{error}</Paragraph>
                    </Card.Content>
                </Card>
            )}

            <Card style={styles.card}>
                <Card.Title title="Summary" />
                <Card.Content>
                    {loading && !portfolioSummary ? (
                        <ActivityIndicator
                            animating={true}
                            size="small"
                            style={styles.loadingIndicator}
                        />
                    ) : portfolioSummary ? (
                        <View>
                            <Paragraph style={styles.summaryText}>
                                Total Value: ${portfolioSummary.totalValue?.toFixed(2) || '0.00'}
                            </Paragraph>
                            <View style={styles.summaryChangeContainer}>
                                <Paragraph style={styles.summaryText}>Day's Change: </Paragraph>
                                {renderChange(portfolioSummary.dayChange || 0)}
                            </View>
                            <Paragraph style={styles.summaryText}>
                                Cash Balance: ${portfolioSummary.cashBalance?.toFixed(2) || '0.00'}
                            </Paragraph>
                        </View>
                    ) : (
                        <Paragraph>Could not load summary.</Paragraph>
                    )}
                </Card.Content>
            </Card>

            <Card style={styles.card}>
                <Card.Title title="Positions" />
                <Card.Content>
                    <FlatList
                        data={positions}
                        renderItem={renderPositionItem}
                        keyExtractor={(item, index) => `${item.symbol}-${index}`}
                        ItemSeparatorComponent={() => <Divider />}
                        ListEmptyComponent={
                            <Paragraph style={styles.emptyListText}>No positions found.</Paragraph>
                        }
                        scrollEnabled={false}
                    />
                </Card.Content>
            </Card>

            <Card style={styles.card}>
                <Card.Title title="Performance (Last Month)" />
                <Card.Content>
                    <FlatList
                        data={performance}
                        renderItem={renderPerformanceItem}
                        keyExtractor={(item) => item.date}
                        ItemSeparatorComponent={() => <Divider />}
                        ListEmptyComponent={
                            <Paragraph style={styles.emptyListText}>
                                No performance data available.
                            </Paragraph>
                        }
                        scrollEnabled={false}
                    />
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
    emptyListText: {
        textAlign: 'center',
        marginVertical: 10,
        fontStyle: 'italic',
    },
});

export default PortfolioScreen;
