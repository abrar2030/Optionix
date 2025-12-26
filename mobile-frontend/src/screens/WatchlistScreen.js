import React, { useState, useEffect, useCallback } from 'react';
import { StyleSheet, ScrollView, View, FlatList, Alert, RefreshControl } from 'react-native';
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
    TextInput,
    IconButton,
} from 'react-native-paper';
import { watchlistService } from '../services/api';

const WatchlistScreen = () => {
    const [watchlist, setWatchlist] = useState([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState(null);
    const [newSymbol, setNewSymbol] = useState('');
    const [addingSymbol, setAddingSymbol] = useState(false);
    const theme = useTheme();

    const getInitialWatchlist = () => [
        { symbol: 'MSFT', price: 420.55, change: '+1.1%' },
        { symbol: 'NVDA', price: 950.02, change: '-0.5%' },
        { symbol: 'AMZN', price: 180.3, change: '+0.8%' },
    ];

    const fetchWatchlist = async () => {
        setLoading(true);
        setError(null);

        try {
            const data = await watchlistService.getWatchlist();
            if (data && data.symbols) {
                setWatchlist(data.symbols);
            } else {
                setWatchlist(getInitialWatchlist());
                setError('Using demo watchlist. Connect to backend to save your list.');
            }
        } catch (err) {
            console.error('Error fetching watchlist:', err);
            setWatchlist(getInitialWatchlist());
            setError('Using demo watchlist. Connect to backend to save your list.');
        } finally {
            setLoading(false);
        }
    };

    const onRefresh = useCallback(async () => {
        setRefreshing(true);
        await fetchWatchlist();
        setRefreshing(false);
    }, []);

    useEffect(() => {
        fetchWatchlist();
    }, []);

    const addSymbolToWatchlist = async () => {
        if (newSymbol.trim() === '') return;

        setAddingSymbol(true);
        try {
            await watchlistService.addToWatchlist(newSymbol.toUpperCase());
            await fetchWatchlist();
            setNewSymbol('');
        } catch (err) {
            console.log('API add failed, adding locally:', err);
            const newEntry = {
                symbol: newSymbol.toUpperCase(),
                price: parseFloat((Math.random() * 1000).toFixed(2)),
                change: `${(Math.random() * 2 - 1).toFixed(1)}%`,
            };
            setWatchlist([...watchlist, newEntry]);
            setNewSymbol('');
        } finally {
            setAddingSymbol(false);
        }
    };

    const removeSymbol = async (symbolToRemove) => {
        Alert.alert('Remove Symbol', `Remove ${symbolToRemove} from watchlist?`, [
            { text: 'Cancel', style: 'cancel' },
            {
                text: 'Remove',
                style: 'destructive',
                onPress: async () => {
                    try {
                        await watchlistService.removeFromWatchlist(symbolToRemove);
                        setWatchlist(watchlist.filter((item) => item.symbol !== symbolToRemove));
                    } catch (err) {
                        console.log('API remove failed, removing locally:', err);
                        setWatchlist(watchlist.filter((item) => item.symbol !== symbolToRemove));
                    }
                },
            },
        ]);
    };

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

    const renderWatchlistItem = ({ item }) => (
        <List.Item
            title={`${item.symbol}: $${item.price}`}
            right={() => (
                <View style={styles.itemRight}>
                    {renderChange(item.change)}
                    <IconButton icon="delete" size={20} onPress={() => removeSymbol(item.symbol)} />
                </View>
            )}
            titleStyle={styles.listItemTitle}
        />
    );

    return (
        <ScrollView
            style={[styles.container, { backgroundColor: theme.colors.background }]}
            refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        >
            <Title style={styles.title}>My Watchlist</Title>

            {error && (
                <Card style={[styles.card, styles.warningCard]}>
                    <Card.Content>
                        <Paragraph style={styles.warningText}>{error}</Paragraph>
                    </Card.Content>
                </Card>
            )}

            <Card style={styles.card}>
                <Card.Title title="Add Symbol" />
                <Card.Content>
                    <TextInput
                        label="Enter Symbol (e.g., GOOG)"
                        value={newSymbol}
                        onChangeText={setNewSymbol}
                        autoCapitalize="characters"
                        style={styles.input}
                        mode="outlined"
                        disabled={addingSymbol}
                    />
                    <Button
                        mode="contained"
                        onPress={addSymbolToWatchlist}
                        style={styles.button}
                        loading={addingSymbol}
                        disabled={addingSymbol || !newSymbol.trim()}
                    >
                        Add to Watchlist
                    </Button>
                </Card.Content>
            </Card>

            <Card style={styles.card}>
                <Card.Title title="Tracked Symbols" />
                <Card.Content>
                    {loading && !watchlist.length ? (
                        <ActivityIndicator
                            animating={true}
                            size="large"
                            style={styles.loadingIndicator}
                        />
                    ) : (
                        <FlatList
                            data={watchlist}
                            renderItem={renderWatchlistItem}
                            keyExtractor={(item) => item.symbol}
                            ItemSeparatorComponent={() => <Divider />}
                            ListEmptyComponent={
                                <Paragraph style={styles.emptyListText}>
                                    Your watchlist is empty. Add symbols above.
                                </Paragraph>
                            }
                            scrollEnabled={false}
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
    warningCard: {
        backgroundColor: '#FFF3CD',
        borderLeftWidth: 4,
        borderLeftColor: '#FFA500',
    },
    warningText: {
        color: '#856404',
    },
    input: {
        marginBottom: 10,
    },
    button: {
        marginTop: 10,
    },
    loadingIndicator: {
        marginVertical: 20,
    },
    listItemTitle: {
        fontSize: 16,
    },
    itemRight: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    positiveChange: {
        color: '#34C759',
        fontWeight: 'bold',
        fontSize: 16,
        marginRight: 8,
    },
    negativeChange: {
        color: '#FF3B30',
        fontWeight: 'bold',
        fontSize: 16,
        marginRight: 8,
    },
    emptyListText: {
        textAlign: 'center',
        marginVertical: 10,
        fontStyle: 'italic',
    },
});

export default WatchlistScreen;
