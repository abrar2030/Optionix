import React, { useState, useEffect, useCallback } from 'react';
import { StyleSheet, ScrollView, View, FlatList, Alert, RefreshControl } from 'react-native';
import {
    ActivityIndicator,
    Card,
    Title,
    Paragraph,
    TextInput,
    Button,
    DataTable,
    Text as PaperText,
    useTheme,
    Divider,
    RadioButton,
    HelperText,
} from 'react-native-paper';
import { marketService, tradingService } from '../services/api';

const TradingScreen = () => {
    const [symbol, setSymbol] = useState('AAPL');
    const [expiry, setExpiry] = useState('2025-12-19');
    const [optionChain, setOptionChain] = useState([]);
    const [orderBook, setOrderBook] = useState({ bids: [], asks: [] });
    const [loadingChain, setLoadingChain] = useState(false);
    const [loadingBook, setLoadingBook] = useState(false);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState(null);
    const [usingFallbackData, setUsingFallbackData] = useState(false);
    const theme = useTheme();

    // Order Form State
    const [orderSide, setOrderSide] = useState('Buy');
    const [orderType, setOrderType] = useState('Market');
    const [quantity, setQuantity] = useState('1');
    const [limitPrice, setLimitPrice] = useState('');
    const [selectedOptionSymbol, setSelectedOptionSymbol] = useState('');
    const [selectedOption, setSelectedOption] = useState(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const getFallbackChain = () => [
        { id: 'C170', type: 'Call', strike: 170, price: 10.5, volume: 1500, delta: 0.75 },
        { id: 'C175', type: 'Call', strike: 175, price: 7.2, volume: 2200, delta: 0.65 },
        { id: 'C180', type: 'Call', strike: 180, price: 4.8, volume: 1800, delta: 0.5 },
        { id: 'P170', type: 'Put', strike: 170, price: 5.5, volume: 1200, delta: -0.25 },
        { id: 'P165', type: 'Put', strike: 165, price: 3.1, volume: 1900, delta: -0.35 },
        { id: 'P160', type: 'Put', strike: 160, price: 1.9, volume: 1600, delta: -0.5 },
    ];

    const getFallbackOrderBook = () => ({
        bids: [
            { price: 174.9, size: 100 },
            { price: 174.85, size: 200 },
            { price: 174.8, size: 150 },
        ],
        asks: [
            { price: 175.05, size: 150 },
            { price: 175.1, size: 250 },
            { price: 175.15, size: 100 },
        ],
    });

    const fetchData = async () => {
        setLoadingChain(true);
        setLoadingBook(true);
        setError(null);

        try {
            // Try to fetch from API
            const [chainData, bookData] = await Promise.all([
                marketService.getOptionChain(symbol, expiry).catch(() => null),
                marketService.getOrderBook(symbol).catch(() => null),
            ]);

            if (chainData && chainData.options) {
                setOptionChain(chainData.options);
                setUsingFallbackData(false);

                // Auto-select first option
                if (chainData.options.length > 0) {
                    const firstOption = chainData.options[0];
                    setSelectedOption(firstOption);
                    setSelectedOptionSymbol(
                        `${symbol} ${expiry} ${firstOption.type} ${firstOption.strike}`,
                    );
                }
            } else {
                // Use fallback data
                const fallbackChain = getFallbackChain();
                setOptionChain(fallbackChain);
                setUsingFallbackData(true);

                if (fallbackChain.length > 0) {
                    setSelectedOption(fallbackChain[0]);
                    setSelectedOptionSymbol(
                        `${symbol} ${expiry} ${fallbackChain[0].type} ${fallbackChain[0].strike}`,
                    );
                }
            }

            if (bookData && (bookData.bids || bookData.asks)) {
                setOrderBook({
                    bids: bookData.bids || [],
                    asks: bookData.asks || [],
                });
            } else {
                setOrderBook(getFallbackOrderBook());
            }

            if (!chainData || !bookData) {
                setError('Using demo data. Connect to backend for live trading data.');
            }
        } catch (err) {
            console.error('Error fetching trading data:', err);
            setOptionChain(getFallbackChain());
            setOrderBook(getFallbackOrderBook());
            setUsingFallbackData(true);
            setError('Using demo data. Please check your connection.');
        } finally {
            setLoadingChain(false);
            setLoadingBook(false);
        }
    };

    const onRefresh = useCallback(async () => {
        setRefreshing(true);
        await fetchData();
        setRefreshing(false);
    }, [symbol, expiry]);

    useEffect(() => {
        fetchData();
    }, []);

    const handlePlaceOrder = async () => {
        // Validation
        if (!selectedOptionSymbol) {
            Alert.alert('Error', 'Please select an option contract.');
            return;
        }
        if (!quantity || parseInt(quantity) <= 0) {
            Alert.alert('Error', 'Please enter a valid quantity.');
            return;
        }
        if (orderType === 'Limit' && (!limitPrice || parseFloat(limitPrice) <= 0)) {
            Alert.alert('Error', 'Please enter a valid limit price for a limit order.');
            return;
        }

        setIsSubmitting(true);
        setError(null);

        try {
            const orderData = {
                symbol: selectedOptionSymbol,
                option_type: selectedOption?.type,
                strike: selectedOption?.strike,
                expiry,
                side: orderSide.toLowerCase(),
                order_type: orderType.toLowerCase(),
                quantity: parseInt(quantity),
                ...(orderType === 'Limit' && { limit_price: parseFloat(limitPrice) }),
            };

            if (!usingFallbackData) {
                // Try real API call
                try {
                    const result = await tradingService.executeTrade(orderData);
                    Alert.alert('Success', result.message || 'Order placed successfully.');
                } catch (apiError) {
                    console.log('API call failed, showing simulation message:', apiError);
                    Alert.alert(
                        'Demo Mode',
                        'Order simulation successful. Connect to backend for live trading.',
                    );
                }
            } else {
                // Simulated order
                await new Promise((resolve) => setTimeout(resolve, 1500));
                Alert.alert(
                    'Demo Mode',
                    'Order simulation successful. Connect to backend for live trading.',
                );
            }
        } catch (err) {
            console.error('Error placing order:', err);
            Alert.alert('Error', 'Failed to place order. Please try again.');
        } finally {
            setIsSubmitting(false);
        }
    };

    const renderOptionItem = ({ item }) => (
        <DataTable.Row
            onPress={() => {
                setSelectedOption(item);
                setSelectedOptionSymbol(`${symbol} ${expiry} ${item.type} ${item.strike}`);
            }}
            style={
                selectedOption?.id === item.id
                    ? { backgroundColor: theme.colors.primaryContainer }
                    : {}
            }
        >
            <DataTable.Cell>{item.type}</DataTable.Cell>
            <DataTable.Cell numeric>{item.strike}</DataTable.Cell>
            <DataTable.Cell numeric>${item.price.toFixed(2)}</DataTable.Cell>
            <DataTable.Cell numeric>{item.volume}</DataTable.Cell>
        </DataTable.Row>
    );

    const renderOrderItem = ({ item }) => (
        <View style={styles.orderItem}>
            <PaperText style={{ color: theme.colors.primary }}>${item.price.toFixed(2)}</PaperText>
            <PaperText>{item.size}</PaperText>
        </View>
    );

    return (
        <ScrollView
            style={[styles.container, { backgroundColor: theme.colors.background }]}
            refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        >
            <Title style={styles.title}>Trading Interface</Title>

            <Card style={styles.card}>
                <Card.Content>
                    <TextInput
                        label="Symbol (e.g., AAPL)"
                        value={symbol}
                        onChangeText={setSymbol}
                        autoCapitalize="characters"
                        style={styles.input}
                        mode="outlined"
                    />
                    <TextInput
                        label="Expiry (YYYY-MM-DD)"
                        value={expiry}
                        onChangeText={setExpiry}
                        style={styles.input}
                        mode="outlined"
                    />
                    <Button
                        mode="contained"
                        onPress={fetchData}
                        style={styles.button}
                        loading={loadingChain || loadingBook}
                    >
                        Load Data
                    </Button>
                </Card.Content>
            </Card>

            {error && (
                <Card style={[styles.card, styles.warningCard]}>
                    <Card.Content>
                        <Paragraph style={styles.warningText}>{error}</Paragraph>
                    </Card.Content>
                </Card>
            )}

            {/* Option Chain Section */}
            <Card style={styles.card}>
                <Card.Title
                    title={`Option Chain (${symbol} - ${expiry})`}
                    subtitle="Tap row to select for trading"
                />
                <Card.Content>
                    {loadingChain ? (
                        <ActivityIndicator
                            animating={true}
                            size="large"
                            style={styles.loadingIndicator}
                        />
                    ) : (
                        <DataTable>
                            <DataTable.Header>
                                <DataTable.Title>Type</DataTable.Title>
                                <DataTable.Title numeric>Strike</DataTable.Title>
                                <DataTable.Title numeric>Price</DataTable.Title>
                                <DataTable.Title numeric>Volume</DataTable.Title>
                            </DataTable.Header>
                            <FlatList
                                data={optionChain}
                                renderItem={renderOptionItem}
                                keyExtractor={(item) => item.id || `${item.type}-${item.strike}`}
                                ListEmptyComponent={
                                    <Paragraph style={styles.emptyListText}>
                                        No option data available.
                                    </Paragraph>
                                }
                                scrollEnabled={false}
                            />
                        </DataTable>
                    )}
                </Card.Content>
            </Card>

            {/* Order Book Section */}
            <Card style={styles.card}>
                <Card.Title title={`Order Book (${symbol})`} />
                <Card.Content>
                    {loadingBook ? (
                        <ActivityIndicator
                            animating={true}
                            size="large"
                            style={styles.loadingIndicator}
                        />
                    ) : (
                        <View style={styles.orderBookContainer}>
                            <View style={styles.orderBookSide}>
                                <Title style={[styles.orderBookTitle, { color: 'green' }]}>
                                    Bids
                                </Title>
                                <FlatList
                                    data={orderBook.bids}
                                    renderItem={renderOrderItem}
                                    keyExtractor={(item, index) => `bid-${item.price}-${index}`}
                                    ListEmptyComponent={
                                        <Paragraph style={styles.emptyListText}>No bids.</Paragraph>
                                    }
                                    ItemSeparatorComponent={() => <Divider />}
                                    scrollEnabled={false}
                                />
                            </View>
                            <View style={styles.orderBookSide}>
                                <Title
                                    style={[styles.orderBookTitle, { color: theme.colors.error }]}
                                >
                                    Asks
                                </Title>
                                <FlatList
                                    data={orderBook.asks}
                                    renderItem={renderOrderItem}
                                    keyExtractor={(item, index) => `ask-${item.price}-${index}`}
                                    ListEmptyComponent={
                                        <Paragraph style={styles.emptyListText}>No asks.</Paragraph>
                                    }
                                    ItemSeparatorComponent={() => <Divider />}
                                    scrollEnabled={false}
                                />
                            </View>
                        </View>
                    )}
                </Card.Content>
            </Card>

            {/* Trading Form */}
            <Card style={styles.card}>
                <Card.Title title="Place Order" />
                <Card.Content>
                    <TextInput
                        label="Selected Option"
                        value={selectedOptionSymbol}
                        editable={false}
                        style={styles.input}
                        mode="outlined"
                    />

                    <Paragraph style={styles.radioLabel}>Side:</Paragraph>
                    <RadioButton.Group
                        onValueChange={(newValue) => setOrderSide(newValue)}
                        value={orderSide}
                    >
                        <View style={styles.radioRow}>
                            <View style={styles.radioItem}>
                                <RadioButton value="Buy" />
                                <PaperText>Buy</PaperText>
                            </View>
                            <View style={styles.radioItem}>
                                <RadioButton value="Sell" />
                                <PaperText>Sell</PaperText>
                            </View>
                        </View>
                    </RadioButton.Group>

                    <Paragraph style={styles.radioLabel}>Order Type:</Paragraph>
                    <RadioButton.Group
                        onValueChange={(newValue) => setOrderType(newValue)}
                        value={orderType}
                    >
                        <View style={styles.radioRow}>
                            <View style={styles.radioItem}>
                                <RadioButton value="Market" />
                                <PaperText>Market</PaperText>
                            </View>
                            <View style={styles.radioItem}>
                                <RadioButton value="Limit" />
                                <PaperText>Limit</PaperText>
                            </View>
                        </View>
                    </RadioButton.Group>

                    <TextInput
                        label="Quantity"
                        value={quantity}
                        onChangeText={setQuantity}
                        style={styles.input}
                        mode="outlined"
                        keyboardType="numeric"
                    />
                    <HelperText type="info">Enter the number of contracts.</HelperText>

                    {orderType === 'Limit' && (
                        <TextInput
                            label="Limit Price"
                            value={limitPrice}
                            onChangeText={setLimitPrice}
                            style={styles.input}
                            mode="outlined"
                            keyboardType="numeric"
                        />
                    )}

                    <Button
                        mode="contained"
                        onPress={handlePlaceOrder}
                        style={styles.button}
                        loading={isSubmitting}
                        disabled={isSubmitting || !selectedOptionSymbol}
                    >
                        {isSubmitting ? 'Submitting...' : 'Submit Order'}
                    </Button>
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
    orderBookContainer: {
        flexDirection: 'row',
        justifyContent: 'space-between',
    },
    orderBookSide: {
        flex: 1,
        marginHorizontal: 5,
    },
    orderBookTitle: {
        fontSize: 16,
        fontWeight: 'bold',
        marginBottom: 10,
        textAlign: 'center',
    },
    orderItem: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        paddingVertical: 8,
    },
    emptyListText: {
        textAlign: 'center',
        marginVertical: 10,
        fontStyle: 'italic',
    },
    radioLabel: {
        marginTop: 10,
        marginBottom: 5,
        fontSize: 14,
        color: 'grey',
    },
    radioRow: {
        flexDirection: 'row',
        marginBottom: 10,
    },
    radioItem: {
        flexDirection: 'row',
        alignItems: 'center',
        marginRight: 20,
    },
});

export default TradingScreen;
