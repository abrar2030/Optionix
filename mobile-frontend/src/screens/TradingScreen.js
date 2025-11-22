import React, { useState, useEffect } from 'react';
import { StyleSheet, ScrollView, View, FlatList, Alert } from 'react-native'; // Added Alert
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
} from 'react-native-paper'; // Added RadioButton, HelperText
// Removed unused imports: import { marketService, tradingService } from '../services/api';

const TradingScreen = () => {
    const [symbol, setSymbol] = useState('AAPL');
    const [expiry, setExpiry] = useState('2025-12-19');
    const [optionChain, setOptionChain] = useState([]);
    const [orderBook, setOrderBook] = useState({ bids: [], asks: [] });
    const [loadingChain, setLoadingChain] = useState(false);
    const [loadingBook, setLoadingBook] = useState(false);
    const [error, setError] = useState(null);
    const theme = useTheme();

    // Order Form State
    const [orderSide, setOrderSide] = useState('Buy'); // 'Buy' or 'Sell'
    const [orderType, setOrderType] = useState('Market'); // 'Market' or 'Limit'
    const [quantity, setQuantity] = useState('1');
    const [limitPrice, setLimitPrice] = useState('');
    const [selectedOptionSymbol, setSelectedOptionSymbol] = useState(''); // Placeholder for selected option
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Fetch Option Chain and Order Book data
    const fetchData = async () => {
        setLoadingChain(true);
        setLoadingBook(true);
        setError(null);
        try {
            // Using placeholder data
            const placeholderChain = [
                { id: 'C170', type: 'Call', strike: 170, price: 10.5, volume: 1500 },
                { id: 'C175', type: 'Call', strike: 175, price: 7.2, volume: 2200 },
                { id: 'C180', type: 'Call', strike: 180, price: 4.8, volume: 1800 },
                { id: 'P170', type: 'Put', strike: 170, price: 5.5, volume: 1200 },
                { id: 'P165', type: 'Put', strike: 165, price: 3.1, volume: 1900 },
                { id: 'P160', type: 'Put', strike: 160, price: 1.9, volume: 1600 },
            ];
            const placeholderBook = {
                bids: [
                    { price: 174.9, size: 100 },
                    { price: 174.85, size: 200 },
                ],
                asks: [
                    { price: 175.05, size: 150 },
                    { price: 175.1, size: 250 },
                ],
            };
            await new Promise((resolve) => setTimeout(resolve, 1200)); // Simulate network delay
            setOptionChain(placeholderChain);
            setOrderBook(placeholderBook);
            // Auto-select first option for the form initially
            if (placeholderChain.length > 0) {
                setSelectedOptionSymbol(
                    `${symbol} ${expiry} ${placeholderChain[0].type} ${placeholderChain[0].strike}`,
                );
            }
        } catch (err) {
            console.error('Error fetching trading data:', err);
            setError('Failed to load trading data. Please try again later.');
            setOptionChain([]);
            setOrderBook({ bids: [], asks: [] });
        } finally {
            setLoadingChain(false);
            setLoadingBook(false);
        }
    };

    useEffect(() => {
        fetchData();
    }, []);

    const handlePlaceOrder = async () => {
        // Basic validation
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
            // Placeholder for actual order submission logic
            console.log('Submitting Order:', {
                symbol: selectedOptionSymbol,
                side: orderSide,
                type: orderType,
                quantity: parseInt(quantity),
                limitPrice: orderType === 'Limit' ? parseFloat(limitPrice) : undefined,
            });
            // Simulate API call
            await new Promise((resolve) => setTimeout(resolve, 1500));
            Alert.alert('Success', 'Order placed successfully (Simulated).');
            // Reset form potentially
            // setQuantity('1');
            // setLimitPrice('');
        } catch (err) {
            console.error('Error placing order:', err);
            setError('Failed to place order. Please try again.');
            Alert.alert('Error', 'Failed to place order. Please try again.');
        } finally {
            setIsSubmitting(false);
        }
    };

    const renderOptionItem = ({ item }) => (
        // Added onPress to select option for trading form
        <DataTable.Row
            onPress={() =>
                setSelectedOptionSymbol(`${symbol} ${expiry} ${item.type} ${item.strike}`)
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
            <PaperText style={{ color: theme.colors.primary }}>
                Price: ${item.price.toFixed(2)}
            </PaperText>
            <PaperText>Size: {item.size}</PaperText>
        </View>
    );

    return (
        <ScrollView style={[styles.container, { backgroundColor: theme.colors.background }]}>
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

            {error && <Paragraph style={styles.errorText}>{error}</Paragraph>}

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
                                keyExtractor={(item) => item.id} // Use unique ID
                                ListEmptyComponent={
                                    <Paragraph style={styles.emptyListText}>
                                        No option data available.
                                    </Paragraph>
                                }
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
                        editable={false} // Make it non-editable, selection happens via table tap
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
