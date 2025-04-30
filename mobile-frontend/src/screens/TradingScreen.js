import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TextInput, Button, ActivityIndicator, FlatList } from 'react-native';
import { marketService, tradingService } from '../services/api'; // Corrected import path

const TradingScreen = () => {
  const [symbol, setSymbol] = useState('AAPL'); // Default symbol
  const [expiry, setExpiry] = useState('2025-12-19'); // Default expiry
  const [optionChain, setOptionChain] = useState([]);
  const [orderBook, setOrderBook] = useState({ bids: [], asks: [] });
  const [loadingChain, setLoadingChain] = useState(false);
  const [loadingBook, setLoadingBook] = useState(false);
  const [error, setError] = useState(null);

  // Fetch Option Chain and Order Book data
  const fetchData = async () => {
    setLoadingChain(true);
    setLoadingBook(true);
    setError(null);
    try {
      // Using placeholder data as backend might not be running
      // const chainData = await marketService.getOptionChain(symbol, expiry);
      // const bookData = await marketService.getOrderBook(symbol);
      const placeholderChain = [
        { type: 'Call', strike: 170, price: 10.50, volume: 1500 },
        { type: 'Call', strike: 175, price: 7.20, volume: 2200 },
        { type: 'Call', strike: 180, price: 4.80, volume: 1800 },
        { type: 'Put', strike: 170, price: 5.50, volume: 1200 },
        { type: 'Put', strike: 165, price: 3.10, volume: 1900 },
        { type: 'Put', strike: 160, price: 1.90, volume: 1600 },
      ];
      const placeholderBook = {
        bids: [{ price: 174.90, size: 100 }, { price: 174.85, size: 200 }],
        asks: [{ price: 175.05, size: 150 }, { price: 175.10, size: 250 }],
      };
      await new Promise(resolve => setTimeout(resolve, 1200)); // Simulate network delay
      setOptionChain(placeholderChain);
      setOrderBook(placeholderBook);
    } catch (err) {
      console.error('Error fetching trading data:', err);
      setError('Failed to load trading data. Please check symbol/expiry and ensure backend is running.');
      setOptionChain([]);
      setOrderBook({ bids: [], asks: [] });
    } finally {
      setLoadingChain(false);
      setLoadingBook(false);
    }
  };

  useEffect(() => {
    fetchData(); // Fetch data on initial load
  }, []); // Run only once

  const renderOptionItem = ({ item }) => (
    <View style={styles.optionItem}>
      <Text>{item.type} @ {item.strike}</Text>
      <Text>Price: ${item.price.toFixed(2)}</Text>
      <Text>Volume: {item.volume}</Text>
    </View>
  );

  const renderOrderItem = ({ item }) => (
    <View style={styles.orderItem}>
      <Text>Price: ${item.price.toFixed(2)}</Text>
      <Text>Size: {item.size}</Text>
    </View>
  );

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Trading Interface</Text>

      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          placeholder="Symbol (e.g., AAPL)"
          value={symbol}
          onChangeText={setSymbol}
          autoCapitalize="characters"
        />
        <TextInput
          style={styles.input}
          placeholder="Expiry (YYYY-MM-DD)"
          value={expiry}
          onChangeText={setExpiry}
        />
        <Button title="Load Data" onPress={fetchData} />
      </View>

      {error && <Text style={styles.errorText}>{error}</Text>}

      {/* Option Chain Section */}
      <View style={styles.sectionContainer}>
        <Text style={styles.sectionTitle}>Option Chain ({symbol} - {expiry})</Text>
        {loadingChain ? (
          <ActivityIndicator size="large" color="#007AFF" />
        ) : (
          <FlatList
            data={optionChain}
            renderItem={renderOptionItem}
            keyExtractor={(item, index) => `${item.type}-${item.strike}-${index}`}
            ListEmptyComponent={<Text>No option data available.</Text>}
          />
        )}
      </View>

      {/* Order Book Section */}
      <View style={styles.sectionContainer}>
        <Text style={styles.sectionTitle}>Order Book ({symbol})</Text>
        {loadingBook ? (
          <ActivityIndicator size="large" color="#007AFF" />
        ) : (
          <View style={styles.orderBookContainer}>
            <View style={styles.orderBookSide}>
              <Text style={styles.orderBookTitle}>Bids</Text>
              <FlatList
                data={orderBook.bids}
                renderItem={renderOrderItem}
                keyExtractor={(item, index) => `bid-${item.price}-${index}`}
                ListEmptyComponent={<Text>No bids.</Text>}
              />
            </View>
            <View style={styles.orderBookSide}>
              <Text style={styles.orderBookTitle}>Asks</Text>
              <FlatList
                data={orderBook.asks}
                renderItem={renderOrderItem}
                keyExtractor={(item, index) => `ask-${item.price}-${index}`}
                ListEmptyComponent={<Text>No asks.</Text>}
              />
            </View>
          </View>
        )}
      </View>

      {/* Trading Form Placeholder */}
      <View style={styles.sectionContainer}>
        <Text style={styles.sectionTitle}>Place Order</Text>
        <Text style={styles.placeholderText}>Trading form components will be added here.</Text>
        {/* Add components for selecting option, order type, quantity, price etc. */}
        <Button title="Submit Order (Placeholder)" onPress={() => alert('Order Submitted (Placeholder)')} />
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
  inputContainer: {
    marginBottom: 20,
    backgroundColor: '#FFFFFF',
    borderRadius: 10,
    padding: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  input: {
    height: 40,
    borderColor: '#E5E5EA',
    borderWidth: 1,
    marginBottom: 10,
    paddingHorizontal: 10,
    borderRadius: 5,
  },
  sectionContainer: {
    marginBottom: 25,
    backgroundColor: '#FFFFFF',
    borderRadius: 10,
    padding: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 15,
    color: '#007AFF',
  },
  optionItem: {
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
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
    marginBottom: 5,
    textAlign: 'center',
  },
  orderItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 5,
  },
  errorText: {
    color: '#FF3B30',
    textAlign: 'center',
    marginVertical: 10,
    fontSize: 16,
  },
  placeholderText: {
    textAlign: 'center',
    color: '#8E8E93',
    marginVertical: 15,
  },
});

export default TradingScreen;

