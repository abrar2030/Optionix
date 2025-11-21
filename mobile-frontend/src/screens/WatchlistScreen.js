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
  Button,
  TextInput,
} from "react-native-paper";

const WatchlistScreen = () => {
  const [watchlist, setWatchlist] = useState([
    { symbol: "MSFT", price: 420.55, change: "+1.1%" },
    { symbol: "NVDA", price: 950.02, change: "-0.5%" },
    { symbol: "AMZN", price: 180.3, change: "+0.8%" },
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [newSymbol, setNewSymbol] = useState("");
  const theme = useTheme();

  // Placeholder function to add symbol
  const addSymbolToWatchlist = () => {
    if (newSymbol.trim() === "") return;
    // In a real app, you'd fetch data for the new symbol
    // For now, add with placeholder price/change
    const newEntry = {
      symbol: newSymbol.toUpperCase(),
      price: (Math.random() * 1000).toFixed(2),
      change: `${(Math.random() * 2 - 1).toFixed(1)}%`,
    };
    setWatchlist([...watchlist, newEntry]);
    setNewSymbol(""); // Clear input
  };

  // Placeholder function to remove symbol
  const removeSymbol = (symbolToRemove) => {
    setWatchlist(watchlist.filter((item) => item.symbol !== symbolToRemove));
  };

  const renderChange = (change) => {
    const isPositive = change.startsWith("+");
    return (
      <PaperText
        style={isPositive ? styles.positiveChange : styles.negativeChange}
      >
        {change}
      </PaperText>
    );
  };

  const renderWatchlistItem = ({ item }) => (
    <List.Item
      title={`${item.symbol}: $${item.price}`}
      right={() => renderChange(item.change)}
      titleStyle={styles.listItemTitle}
      // Add a button or swipe action to remove
      left={() => (
        <Button icon="delete" onPress={() => removeSymbol(item.symbol)} compact>
          Remove
        </Button>
      )}
    />
  );

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: theme.colors.background }]}
    >
      <Title style={styles.title}>My Watchlist</Title>

      {error && <Paragraph style={styles.errorText}>{error}</Paragraph>}

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
          />
          <Button
            mode="contained"
            onPress={addSymbolToWatchlist}
            style={styles.button}
          >
            Add to Watchlist
          </Button>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Title title="Tracked Symbols" />
        <Card.Content>
          {loading ? (
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
    fontWeight: "bold",
    marginBottom: 20,
    textAlign: "center",
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
  listItemTitle: {
    fontSize: 16,
  },
  positiveChange: {
    color: "#34C759",
    fontWeight: "bold",
    fontSize: 16,
    alignSelf: "center",
  },
  negativeChange: {
    color: "#FF3B30",
    fontWeight: "bold",
    fontSize: 16,
    alignSelf: "center",
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

export default WatchlistScreen;
