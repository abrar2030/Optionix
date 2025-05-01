import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons'; // Import icons

import DashboardScreen from '../screens/DashboardScreen';
import TradingScreen from '../screens/TradingScreen';
import PortfolioScreen from '../screens/PortfolioScreen';
import AnalyticsScreen from '../screens/AnalyticsScreen';
import WatchlistScreen from '../screens/WatchlistScreen'; // Import Watchlist screen

const Tab = createBottomTabNavigator();

const AppNavigator = () => {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused, color, size }) => {
            let iconName;

            if (route.name === 'Dashboard') {
              iconName = focused ? 'view-dashboard' : 'view-dashboard-outline';
            } else if (route.name === 'Trading') {
              iconName = focused ? 'swap-horizontal-bold' : 'swap-horizontal';
            } else if (route.name === 'Watchlist') { // Add icon for Watchlist
              iconName = focused ? 'star' : 'star-outline';
            } else if (route.name === 'Portfolio') {
              iconName = focused ? 'briefcase' : 'briefcase-outline';
            } else if (route.name === 'Analytics') {
              iconName = focused ? 'chart-line' : 'chart-line-variant';
            }

            // You can return any component that you like here!
            return <MaterialCommunityIcons name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: '#007AFF', // Example active color
          tabBarInactiveTintColor: 'gray',
          headerShown: false, // Optionally hide headers for tab screens
        })}
      >
        <Tab.Screen name="Dashboard" component={DashboardScreen} />
        <Tab.Screen name="Trading" component={TradingScreen} />
        <Tab.Screen name="Watchlist" component={WatchlistScreen} /> {/* Add Watchlist screen */}
        <Tab.Screen name="Portfolio" component={PortfolioScreen} />
        <Tab.Screen name="Analytics" component={AnalyticsScreen} />
      </Tab.Navigator>
    </NavigationContainer>
  );
};

export default AppNavigator;
