import React from 'react';
import { Alert } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import { IconButton } from 'react-native-paper';

import DashboardScreen from '../screens/DashboardScreen';
import TradingScreen from '../screens/TradingScreen';
import PortfolioScreen from '../screens/PortfolioScreen';
import AnalyticsScreen from '../screens/AnalyticsScreen';
import WatchlistScreen from '../screens/WatchlistScreen';
import { useAuth } from '../context/AuthContext';

const Tab = createBottomTabNavigator();

const AppNavigator = () => {
    const { logout, user } = useAuth();

    const handleLogout = () => {
        Alert.alert('Logout', 'Are you sure you want to logout?', [
            {
                text: 'Cancel',
                style: 'cancel',
            },
            {
                text: 'Logout',
                onPress: async () => {
                    try {
                        await logout();
                    } catch (error) {
                        console.error('Logout error:', error);
                    }
                },
                style: 'destructive',
            },
        ]);
    };

    return (
        <Tab.Navigator
            screenOptions={({ route }) => ({
                tabBarIcon: ({ focused, color, size }) => {
                    let iconName;

                    if (route.name === 'Dashboard') {
                        iconName = focused ? 'view-dashboard' : 'view-dashboard-outline';
                    } else if (route.name === 'Trading') {
                        iconName = focused ? 'swap-horizontal-bold' : 'swap-horizontal';
                    } else if (route.name === 'Watchlist') {
                        iconName = focused ? 'star' : 'star-outline';
                    } else if (route.name === 'Portfolio') {
                        iconName = focused ? 'briefcase' : 'briefcase-outline';
                    } else if (route.name === 'Analytics') {
                        iconName = focused ? 'chart-line' : 'chart-line-variant';
                    }

                    return <MaterialCommunityIcons name={iconName} size={size} color={color} />;
                },
                tabBarActiveTintColor: '#007AFF',
                tabBarInactiveTintColor: 'gray',
                headerShown: true,
                headerRight: () => <IconButton icon="logout" size={24} onPress={handleLogout} />,
                headerTitle: route.name,
            })}
        >
            <Tab.Screen
                name="Dashboard"
                component={DashboardScreen}
                options={{
                    headerTitle: `Welcome${user?.full_name ? `, ${user.full_name.split(' ')[0]}` : ''}`,
                }}
            />
            <Tab.Screen name="Trading" component={TradingScreen} />
            <Tab.Screen name="Watchlist" component={WatchlistScreen} />
            <Tab.Screen name="Portfolio" component={PortfolioScreen} />
            <Tab.Screen name="Analytics" component={AnalyticsScreen} />
        </Tab.Navigator>
    );
};

export default AppNavigator;
