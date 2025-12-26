import React from 'react';
import { render, waitFor } from '@testing-library/react-native';
import { Provider as PaperProvider } from 'react-native-paper';
import DashboardScreen from '../screens/DashboardScreen';
import TradingScreen from '../screens/TradingScreen';
import PortfolioScreen from '../screens/PortfolioScreen';
import AnalyticsScreen from '../screens/AnalyticsScreen';
import WatchlistScreen from '../screens/WatchlistScreen';

jest.mock('../services/api');

const renderWithProviders = (component) => {
    return render(<PaperProvider>{component}</PaperProvider>);
};

describe('DashboardScreen', () => {
    it('renders dashboard title', () => {
        const { getByText } = renderWithProviders(<DashboardScreen />);
        expect(getByText('Market Dashboard')).toBeTruthy();
    });

    it('displays market overview data', async () => {
        const { getByText } = renderWithProviders(<DashboardScreen />);

        await waitFor(
            () => {
                expect(getByText(/Market Status/)).toBeTruthy();
            },
            { timeout: 3000 },
        );
    });
});

describe('TradingScreen', () => {
    it('renders trading interface', () => {
        const { getByText } = renderWithProviders(<TradingScreen />);
        expect(getByText('Trading Interface')).toBeTruthy();
    });

    it('displays option chain data table', async () => {
        const { getByText } = renderWithProviders(<TradingScreen />);

        await waitFor(
            () => {
                expect(getByText(/Option Chain/)).toBeTruthy();
            },
            { timeout: 3000 },
        );
    });
});

describe('PortfolioScreen', () => {
    it('renders portfolio title', () => {
        const { getByText } = renderWithProviders(<PortfolioScreen />);
        expect(getByText('Portfolio Management')).toBeTruthy();
    });

    it('displays portfolio summary', async () => {
        const { getByText } = renderWithProviders(<PortfolioScreen />);

        await waitFor(
            () => {
                expect(getByText('Summary')).toBeTruthy();
            },
            { timeout: 3000 },
        );
    });
});

describe('AnalyticsScreen', () => {
    it('renders analytics title', () => {
        const { getByText } = renderWithProviders(<AnalyticsScreen />);
        expect(getByText('Analytics & Insights')).toBeTruthy();
    });

    it('displays risk assessment', async () => {
        const { getByText } = renderWithProviders(<AnalyticsScreen />);

        await waitFor(
            () => {
                expect(getByText('Risk Assessment')).toBeTruthy();
            },
            { timeout: 3000 },
        );
    });
});

describe('WatchlistScreen', () => {
    it('renders watchlist title', () => {
        const { getByText } = renderWithProviders(<WatchlistScreen />);
        expect(getByText('My Watchlist')).toBeTruthy();
    });

    it('displays add symbol form', () => {
        const { getByText } = renderWithProviders(<WatchlistScreen />);
        expect(getByText('Add Symbol')).toBeTruthy();
    });
});
