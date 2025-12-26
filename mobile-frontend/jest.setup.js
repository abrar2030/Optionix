// Mock AsyncStorage
jest.mock('@react-native-async-storage/async-storage', () => ({
    default: {
        getItem: jest.fn(() => Promise.resolve(null)),
        setItem: jest.fn(() => Promise.resolve()),
        removeItem: jest.fn(() => Promise.resolve()),
        clear: jest.fn(() => Promise.resolve()),
    },
}));

// Mock navigation
jest.mock('@react-navigation/native');
jest.mock('@react-navigation/stack');
jest.mock('@react-navigation/bottom-tabs');

// Mock expo modules
jest.mock('expo-status-bar');
jest.mock('expo-constants', () => ({
    default: {
        expoConfig: { extra: {} },
    },
}));

// Mock react-native modules
jest.mock('react-native-vector-icons/MaterialCommunityIcons', () => 'Icon');
jest.mock('react-native-paper');
jest.mock('react-native-gesture-handler');
jest.mock('react-native-safe-area-context', () => ({
    SafeAreaProvider: ({ children }) => children,
}));
