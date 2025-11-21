// Mock the AsyncStorage
jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

// Mock the navigation
jest.mock("@react-navigation/native", () => {
  const actualNav = jest.requireActual("@react-navigation/native");
  return {
    ...actualNav,
    useNavigation: () => ({
      navigate: jest.fn(),
      goBack: jest.fn(),
    }),
  };
});

// Mock the Expo modules
jest.mock("expo", () => ({
  ...jest.requireActual("expo"),
  Linking: {
    makeUrl: jest.fn(),
    parse: jest.fn(),
  },
}));

// Silence the warning: Animated: `useNativeDriver` is not supported
jest.mock("react-native/Libraries/Animated/NativeAnimatedHelper");

// Mock the StatusBar
jest.mock("expo-status-bar", () => ({
  StatusBar: {
    setBarStyle: jest.fn(),
  },
}));
