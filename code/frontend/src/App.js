import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import styled, { ThemeProvider } from 'styled-components';
import { AppProvider } from './utils/AppContext';

// Pages
import Dashboard from './pages/Dashboard';
import Trading from './pages/Trading';
import Portfolio from './pages/Portfolio';
import Analytics from './pages/Analytics';

// Components
import Navbar from './components/common/Navbar';
import Sidebar from './components/common/Sidebar';
import Footer from './components/common/Footer';

// Theme
const theme = {
  colors: {
    primary: '#2962ff',
    primaryDark: '#0039cb',
    primaryLight: '#768fff',
    secondary: '#ff6d00',
    secondaryDark: '#c43c00',
    secondaryLight: '#ff9e40',
    backgroundDark: '#131722',
    backgroundLight: '#1e222d',
    textPrimary: '#ffffff',
    textSecondary: '#b2b5be',
    success: '#26a69a',
    danger: '#ef5350',
    warning: '#ffca28',
    info: '#42a5f5',
    border: '#2a2e39',
    cardBg: '#1e222d',
  },
  breakpoints: {
    mobile: '576px',
    tablet: '768px',
    desktop: '992px',
    wide: '1200px',
  }
};

const AppContainer = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: ${props => props.theme.colors.backgroundDark};
  color: ${props => props.theme.colors.textPrimary};
`;

const MainContent = styled.main`
  display: flex;
  flex: 1;
`;

const ContentArea = styled.div`
  flex: 1;
  padding: 20px;
  margin-left: 240px;
  
  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    margin-left: 0;
    padding-top: 70px;
  }
`;

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <AppProvider>
      <ThemeProvider theme={theme}>
        <Router>
          <AppContainer>
            <Navbar toggleSidebar={toggleSidebar} />
            <MainContent>
              <Sidebar isOpen={sidebarOpen} />
              <ContentArea>
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/trading" element={<Trading />} />
                  <Route path="/portfolio" element={<Portfolio />} />
                  <Route path="/analytics" element={<Analytics />} />
                </Routes>
              </ContentArea>
            </MainContent>
            <Footer />
          </AppContainer>
        </Router>
      </ThemeProvider>
    </AppProvider>
  );
}

export default App;
