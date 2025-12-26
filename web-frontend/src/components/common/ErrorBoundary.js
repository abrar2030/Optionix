import React from 'react';
import styled from 'styled-components';
import { FiAlertTriangle } from 'react-icons/fi';

const ErrorContainer = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 400px;
    padding: 40px;
    background-color: ${(props) => props.theme.colors.cardBg};
    border-radius: 8px;
    border: 1px solid ${(props) => props.theme.colors.border};
    margin: 20px;
`;

const ErrorIcon = styled(FiAlertTriangle)`
    font-size: 64px;
    color: ${(props) => props.theme.colors.danger};
    margin-bottom: 20px;
`;

const ErrorTitle = styled.h2`
    color: ${(props) => props.theme.colors.textPrimary};
    font-size: 24px;
    margin-bottom: 12px;
`;

const ErrorMessage = styled.p`
    color: ${(props) => props.theme.colors.textSecondary};
    font-size: 16px;
    text-align: center;
    max-width: 500px;
    margin-bottom: 24px;
`;

const ErrorDetails = styled.details`
    color: ${(props) => props.theme.colors.textSecondary};
    font-size: 14px;
    font-family: monospace;
    max-width: 600px;
    margin-bottom: 24px;
    cursor: pointer;

    summary {
        margin-bottom: 8px;
        font-weight: 500;
    }

    pre {
        background-color: ${(props) => props.theme.colors.backgroundDark};
        padding: 12px;
        border-radius: 4px;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
`;

const ReloadButton = styled.button`
    background-color: ${(props) => props.theme.colors.primary};
    color: white;
    padding: 12px 24px;
    border-radius: 4px;
    border: none;
    font-size: 16px;
    cursor: pointer;
    transition: background-color 0.2s;

    &:hover {
        background-color: ${(props) => props.theme.colors.primaryDark};
    }
`;

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            hasError: false,
            error: null,
            errorInfo: null,
        };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true };
    }

    componentDidCatch(error, errorInfo) {
        console.error('Error boundary caught an error:', error, errorInfo);
        this.setState({
            error,
            errorInfo,
        });

        // Log error to monitoring service if available
        if (window.errorLogger) {
            window.errorLogger.log(error, errorInfo);
        }
    }

    handleReload = () => {
        this.setState({ hasError: false, error: null, errorInfo: null });
        window.location.reload();
    };

    render() {
        if (this.state.hasError) {
            return (
                <ErrorContainer>
                    <ErrorIcon />
                    <ErrorTitle>Oops! Something went wrong</ErrorTitle>
                    <ErrorMessage>
                        We're sorry for the inconvenience. An unexpected error has occurred. Please
                        try reloading the page.
                    </ErrorMessage>
                    {process.env.NODE_ENV === 'development' && this.state.error && (
                        <ErrorDetails>
                            <summary>Error Details (Development Mode)</summary>
                            <pre>
                                <strong>Error:</strong> {this.state.error.toString()}
                                {'\n\n'}
                                <strong>Stack Trace:</strong>
                                {'\n'}
                                {this.state.errorInfo?.componentStack}
                            </pre>
                        </ErrorDetails>
                    )}
                    <ReloadButton onClick={this.handleReload}>Reload Page</ReloadButton>
                </ErrorContainer>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
