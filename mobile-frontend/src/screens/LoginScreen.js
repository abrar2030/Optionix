import React, { useState } from 'react';
import { StyleSheet, View, KeyboardAvoidingView, Platform, ScrollView, Alert } from 'react-native';
import {
    TextInput,
    Button,
    Title,
    Paragraph,
    useTheme,
    HelperText,
    ActivityIndicator,
} from 'react-native-paper';
import { useAuth } from '../context/AuthContext';

const LoginScreen = ({ navigation }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [secureTextEntry, setSecureTextEntry] = useState(true);
    const [errors, setErrors] = useState({});
    const [isSubmitting, setIsSubmitting] = useState(false);

    const { login, error: authError } = useAuth();
    const theme = useTheme();

    const validate = () => {
        const newErrors = {};

        if (!email.trim()) {
            newErrors.email = 'Email is required';
        } else if (!/\S+@\S+\.\S+/.test(email)) {
            newErrors.email = 'Email is invalid';
        }

        if (!password) {
            newErrors.password = 'Password is required';
        } else if (password.length < 6) {
            newErrors.password = 'Password must be at least 6 characters';
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleLogin = async () => {
        if (!validate()) {
            return;
        }

        setIsSubmitting(true);
        try {
            await login(email, password);
            // Navigation is handled by App.js based on auth state
        } catch (err) {
            Alert.alert(
                'Login Failed',
                err.message || 'Please check your credentials and try again.',
            );
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            style={styles.container}
        >
            <ScrollView
                contentContainerStyle={styles.scrollContent}
                keyboardShouldPersistTaps="handled"
            >
                <View style={styles.formContainer}>
                    <Title style={styles.title}>Welcome to Optionix</Title>
                    <Paragraph style={styles.subtitle}>Sign in to continue</Paragraph>

                    <TextInput
                        label="Email"
                        value={email}
                        onChangeText={setEmail}
                        autoCapitalize="none"
                        keyboardType="email-address"
                        autoComplete="email"
                        style={styles.input}
                        mode="outlined"
                        error={!!errors.email}
                        disabled={isSubmitting}
                        left={<TextInput.Icon icon="email" />}
                    />
                    {errors.email && (
                        <HelperText type="error" visible={!!errors.email}>
                            {errors.email}
                        </HelperText>
                    )}

                    <TextInput
                        label="Password"
                        value={password}
                        onChangeText={setPassword}
                        secureTextEntry={secureTextEntry}
                        autoComplete="password"
                        style={styles.input}
                        mode="outlined"
                        error={!!errors.password}
                        disabled={isSubmitting}
                        left={<TextInput.Icon icon="lock" />}
                        right={
                            <TextInput.Icon
                                icon={secureTextEntry ? 'eye' : 'eye-off'}
                                onPress={() => setSecureTextEntry(!secureTextEntry)}
                            />
                        }
                    />
                    {errors.password && (
                        <HelperText type="error" visible={!!errors.password}>
                            {errors.password}
                        </HelperText>
                    )}

                    {authError && (
                        <HelperText type="error" visible={!!authError} style={styles.authError}>
                            {authError}
                        </HelperText>
                    )}

                    <Button
                        mode="contained"
                        onPress={handleLogin}
                        style={styles.button}
                        disabled={isSubmitting}
                        loading={isSubmitting}
                    >
                        {isSubmitting ? 'Signing In...' : 'Sign In'}
                    </Button>

                    <Button
                        mode="text"
                        onPress={() => navigation.navigate('Register')}
                        style={styles.textButton}
                        disabled={isSubmitting}
                    >
                        Don't have an account? Sign Up
                    </Button>

                    <Button
                        mode="text"
                        onPress={() =>
                            Alert.alert('Password Reset', 'Password reset feature coming soon!')
                        }
                        style={styles.textButton}
                        disabled={isSubmitting}
                    >
                        Forgot Password?
                    </Button>
                </View>
            </ScrollView>
        </KeyboardAvoidingView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    scrollContent: {
        flexGrow: 1,
        justifyContent: 'center',
        padding: 20,
    },
    formContainer: {
        width: '100%',
        maxWidth: 400,
        alignSelf: 'center',
    },
    title: {
        fontSize: 28,
        fontWeight: 'bold',
        textAlign: 'center',
        marginBottom: 10,
    },
    subtitle: {
        fontSize: 16,
        textAlign: 'center',
        marginBottom: 30,
        opacity: 0.7,
    },
    input: {
        marginBottom: 5,
    },
    button: {
        marginTop: 20,
        paddingVertical: 8,
    },
    textButton: {
        marginTop: 10,
    },
    authError: {
        textAlign: 'center',
        marginTop: 10,
    },
});

export default LoginScreen;
