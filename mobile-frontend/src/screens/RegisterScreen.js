import React, { useState } from 'react';
import { StyleSheet, View, KeyboardAvoidingView, Platform, ScrollView, Alert } from 'react-native';
import {
    TextInput,
    Button,
    Title,
    Paragraph,
    useTheme,
    HelperText,
    Checkbox,
} from 'react-native-paper';
import { useAuth } from '../context/AuthContext';

const RegisterScreen = ({ navigation }) => {
    const [formData, setFormData] = useState({
        fullName: '',
        email: '',
        password: '',
        confirmPassword: '',
    });
    const [acceptedTerms, setAcceptedTerms] = useState(false);
    const [secureTextEntry, setSecureTextEntry] = useState(true);
    const [secureConfirmEntry, setSecureConfirmEntry] = useState(true);
    const [errors, setErrors] = useState({});
    const [isSubmitting, setIsSubmitting] = useState(false);

    const { register, error: authError } = useAuth();
    const theme = useTheme();

    const handleChange = (field, value) => {
        setFormData({ ...formData, [field]: value });
        // Clear error for this field
        if (errors[field]) {
            setErrors({ ...errors, [field]: null });
        }
    };

    const validate = () => {
        const newErrors = {};

        if (!formData.fullName.trim()) {
            newErrors.fullName = 'Full name is required';
        }

        if (!formData.email.trim()) {
            newErrors.email = 'Email is required';
        } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
            newErrors.email = 'Email is invalid';
        }

        if (!formData.password) {
            newErrors.password = 'Password is required';
        } else if (formData.password.length < 8) {
            newErrors.password = 'Password must be at least 8 characters';
        } else if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(formData.password)) {
            newErrors.password = 'Password must contain uppercase, lowercase, and number';
        }

        if (!formData.confirmPassword) {
            newErrors.confirmPassword = 'Please confirm your password';
        } else if (formData.password !== formData.confirmPassword) {
            newErrors.confirmPassword = 'Passwords do not match';
        }

        if (!acceptedTerms) {
            newErrors.terms = 'You must accept the terms and conditions';
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleRegister = async () => {
        if (!validate()) {
            return;
        }

        setIsSubmitting(true);
        try {
            const result = await register({
                full_name: formData.fullName,
                email: formData.email,
                password: formData.password,
                data_retention_consent: acceptedTerms,
                data_processing_consent: acceptedTerms,
                marketing_consent: false,
            });

            if (result.requiresLogin) {
                Alert.alert('Registration Successful', 'Please sign in with your credentials.', [
                    {
                        text: 'OK',
                        onPress: () => navigation.navigate('Login'),
                    },
                ]);
            }
            // If token was provided, AuthContext will handle navigation
        } catch (err) {
            Alert.alert(
                'Registration Failed',
                err.message || 'Please check your information and try again.',
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
                    <Title style={styles.title}>Create Account</Title>
                    <Paragraph style={styles.subtitle}>Join Optionix today</Paragraph>

                    <TextInput
                        label="Full Name"
                        value={formData.fullName}
                        onChangeText={(value) => handleChange('fullName', value)}
                        autoCapitalize="words"
                        autoComplete="name"
                        style={styles.input}
                        mode="outlined"
                        error={!!errors.fullName}
                        disabled={isSubmitting}
                        left={<TextInput.Icon icon="account" />}
                    />
                    {errors.fullName && (
                        <HelperText type="error" visible={!!errors.fullName}>
                            {errors.fullName}
                        </HelperText>
                    )}

                    <TextInput
                        label="Email"
                        value={formData.email}
                        onChangeText={(value) => handleChange('email', value)}
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
                        value={formData.password}
                        onChangeText={(value) => handleChange('password', value)}
                        secureTextEntry={secureTextEntry}
                        autoComplete="password-new"
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

                    <TextInput
                        label="Confirm Password"
                        value={formData.confirmPassword}
                        onChangeText={(value) => handleChange('confirmPassword', value)}
                        secureTextEntry={secureConfirmEntry}
                        autoComplete="password-new"
                        style={styles.input}
                        mode="outlined"
                        error={!!errors.confirmPassword}
                        disabled={isSubmitting}
                        left={<TextInput.Icon icon="lock-check" />}
                        right={
                            <TextInput.Icon
                                icon={secureConfirmEntry ? 'eye' : 'eye-off'}
                                onPress={() => setSecureConfirmEntry(!secureConfirmEntry)}
                            />
                        }
                    />
                    {errors.confirmPassword && (
                        <HelperText type="error" visible={!!errors.confirmPassword}>
                            {errors.confirmPassword}
                        </HelperText>
                    )}

                    <View style={styles.checkboxContainer}>
                        <Checkbox
                            status={acceptedTerms ? 'checked' : 'unchecked'}
                            onPress={() => setAcceptedTerms(!acceptedTerms)}
                            disabled={isSubmitting}
                        />
                        <Paragraph style={styles.checkboxLabel}>
                            I accept the Terms and Conditions and Privacy Policy
                        </Paragraph>
                    </View>
                    {errors.terms && (
                        <HelperText type="error" visible={!!errors.terms}>
                            {errors.terms}
                        </HelperText>
                    )}

                    {authError && (
                        <HelperText type="error" visible={!!authError} style={styles.authError}>
                            {authError}
                        </HelperText>
                    )}

                    <Button
                        mode="contained"
                        onPress={handleRegister}
                        style={styles.button}
                        disabled={isSubmitting}
                        loading={isSubmitting}
                    >
                        {isSubmitting ? 'Creating Account...' : 'Sign Up'}
                    </Button>

                    <Button
                        mode="text"
                        onPress={() => navigation.navigate('Login')}
                        style={styles.textButton}
                        disabled={isSubmitting}
                    >
                        Already have an account? Sign In
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
    checkboxContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        marginTop: 10,
        marginBottom: 5,
    },
    checkboxLabel: {
        flex: 1,
        marginLeft: 8,
        fontSize: 14,
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

export default RegisterScreen;
