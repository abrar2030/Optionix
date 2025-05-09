import unittest
from app import app
import json

class TestBackendAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.auth_token = None

    def tearDown(self):
        # Clean up any test data
        if hasattr(self, 'test_user_id'):
            # Add cleanup code here if needed
            pass

    def get_auth_headers(self):
        return {'Authorization': f'Bearer {self.auth_token}'} if self.auth_token else {}

    def test_health_check(self):
        """Test the health check endpoint"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')

    def test_authentication(self):
        """Test authentication endpoints"""
        # Test login
        login_data = {
            'username': 'test_user',
            'password': 'test_password'
        }
        response = self.app.post('/auth/login', json=login_data)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('token', data)
        self.auth_token = data['token']

        # Test invalid login
        invalid_login = {
            'username': 'wrong_user',
            'password': 'wrong_password'
        }
        response = self.app.post('/auth/login', json=invalid_login)
        self.assertEqual(response.status_code, 401)

    def test_user_management(self):
        """Test user management endpoints"""
        # Test user registration with valid data
        user_data = {
            'username': 'new_user',
            'email': 'new_user@example.com',
            'password': 'password123'
        }
        response = self.app.post('/users/register', json=user_data)
        self.assertEqual(response.status_code, 201)

        # Test user registration with invalid email
        invalid_email_data = {
            'username': 'invalid_user',
            'email': 'invalid-email',
            'password': 'password123'
        }
        response = self.app.post('/users/register', json=invalid_email_data)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

        # Test user registration with weak password
        weak_password_data = {
            'username': 'weak_user',
            'email': 'weak@example.com',
            'password': '123'
        }
        response = self.app.post('/users/register', json=weak_password_data)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

        # Test get user profile with authentication
        response = self.app.get('/users/profile', headers=self.get_auth_headers())
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('username', data)

        # Test get user profile without authentication
        response = self.app.get('/users/profile')
        self.assertEqual(response.status_code, 401)

    def test_error_handling(self):
        """Test error handling"""
        # Test 404
        response = self.app.get('/nonexistent_endpoint')
        self.assertEqual(response.status_code, 404)

        # Test invalid JSON
        response = self.app.post('/auth/login', data='invalid json')
        self.assertEqual(response.status_code, 400)

        # Test missing required fields
        response = self.app.post('/users/register', json={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main() 