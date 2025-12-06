import json
import unittest
from app import app


class TestBackendAPI(unittest.TestCase):

    def setUp(self) -> Any:
        self.app = app.test_client()
        self.app.testing = True
        self.auth_token = None

    def tearDown(self) -> Any:
        if hasattr(self, "test_user_id"):
            pass

    def get_auth_headers(self) -> Any:
        return {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}

    def test_health_check(self) -> Any:
        """Test the health check endpoint"""
        response = self.app.get("/health")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "healthy")

    def test_authentication(self) -> Any:
        """Test authentication endpoints"""
        login_data = {"username": "test_user", "password": "test_password"}
        response = self.app.post("/auth/login", json=login_data)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("token", data)
        self.auth_token = data["token"]
        invalid_login = {"username": "wrong_user", "password": "wrong_password"}
        response = self.app.post("/auth/login", json=invalid_login)
        self.assertEqual(response.status_code, 401)

    def test_user_management(self) -> Any:
        """Test user management endpoints"""
        user_data = {
            "username": "new_user",
            "email": "new_user@example.com",
            "password": "password123",
        }
        response = self.app.post("/users/register", json=user_data)
        self.assertEqual(response.status_code, 201)
        invalid_email_data = {
            "username": "invalid_user",
            "email": "invalid-email",
            "password": "password123",
        }
        response = self.app.post("/users/register", json=invalid_email_data)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)
        weak_password_data = {
            "username": "weak_user",
            "email": "weak@example.com",
            "password": "123",
        }
        response = self.app.post("/users/register", json=weak_password_data)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)
        response = self.app.get("/users/profile", headers=self.get_auth_headers())
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("username", data)
        response = self.app.get("/users/profile")
        self.assertEqual(response.status_code, 401)

    def test_error_handling(self) -> Any:
        """Test error handling"""
        response = self.app.get("/nonexistent_endpoint")
        self.assertEqual(response.status_code, 404)
        response = self.app.post("/auth/login", data="invalid json")
        self.assertEqual(response.status_code, 400)
        response = self.app.post("/users/register", json={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
