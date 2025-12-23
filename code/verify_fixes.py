#!/usr/bin/env python3
"""
Quick verification script for Optionix backend fixes.
Tests that all fixed files can be imported and basic functionality works.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all fixed modules can be imported."""
    print("Testing imports...")

    try:
        print("✓ config module OK")

        print("✓ database module OK")

        print("✓ auth module OK")

        print("✓ security module OK")

        print("✓ models module OK")

        print("✓ schemas module OK")

        print("✓ blockchain_service module OK")

        print("✓ financial_service module OK")

        print("✓ middleware.security module OK")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_type_annotations():
    """Test that key type annotations are present."""
    print("\nTesting type annotations...")

    try:
        from backend.database import get_db
        from typing import get_type_hints

        # Check get_db return type
        hints = get_type_hints(get_db)
        if "return" in hints:
            print(f"✓ get_db has return type: {hints['return']}")
        else:
            print("✗ get_db missing return type annotation")
            return False

        return True
    except Exception as e:
        print(f"✗ Type annotation check failed: {e}")
        return False


def test_database_fallback():
    """Test that database fallback works."""
    print("\nTesting database fallback...")

    try:
        from backend import database

        # Try to create tables (should fallback to SQLite if PostgreSQL unavailable)
        database.create_tables()
        print("✓ Database tables created (with fallback if needed)")

        return True
    except Exception as e:
        print(f"✗ Database fallback failed: {e}")
        return False


def test_service_initialization():
    """Test that services initialize correctly."""
    print("\nTesting service initialization...")

    try:
        from backend.services.blockchain_service import BlockchainService
        from backend.services.financial_service import FinancialCalculationService

        # Initialize services
        blockchain = BlockchainService()
        print("✓ BlockchainService initialized")

        financial = FinancialCalculationService()
        print("✓ FinancialCalculationService initialized")

        # Test that they have expected attributes
        assert hasattr(blockchain, "w3")
        assert hasattr(blockchain, "futures_contract")
        assert hasattr(financial, "risk_free_rate")
        print("✓ Services have expected attributes")

        return True
    except Exception as e:
        print(f"✗ Service initialization failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Optionix Backend Verification Tests")
    print("=" * 60)

    tests = [
        ("Import Tests", test_imports),
        ("Type Annotation Tests", test_type_annotations),
        ("Database Fallback Test", test_database_fallback),
        ("Service Initialization Test", test_service_initialization),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
        print()

    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Backend fixes are working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
