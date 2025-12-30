# Example: Blockchain Integration

Create and manage options on the blockchain.

## Create Option Contract

```python
from code.backend.services.blockchain_service import BlockchainService

blockchain = BlockchainService()

contract = blockchain.create_option_contract(
    writer_address='0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1',
    holder_address='0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed',
    option_type='call',
    strike_price=100.0,
    expiration=1704067200,
    premium=5.0,
    collateral=100.0
)

print(f"Contract ID: {contract['contract_id']}")
print(f"TX Hash: {contract['tx_hash']}")
```

## Exercise Option

```python
result = blockchain.exercise_option(
    contract_id=contract['contract_id'],
    holder_address='0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed'
)

print(f"Exercise successful: {result['success']}")
print(f"Payout: ${result['payout']:.2f}")
```
