// SPDX-License-Identifier: MIT  
pragma solidity ^0.8.0;  
import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";  

contract FuturesContract {  
    AggregatorV3Interface internal priceFeed;  
    address public owner;  
    uint256 public marginRequirement;  

    struct Position {  
        address trader;  
        uint256 size;  
        bool isLong;  
        uint256 entryPrice;  
    }  

    mapping(address => Position) public positions;  

    constructor(address _priceFeed) {  
        priceFeed = AggregatorV3Interface(_priceFeed);  
        owner = msg.sender;  
        marginRequirement = 10; // 10%  
    }  

    function openPosition(uint256 _size, bool _isLong) external payable {  
        require(msg.value >= (_size * marginRequirement) / 100, "Insufficient margin");  
        positions[msg.sender] = Position(msg.sender, _size, _isLong, getLatestPrice());  
    }  

    function getLatestPrice() public view returns (uint256) {  
        (, int256 price,,,) = priceFeed.latestRoundData();  
        return uint256(price * 1e10); // Chainlink 8-decimals adjustment  
    }  

    function liquidatePosition(address _trader) external {  
        Position memory pos = positions[_trader];  
        uint256 currentPrice = getLatestPrice();  
        uint256 pnl = pos.isLong ? currentPrice - pos.entryPrice : pos.entryPrice - currentPrice;  
        require(pnl > pos.size * marginRequirement / 100, "Position safe");  
        delete positions[_trader];  
    }  
}  